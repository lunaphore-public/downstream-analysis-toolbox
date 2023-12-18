from scipy.signal import find_peaks, peak_widths
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter1d
from scipy.stats.mstats import winsorize
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from anndata import AnnData
from tqdm import tqdm
import squidpy as sq
import numpy as np
import tifffile
import numpy
import copy


def get_sorted_clusters(cluster_names: numpy.ndarray,
                        expression_table: numpy.ndarray):
    """
    Sort the unique cluster names based on the average expression per all markers.

    Parameters:
        - cluster_names (numpy.ndarray): Input array containing cluster names [cells]
        - expression_table (numpy.ndarray): Input array containing expression table [cells, markers]

    Return:
        tuple [numpy.ndarray, numpy.ndarray] Sorted cluster, sorted av_expression values

    """
    # Get unique clusters
    unique_clusters = np.unique(cluster_names)

    # Get average expression per all markers
    av_expression = np.array([np.mean(expression_table[cluster_names == u_cluster]) for u_cluster in unique_clusters])

    # Get the indices that would sort the expressions in descending order
    sorted_indices = np.argsort(av_expression)[::-1]

    # Use the sorted indices to reorder both unique_clusters and expressions
    sorted_unique_clusters = unique_clusters[sorted_indices]
    sorted_av_expression = av_expression[sorted_indices]

    return sorted_unique_clusters, sorted_av_expression


def normalize_median(markers_counts: numpy.ndarray) -> numpy.ndarray:
    """
    Normalize each column of the input array using median normalization and division by std.
    This is also known as a robust normalization technique.

    Parameters:
        - markers_counts (numpy.ndarray): Input array containing marker counts.

    Returns:
    - numpy.ndarray: Normalized array.
    """

    # Iterate over columns and apply median normalization in-place
    for i, column in enumerate(markers_counts.T):
        # Apply median normalization in-place
        markers_counts[:, i] = (column - np.nanmedian(column)) / np.nanstd(column)

    return markers_counts


def normalize_z_scaling(markers_counts: numpy.ndarray) -> numpy.ndarray:
    """
    Normalize each column of the input array using Z-scaling.

    Parameters:
        - markers_counts (numpy.ndarray): Input array containing marker counts.

    Returns:
        -  numpy.ndarray: Normalized array.
    """

    # Iterate over columns and apply z normalization in-place
    for i, column in enumerate(markers_counts.T):
        # Apply z normalization in-place
        markers_counts[:, i] = (column - np.nanmean(column)) / np.nanstd(column)

    return markers_counts


def winsorize_column_wise(markers_counts: numpy.ndarray,
                          limits: list = None) -> numpy.ndarray:
    """
    Winsorize each column of the input array using provided percentiles.

    Parameters:
        - markers_counts (numpy.ndarray): Input array containing marker counts.
        - limits (list): limits to be used for winsorization

    Returns:
        - numpy.ndarray: Winsorized array.
    """
    if limits is None:
        limits = [0, 0]

    # Iterate over columns and winsorize
    for i, column in enumerate(markers_counts.T):
        # Apply winsorization in-place
        markers_counts[:, i] = winsorize(column, limits)

    return markers_counts


def get_img_array(img: numpy.ndarray,
                  crop_size: int,
                  coords: numpy.ndarray,
                  rescale_img: bool = True,
                  px_size: float = 1.0) -> numpy.ndarray:
    """
    Get crops from an image given a list of X,Y coordinates.

    Parameters:
        - img (numpy.ndarray): Input img to make crops from
        - crop_size (int): Size of a square crop
        - coords (numpy.ndarray): X,Y coords of centers for each crop in um
        - rescale_img (bool): Rescale image intensity or not
        - px_size (float): Size of the pixel in img.

    Returns:
        - numpy.ndarray of image crops: Normalized array.
    """

    # Define nested functions
    def get_image_mod():
        # Get crop coordinates
        Xmin, Xmax, Ymin, Ymax = patch_maker_fixed()
        # Extract the region of interest from the padded image
        img_return = img_padded[Ymin+padding:Ymax+padding, Xmin+padding:Xmax+padding]
        # Make sure that the image is not bigger than the size given
        img_return = np.squeeze(img_return)
        img_return = img_return[0:crop_size, 0:crop_size]
        return img_return

    def patch_maker_fixed():
        # Get center coordinates
        X_px = coord[0] / px_size
        Y_px = coord[1] / px_size
        return int(X_px - crop_size / 2), int(X_px + crop_size / 2), int(Y_px - crop_size / 2), int(Y_px + crop_size / 2)

    # Main part
    img_array = []
    # Pad the main image to make sure no CROPS will be outside
    padding = crop_size // 2
    # Pad the image
    img_padded = np.pad(img, pad_width=padding, mode='constant', constant_values=0)
    # Iterate through coordinates and add the crops to a list
    for coord in coords:
        img_crop = get_image_mod()
        # Check if rescaling is needed
        if rescale_img:
            img_array.append(rescale_intensity(img_crop))
        else:
            img_array.append(img_crop)

    return np.array(img_array)


def merge_clusters(cluster_arr: numpy.ndarray,
                   clusters_to_merge: numpy.ndarray,
                   new_name: str):
    """
    Merge specified clusters in a cluster array or integers.

    Parameters:
        - cluster_arr (numpy.ndarray): Array of integers representing cluster names.
        - clusters_to_merge (numpy.ndarray): Array of integers specifying clusters to be merged.
        - new_name (str): New cluster name to assign to merged clusters.

    Returns:
        - numpy.ndarray: New cluster array with specified clusters merged.
    """
    mask = np.isin(cluster_arr, clusters_to_merge)
    new_cluster_arr = np.where(mask, new_name, cluster_arr)
    return new_cluster_arr


def get_thresholds(markers_counts: numpy.ndarray,
                   peak_range: tuple = None,
                   sigma_mult: float = 1.0) -> tuple:
    """
    Get thresholds from Z-normalized marker expression data.
    The assumption here is that the first peak in image histogram is related to background and
    the background is normally distributed.
    Thresholds and background values are calculated column-wise.

    Parameters:
        - markers_counts (numpy.ndarray): Input array containing Z-norm marker counts.
        - peak_range (tuple): Range of the histogram to be used for a peak search
        - sigma_mult (float): Multiplier for computed background sigma value.

    Returns:
        - numpy.ndarray: Thresholds for each marker
        - numpy.ndarray: Background values
    """
    # Check peak search parameters
    if peak_range is None:
        peak_range = (np.min(markers_counts), np.max(markers_counts))

    # Transpose. Data should be z-normalized already
    markers_counts = np.transpose(markers_counts)

    # Create empty lists for thresholds and background peaks
    marker_thresholds = []
    background_values = []

    # Iterate through all the markers
    for marker in markers_counts:

        # Make a histogram
        hist = np.histogram(marker, bins=1000, range=peak_range)

        # Get histogram bars and
        hist_values = hist[0]
        bin_edges = hist[1]

        # Get bin centers
        bin_centers = np.cumsum(np.diff(bin_edges)) + bin_edges[0]

        # Smooth histogram for peak finding
        smoothed_histogram = gaussian_filter1d(hist_values, 5)

        # Look for the first peak in the histogram (background peak)
        peaks, _ = find_peaks(smoothed_histogram, prominence=50)

        if len(peaks) > 0:
            background_peak_index = peaks[0]

            # Get first peak width in bins
            hist_first_peak_width = peak_widths(smoothed_histogram, np.array([background_peak_index]), rel_height=0.5)

            # Compute real peak width
            peak_width = hist_first_peak_width[0][0] * np.mean(np.diff(bin_centers))

            # Assume that it is a normal distribution and compute sigma. Simply divide by 2.355
            sigma = peak_width / round(2 * np.sqrt(2 * np.log(2)), 3)

            # Append values to corresponding arrays
            marker_thresholds.append(bin_centers[background_peak_index] + sigma * sigma_mult)
            background_values.append(bin_centers[background_peak_index])

        # If no peaks are found add None
        else:
            marker_thresholds.append(None)
            background_values.append(None)

    return np.asarray(marker_thresholds), np.asarray(background_values)


def get_binary_cells(z_norm_data: numpy.ndarray,
                     peak_range: tuple = None,
                     sigma_mult: float = 1.0) -> numpy.ndarray:
    """
    Generate a binary matrix indicating cells that are positive

    Parameters:
        - z_norm_data (numpy.ndarray): 2D array of normalized expression values (markers x cells).
        - peak_range (tuple): Range of the histogram to be used for a peak search.
        - sigma_mult (float): Multiplier for computed background sigma value.


    Returns:
        - numpy.ndarray: Binary matrix indicating cells that are positive.
    """

    # Compute thresholds
    thresholds, _ = get_thresholds(z_norm_data, peak_range, sigma_mult)

    # Transpose the input data for marker-wise processing
    z_norm_data_tr = np.transpose(z_norm_data)

    # Create a binary matrix for positive cells
    markers_norm_frame = [column > threshold for column, threshold in zip(z_norm_data_tr, thresholds)]

    # Transpose the binary matrix back to the original orientation
    return np.transpose(markers_norm_frame)


def recenter_cores(adata: AnnData,
                   core_names_arranged: list):
    """
    Recenter spatial coordinates of cores.

    Parameters:
        - adata (AnnData): An AnnData object containing spatial information.
        - core_names_arranged (list): A list of core names representing the desired arrangement.

    Returns:
        - tuple: Modified X and Y coordinates.
    """
    # Extract spatial coordinates and core names
    X_coords = adata.obsm['spatial'][:, 0].copy()
    Y_coords = adata.obsm['spatial'][:, 1].copy()
    core_names = adata.obs['cores'].to_numpy()

    # Iterate through unique core names and recenter them
    row = 0
    col = 0
    for core, unique_name in enumerate(core_names_arranged):
        mask = core_names == unique_name
        X_coords[mask] = X_coords[mask] - np.mean(X_coords[mask]) + 1000 * col
        Y_coords[mask] = Y_coords[mask] - np.mean(Y_coords[mask]) - 1000 * row
        col += 1
        if col % 5 == 0:
            col = 0
            row += 1

    return X_coords, Y_coords


def get_channel_names(tiff_path: str) -> numpy.ndarray:
    """
    Retrieve the names of channels from the OME metadata of a TIFF file.

    Parameters:
        - tiff_path (str): The file path to the TIFF image.

    Returns:
        - list: A list containing the names of channels extracted from the OME metadata.
    """
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            ome_metadata = tif.ome_metadata
    except Exception as e:
        raise ValueError(f"Error reading TIFF file: {e}")
    channel_names = []
    try:
        tree = ET.ElementTree(ET.fromstring(ome_metadata))
        for elem in tree.findall(".//*[@Name]"):
            if 'Channel' in elem.tag:
                channel_names.append(elem.attrib['Name'])
    except Exception as e:
        raise ValueError(f"Error parsing OME metadata: {e}")

    return np.array(channel_names)


def make_averaged_crop(img_path: str,
                       marker_name: str,
                       cluster_name: str,
                       cluster_names: list,
                       adata: AnnData,
                       img_px_size: float,
                       num_samples: int = 2000,
                       crop_size: int = 100,
                       shuffle_bool: bool = True,
                       ) -> numpy.ndarray:
    """
    Process a specified channel in the full resolution ome.tiff file and return an averaged crop

    Parameters:
        - img_path (str): Path to the image file.
        - marker_name (str): Name of the marker.
        - cluster_name (str): Name of the cluster.
        - full_names (list): Full list of names of clusters
        - adata (object): Anndata object containing data.
        - img_px_size (float): Pixel size of the image.
        - num_samples (int): Number of samples to process.
        - crop_size (int): Size of the crop in pixels
        - shuffle_bool (boolean): Shuffle or not to shuffle

    Returns:
        - img_return (object): The averaged image
    """

    # Get channel names from the image metadata
    img_ch_names = get_channel_names(img_path)

    # Find indices of the specified cluster name in the full_names list
    cluster_indices = [index for index, name in enumerate(cluster_names) if name == cluster_name]

    if shuffle_bool:
        # Shuffle the indices to make it random
        np.random.shuffle(cluster_indices)

    # Extract spatial coordinates from the provided Anndata object
    X_coords = copy.deepcopy(adata.obsm['spatial'][:, 0])
    Y_coords = copy.deepcopy(adata.obsm['spatial'][:, 1])

    # Select a subset of coordinates based on cluster indices
    if len(cluster_indices) < num_samples:
        num_samples = int(len(cluster_indices) - 1)
        print(f'There are not enough cells in the cluster, taking {len(cluster_indices)} cells: ')

    selected_coords = np.column_stack([X_coords, Y_coords])[cluster_indices[:num_samples]]

    # Read an image layer based on marker name
    marker_img = tifffile.imread(img_path, series=0, level=0, key=np.where(img_ch_names == marker_name)[0][0])

    # Get an image array for selected channel using selected coordinates
    imgs = get_img_array(marker_img, crop_size, selected_coords, rescale_img=False, px_size=img_px_size)

    # Compute the mean of the selected images
    img_return = np.mean(imgs, axis=0)

    # Cleanup
    del marker_img

    return img_return


def process_matrix(all_clusters: numpy.ndarray,
                   matrix: tuple,
                   core_cluster_names: numpy.ndarray,
                   intervals: numpy.ndarray) -> numpy.ndarray:
    """
    Process a matrix based on specified cluster names and intervals.

    Parameters:
        - all_clusters (numpy.ndarray): List of all possible cluster names.
        - matrix (numpy.ndarray): Input matrix to be processed.
        - core_cluster_names (numpy.ndarray): List of cluster names to consider in the processing.
        - intervals (numpy.ndarray): List of intervals for processing.

    Returns:
        - new_matrix (numpy.ndarray): Processed matrix which has the same number of cols as len(all_clusters)
    """

    new_matrix = []

    for cluster_a in all_clusters:
        # Iterating through all possible cluster names

        if cluster_a in core_cluster_names:
            # Checking if cluster_a is in core cluster names

            temp_cluster = []

            for cluster_b in all_clusters:
                # Checking if cluster_b is in cluster_names

                matrix_arr_indx_a = np.where(core_cluster_names == cluster_a)[0]
                matrix_arr_indx_b = np.where(core_cluster_names == cluster_b)[0]
                tmp_arr = matrix[0][matrix_arr_indx_a, matrix_arr_indx_b]

                if len(tmp_arr) == 0:
                    # If cluster was not found, make array of nans
                    tmp_arr = np.full((len(intervals) - 1), np.nan)

                temp_cluster.append(np.ravel(tmp_arr))

            new_matrix.append(np.asarray(temp_cluster, dtype='float32'))

        else:
            # If cluster_a is not in cluster_names
            empty_matrix = np.full((len(all_clusters), len(intervals) - 1), np.nan)

            new_matrix.append(empty_matrix)

    return np.asarray(new_matrix)


def get_average_co_occurrence_matrix(adata_obj: AnnData,
                                     intervals: numpy.ndarray,
                                     cluster_obs_key: str,
                                     all_clusters: str = None,
                                     cores_obs_key: str = 'cores') -> numpy.ndarray:
    """
    Compute averaged co-occurrence score per given classes.

    Parameters:
        - adata_obj (AnnData): Input data to be analyzed.
        - intervals (numpy.ndarray): Distances interval at which co-occurrence is computed.
        - cluster_obs_key (string): adata_obj.obs key for cluster names
        - cores_obs_key (string): adata_obj.obs key for core names

    Returns:
        - numpy.ndarray: co-occurrence matrix
    """

    if not all_clusters:
        # Get all unique names of clusters to be analyzed
        all_clusters = np.unique(np.asarray(adata_obj.obs[cluster_obs_key], dtype='str'))

    # Get core names to be analyzed and treat each core as a separate dataset
    core_names = np.unique(np.asarray(adata_obj.obs[cores_obs_key], dtype='str'))

    # Co-occurrence matrices will be stored in a list
    co_occ_matrices = []

    for core in tqdm(core_names):
        # Filter the adata_obj for each core
        adata_copy = adata_obj[adata_obj.obs[cores_obs_key] == core]

        # Get cluster names that are present in the core of interest
        cluster_names = np.unique(np.asarray(adata_copy.obs[cluster_obs_key], dtype='str'))

        # Compute the co-occurrence matrix
        coo_matrix = sq.gr.co_occurrence(adata_copy,
                                         cluster_key=cluster_obs_key,
                                         interval=intervals,
                                         show_progress_bar=False,
                                         copy=True)

        # Process occurrence matrix
        coo_matrix_homogenized = process_matrix(all_clusters,
                                                coo_matrix,
                                                cluster_names,
                                                intervals)
        co_occ_matrices.append(coo_matrix_homogenized)

    # Average all matrices along intervals
    av_co_occ_matrix = np.nanmean(co_occ_matrices, axis=0)

    return av_co_occ_matrix


def plot_bar_chart(categories: list,
                   values: list) -> tuple:
    """
    Create and return a bar plot with white background and gray bars.

    Parameters:
        - categories (list): List of category names for the x-axis.
        - values (list): List of corresponding values for the y-axis.

    Returns:
        - tuple: A tuple containing the figure and axis objects.
    """

    # Create Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color='gray')
    # Set the default matplotlib style
    ax.set_facecolor('black')

    # Display values above each bar
    for bar, value in zip(bars, values):
        name = " " + str(value)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                name, ha='center', va='bottom', color='white', rotation=90, size=10)

    # Make labels and customize the plot
    plt.xlabel('Cell classes')
    plt.xticks(rotation=90)
    ax.set_ylim(0, 20000)
    plt.ylabel('Cell counts')
    plt.title('Cell counts per class')
    plt.grid(False)

    return fig, ax
