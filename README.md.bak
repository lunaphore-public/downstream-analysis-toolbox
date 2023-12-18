## Semi-automated approaches for interrogating spatial heterogeneity of tissue samples

This repository contains Python scripts that were used to analyze multiplex sequential immunofluorescence (seqIF) 
single-cell data. Data table was extracted from [COMET](https://lunaphore.com/products/comet/) image using the QuPath software as described in the manuscript.
Multiplex seqIF image is available [here](https://lunaphore.com/). Full description of data analysis is provided in the manuscript.

<br>
The Python code in this repository is provided as is with no warranties.

### Setup
The conda environment file is provided to simplify the setup process. To install all
dependencies follow the instructions below:

* Download and install [Anaconda](https://www.anaconda.com/).
* Create a new environment by using downstream_analysis_env.yml file provided in this repository as described [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file ). 
* Activate the environment by using the following command:
* `conda activate downstream_analysis_env`
* Install the environment as a kernel for Jupyter Notebook by using the following command:
* `python -m ipykernel install --user --name=downstream_analysis_env`
* Navigate to the project root folder and start Jupyter Notebook by using the following command:
* `jupyter notebook`
* Jupyter Notebook will open in your default browser.
* Make sure that the kernel is set to downstream_analysis_env
* Open downstream_analysis_notebook.ipynb

### System requirements
The code was tested on Windows 10 machine with 32 GB of RAM and i7 CPU.

### Running the analysis
Make sure that all associated data files are in the data folder. The notebook can be executed partially without .ome.tiff file.
The data folder should contain the following files:
* single_cell_dataset.csv
* m_plex_img.ome.tiff
* adata_non_subtracted.zip
* adata_unsupervised.zip

Execute all the cells in the notebook by pressing the **Run All** button.
We recommend to use [collapsible headings](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/collapsible_headings/readme.html) for easier navigation. 
Further instructions are provided in the notebook.


	
