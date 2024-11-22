# Automatic RPE cell counting in AO-OCT images

This repository contains the dataset and model source codes for the paper "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," *Biomedical Optics Express*, 15(12), pp. 6922-6939 (2024) [[paper](https://opg.optica.org/boe/fulltext.cfm?uri=boe-15-12-6922&id=563701)]

## Dataset

Each image comes with the ground truth manual markings saved in the same folder. The cell centers are stored in the variable cellXY which denotes the x and y coordinates of the cells in the first and second columns, respectively.

## Installation instructions using conda
The following will create a conda environment named ```rpe_torch``` and install all necessary packages:

```conda env create -f environment.yml```

## Codes
To run any script, first activate the created conda environment by:

```conda activate rpe_torch```

### Training the cell detection module
For training details using the data in the paper, please see ```train.py```. Make sure the dataset are stored as they appear in this repository.

### Training the vessel segmentation module
For training details using the data in the paper, please see ```DconnNet\train.py```.

## Citation
If you use our model or dataset, please cite our work:

* S. Soltanian-Zadeh, K. Kovalick, S. Aghayee, D. T. Miller, Z. Liu, D. X. Hammer, and S. Farsiu, "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," *Biomedical Optics Express*, 15(12), pp. 6922-6939 (2024).
