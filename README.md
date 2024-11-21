# Automatic RPE cell counting in AO-OCT images

This repository contains the dataset and model source codes for the paper "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy" [[paper](LINK)]

## Dataset

Each image comes with the ground truth manual markings saved in the same folder. The cell centers are stored in the variable cellXY which denotes the x and y coordinates of the cells in the first and second columns, respectively.

## Installation instructions using conda
Coming soon

## Codes

### Training the cell detection module
For training details using the data in the paper, please see ```training.py```. Make sure the dataset are stored as they appear in this repository.

### Training the vessel segmentation module
For training details using the data in the paper, please see ```training_vessel.py```.

## Citation
If you use our model or dataset, please cite our work:

* S. Soltanian-Zadeh, K. Kovalick, S. Aghayee, D. T. Miller, Z. Liu, D. X. Hammer, and S. Farsiu, "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 15(12), 2024.
