# Trunk Width Estimation

## Description 
This package is aimed at estimating the width of tree trunks from RGB-D data. A YOLOv8 model is used to segment the trunks and posts in the images. From the segmentation mask the pixel width of the trunk can be determined. The aligned depth can then be used to calculate the real-world width.

## Installation
### Option 1: VSCode Devcontainer
If familiar, the easiest way to get started is likely using a VSCode devcontainer. A .devcontainer folder with several options is included in the repository. The desktop-ubuntu-22 option is likely the most useful.

Alternatively, the dockerfile in the devcontainer folder could also be used to make a docker container and developement can be done in there. 

### Option 2: Ubuntu 20.04/22.04
A python virtual environment on linux should also work fine. The only python packages needed are the following:

``` pip install ultralytics scikit-image opencv-python pydantic ```

Cuda packages may be needed for ultralytics to work, refer to thier documentation for that.

### Installing the package
Regardless of the option for dependencies, to install the package itself with pip, first download the repository, then run ```pip install -e /path/to/trunk_width_estimation```. 

## Usage
```usage_example.py``` in ```scripts/``` provides an example of how to use the trunk_segmenter and trunk_analyzer classes. It loads images from the ```/data/test_images/``` and displays the segmentations and prints the width_estimation results.
