# Trunk Width Estimation

## Description 
This package is aimed at estimating the width of tree trunks from RGB-D data. A YOLOv8 model is used to segment the trunks and posts in the images. From the segmentation mask the pixel width of the trunk can be determined. The aligned depth can then be used to calculate the real-world width.

## Installation
### Option 1: VSCode Devcontainer
If familiar, the easiest way to get started is likely using a VSCode devcontainer. A .devcontainer folder with several options is included in the repository. The 'width_estimation_desktop' option is likely the most useful.

A GPU and Nvidia Runtime is needed for better processing, instructions to install Nvidia runtime are [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Alternatively, the code can be run on CPU, the ```runtime: nvidia``` argument in the docker compose file will have to be removed however.

Alternatively, the dockerfiles folder could also be used to make a docker container and development can be done in there. 

### Option 2: Ubuntu 20.04/22.04
A python virtual environment on linux should also work fine.

```bash
cd /where/you/want/to/install/venv
python3.10 venv -m trunk_width_venv
source /path/to/venv/bin/activate
```

The only python packages needed are the following:

``` pip install ultralytics scikit-image opencv-python pydantic ```

Cuda packages may be needed for ultralytics to work, refer to the ultralytics documentation for that.

### Installing the package
To install the package itself with pip, first download the repository, then install the package in editable mode.

```bash
git clone https://github.com/Jostan86/trunk_width_estimation.git
pip install -e /path/to/trunk_width_estimation
```

## Usage
```usage_example.py``` in ```scripts/``` provides an example of how to use the trunk_segmenter and trunk_analyzer classes. It loads images from  ```/data/test_images/``` and displays the segmentations and prints the width_estimation results. The path on line 119 must be set to to the root directory of the package for it to work.

A script to create a ROS2 node to subscribe to ROS topics from a Realsense D435 is also available at ```scrpts/ros_publisher_node.py```. It then publishes the data using custom message types from the [pf_orchard_interfaces](https://github.com/Jostan86/pf_orchard_interfaces) package. Alternatively, ```scripts/ros_service_node.py``` provides a ROS service that can be sent RGB-D data and returns the trunk width data. 