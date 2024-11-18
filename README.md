# Trunk Width Estimation

## Description 
This package is aimed at estimating the width of tree trunks from RGB-D data. A YOLOv8 model is used to segment the trunks and posts in the images. From the segmentation mask the pixel width of the trunk can be determined. The aligned depth can then be used to calculate the real-world width. Scroll through the [jupyter notebook](https://github.com/Jostan86/trunk_width_estimation/blob/main/scripts/usage_example.ipynb) to see a visualization of the estimation process.

## Installation
### Option 1: VSCode Devcontainer
If familiar, the easiest way to get started is likely using a VSCode devcontainer. Two devcontainer setups are available in the .devcontinaer folder, one for running on a desktop, and one for a jetson.

A GPU and Nvidia Runtime is needed for better processing, instructions to install Nvidia runtime are [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Alternatively, the code can be run on CPU, the ```runtime: nvidia``` and ```deploy:``` arguments in the docker compose file may have to be removed however.

Alternatively, the dockerfiles folder could also be used to make a docker container and development can be done in there. 

### Option 2: Ubuntu 20.04/22.04
A python virtual environment on linux should also work fine.


```bash
cd /where/you/want/to/install/venv
python3.10 -m venv trunk_width_venv
source /path/to/venv/bin/activate
```

#### Dependencies
The python packages needed are the following:
```bash
pip install ultralytics \
            scikit-image \
            opencv-python \
            pydantic \
            pandas \
            matplotlib  \
            pyqt6 \
            superqt
```

Cuda packages may be needed for ultralytics to work, refer to the ultralytics documentation for that.

#### Installing the package
To install the package itself with pip, first download the repository, then install the package. The ```-e``` option installs it editable mode.

```bash
cd /path/to/workspace
git clone https://github.com/Jostan86/trunk_width_estimation.git
pip install -e /path/to/trunk_width_estimation
```
#### Installing pf_orchard_interfaces
To use the ```ros_service_node.py``` and ```ros_publisher_node.py``` scripts as ROS2 nodes, messages from the [pf_orchard_interfaces](https://github.com/Jostan86/pf_orchard_interfaces) package are needed. These can be installed with the below commands, change ~/ros2_pf_ws to where you'd like your workspace.
```bash
mkdir -p ~/ros2_pf_ws/src
cd ~/ros2_pf_ws/src
git clone https://github.com/Jostan86/pf_orchard_interfaces.git
cd ..
colcon build
```
Then the workspace will also need to be sourced for any terminal using it. 
```bash
source ~/ros2_pf_ws/install/setup.bash
```
## Package Data
If using the devcontainer/docker setup, the package data is downloaded and setup in the dockerfile. Otherwise the data can be downloaded and the environment setup as follows.
### Handling Paths
The package handles paths using the PackagePaths object in the ```package_paths.py``` file, it expects two environment variables to be set:
- WIDTH_ESTIMATION_PACKAGE_PATH: Points to the main package directory location
- WIDTH_ESTIMATION_PACKAGE_DATA_PATH: Points to the directory with the package data, including the model and test images.

These can be the same, if the package data is stored in the main package directory.
### Downloading Data
The package data has been split into two files:
- [Main Data](https://1drv.ms/u/s!AhPJ6XcTEu5um4h39NKf_4RSMTpFFw?e=bB0cPN): The data needed to run the TrunkAnalyzer, which inludes the model, a startup image, and the test images.
- [Analysis Data](https://1drv.ms/u/s!AhPJ6XcTEu5um4h42BhYmvwnBR4MDw): The data needed to run an analysis or the tuning app.

If both are downloaded, the data from both should be put in the same directory.

### Directory Structure
The PackagePaths object expects the file structure to be organized as:
```text
trunk_width_estimation/
  ├── config/
  │   └── yaml config files
  ├── docker/
  │   └── docker files
  ├── scripts/
  │   └── scripts
  ├── src/
  │   └── source code
  ├── README.md/
  └── setup.py

trunk_width_estimation_package_data/
  ├── analysis_results/
  │   ├── sample_results/ 
  |   ├── results_1/ 
  │   └── results_2.../
  ├── models/
  │   ├── jazz_s_v8.pt
  │   └── other models...
  ├── orchard_gt_models/ 
  │   ├── dataset1/
  │   └── dataset2.../
  ├── test_images/ 
  └── startup_image.png
  ```
**analysis_results/**: Part of analysis data, stores results from analyses.

**orchard_gt_models/**: Part of analysis data, has image datasets with ground truth width data.

## Usage
All of the below scripts should be ready to run without any setup if using the devcontainer/docker setup.

### usage_example.py
```usage_example.py``` in ```scripts/``` provides an example of how to use the TrunkAnalyzer and TrunkSegmenter classes. It loads images from  ```trunk_width_estimation_package_data/test_images/``` and displays the segmentations and prints the width_estimation results. The path at the end must be set, or the environment variables setup as shown above in [Handling Paths](#handling-paths). The 'Main Data' is needed for this script (see [Package Data](#package-data)).

### Jupyter Notebook
A Jupyter notebook has been setup [here](https://github.com/Jostan86/trunk_width_estimation/blob/main/scripts/usage_example.ipynb) at ```scripts/usage_example.ipynb```  that illustrates how the package works by stepping through each of the steps in the process and showing a visualization of it. The 'Main Data' is needed for this script (see [Package Data](#package-data)).

### ROS Scripts

A script to create a ROS2 node to subscribe to ROS topics from a Realsense D435 is also available at ```scrpts/ros_publisher_node.py```. It then publishes the data using custom message types from the [pf_orchard_interfaces](https://github.com/Jostan86/pf_orchard_interfaces) package. Alternatively, ```scripts/ros_service_node.py``` provides a ROS service that can be sent RGB-D data and returns the trunk width data. 

### Tuning App
A PyQt6 based app has been developed that aids in tuning, analyzing, and visualizing the algorithm. This is available at ```scripts/tuning_app.py```. It requires the 'Analysis Data' to be downloaded (see [Package Data](#package-data)). 
<!-- Tool tips have been setup in the app to try and aid in usage. -->

#### App Layout
On the right side of the app each step of the algorithm has been visualized as a series of images. On the left side are some controls to browse the images, and below that is a set of tools in tabs, which include:
- **Adjust Parameters**: At the top, select which step in the algorithm to adjust. Then the parameters for that operation can be altered and the effects visualized.
- **Operation Selection**: Some of the steps/filters in the algorithm aren't strictly necessary, so this tab provides the ability to deselect those for testing purposes. It also allows visualizations on the right side to be individually hidden so as to de-clutter if needed.
- **Analyze and Filter**: This tab allows for filtering the images included based on the analysis results that have been loaded. The 'Apply Filters' button must be pressed in order for it to take affect. 
- **App Settings**: Some app settings are adjustable, such as the size of the images, whether to include explanations with the visualizations, and the analysis results the app uses.

### Performance Analysis
`/scripts/misc_utils/performance_analysis.py` provides the entrypoint for running a performance analysis on the algorithm. The 'Analysis Data' will need to be downloaded (see [Package Data](#package-data)). An analysis can also be run from inside the tuning app.


