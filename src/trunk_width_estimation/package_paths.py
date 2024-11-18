import os

class PackagePaths:
    def __init__(self, config_file: str = None):
        """
        This class is used to store the paths to the data and config directories for the package. It also stores the paths to the test images and the model directory.
        
        Args:
            config_file (str): The name of the config file to use.
        """
        
        self._package_dir = os.environ.get('WIDTH_ESTIMATION_PACKAGE_PATH')
        
        self._data_dir = os.environ.get('WIDTH_ESTIMATION_PACKAGE_DATA_PATH')
        
        self._config_dir = os.path.join(self.package_dir, "config")
        self._startup_image_path = os.path.join(self.data_dir, "startup_image.png")
        self._model_dir = os.path.join(self.data_dir, "models")
        self._rgb_test_images_dir = os.path.join(self.data_dir, "test_images", "rgb")
        self._depth_test_images_dir = os.path.join(self.data_dir, "test_images", "depth")
        
        if config_file is not None:
            self._config_file_path = os.path.join(self.config_dir, config_file)
        else:
            self._config_file_path = None
        self._startup_image_path = os.path.join(self.data_dir, "startup_image.png")
        
        self._rgb_test_image_paths = [os.path.join(self.rgb_test_images_dir, f) for f in os.listdir(self.rgb_test_images_dir) if f.endswith('.png')]
        self._rgb_test_image_paths.sort()
        
        self._depth_test_image_paths = [os.path.join(self.depth_test_images_dir, f) for f in os.listdir(self.depth_test_images_dir) if f.endswith('.png')]
        self._depth_test_image_paths.sort()

        self._analysis_results_dir = os.path.join(self.data_dir, "analysis_results")
        self._analysis_gt_data_dir = os.path.join(self.data_dir, "orchard_gt_data")
        self._analysis_gt_data_dirs = [os.path.join(self._analysis_gt_data_dir, f) for f in os.listdir(self._analysis_gt_data_dir) if os.path.isdir(os.path.join(self._analysis_gt_data_dir, f))]

        self._current_analysis_results_dir = None
        self._analysis_results_data_path = None
        self._analysis_results_summary_path = None
        self._current_analysis_config_path = None
    
    def set_config_file_path(self, config_file_path: str):
        self._config_file_path = config_file_path

    def set_current_analysis_data(self, analysis_name: str, set_config_to_current: bool = False):
        self._current_analysis_results_dir = os.path.join(self._analysis_results_dir, analysis_name)
        self._analysis_results_data_path = os.path.join(self._current_analysis_results_dir, "results_data.csv")
        self._analysis_results_summary_path = os.path.join(self._current_analysis_results_dir, "results_summary.csv")
        self._current_analysis_config_path = os.path.join(self._current_analysis_results_dir, "config_parameters.yaml")
        if set_config_to_current:
            self._config_file_path = self._current_analysis_config_path

    @property
    def package_dir(self):
        if self._package_dir is None:
            raise ValueError("The WIDTH_ESTIMATION_PACKAGE_PATH environment variable is not set. Please set it to the root directory of the package.")
        if not os.path.isdir(self._package_dir):
            raise ValueError(f"The directory {self._package_dir} does not exist. Please set the WIDTH_ESTIMATION_PACKAGE_PATH environment variable to the root directory of the package.")
        return self._package_dir
    
    @property
    def data_dir(self):
        if self._data_dir is None:
            raise ValueError("The WIDTH_ESTIMATION_PACKAGE_DATA_PATH environment variable is not set. Please set it to the data directory of the package.")
        if not os.path.isdir(self._data_dir):
            raise ValueError(f"The directory {self._data_dir} does not exist. Please set the WIDTH_ESTIMATION_PACKAGE_DATA_PATH environment variable to the data directory of the package.")
        return self._data_dir
    
    @property
    def config_dir(self):
        if not os.path.isdir(self._config_dir):
            raise ValueError(f"The directory {self._config_dir} does not exist.")
        return self._config_dir
    
    @property
    def startup_image_path(self):
        if not os.path.isfile(self._startup_image_path):
            raise ValueError(f"The file {self._startup_image_path} does not exist.")
        return self._startup_image_path
    
    @property
    def model_dir(self):
        if not os.path.isdir(self._model_dir):
            raise ValueError(f"The directory {self._model_dir} does not exist.")
        return self._model_dir
    
    @property
    def rgb_test_images_dir(self):
        if not os.path.isdir(self._rgb_test_images_dir):
            raise ValueError(f"The directory {self._rgb_test_images_dir} does not exist.")
        return self._rgb_test_images_dir
    
    @property
    def depth_test_images_dir(self):
        if not os.path.isdir(self._depth_test_images_dir):
            raise ValueError(f"The directory {self._depth_test_images_dir} does not exist.")
        return self._depth_test_images_dir
    
    @property
    def config_file_path(self):
        if self._config_file_path is None:
            raise ValueError("The config_file attribute was not set when the class was instantiated.")
        if not os.path.isfile(self._config_file_path):
            raise ValueError(f"The file {self._config_file_path} does not exist.")
        return self._config_file_path
    
    @property
    def rgb_test_image_paths(self):
        if len(self._rgb_test_image_paths) == 0:
            raise ValueError(f"There are no test images in the directory {self._rgb_test_images_dir}.")
        return self._rgb_test_image_paths

    @property
    def depth_test_image_paths(self):
        if len(self._depth_test_image_paths) == 0:
            raise ValueError(f"There are no test images in the directory {self._depth_test_images_dir}.")
        return self._depth_test_image_paths  
    
    @property
    def analysis_results_dir(self):
        if not os.path.isdir(self._analysis_results_dir):
            raise ValueError(f"The directory {self._analysis_results_dir} does not exist.")
        return self._analysis_results_dir
    
    @property
    def analysis_gt_data_dir(self):
        if not os.path.isdir(self._analysis_gt_data_dir):
            raise ValueError(f"The directory {self._analysis_gt_data_dir} does not exist.")
        return self._analysis_gt_data_dir
    
    @property
    def analysis_gt_data_dirs(self):
        if len(self._analysis_gt_data_dirs) == 0:
            raise ValueError(f"There are no directories in the directory {self._analysis_gt_data_dir}.")
        return self._analysis_gt_data_dirs
    
    @property
    def current_analysis_results_dir(self):
        if self._current_analysis_results_dir is None:
            raise ValueError("The current_analysis_results_dir attribute was not set, set it using the set_current_analysis_data method.")
        # if not os.path.isdir(self._current_analysis_results_dir):
        #     raise ValueError(f"The directory {self._current_analysis_results_dir} does not exist.")
        return self._current_analysis_results_dir
        
    @property
    def analysis_results_data_path(self):
        if self._analysis_results_data_path is None:
            raise ValueError("The analysis_results_data_path attribute was not set, set it using the set_current_analysis_data method.")
        # if not os.path.isfile(self._analysis_results_data_path):
        #     raise ValueError(f"The file {self._analysis_results_data_path} does not exist.")
        return self._analysis_results_data_path
    
    @property
    def analysis_results_summary_path(self):
        if self._analysis_results_summary_path is None:
            raise ValueError("The analysis_results_summary_path attribute was not set, set it using the set_current_analysis_data method.")
        # if not os.path.isfile(self._analysis_results_summary_path):
        #     raise ValueError(f"The file {self._analysis_results_summary_path} does not exist.")
        return self._analysis_results_summary_path
    
    @property
    def current_analysis_config_path(self):
        if self._current_analysis_config_path is None:
            raise ValueError("The current_analysis_config_path attribute was not set, set it using the set_current_analysis_data method.")
        # if not os.path.isfile(self._current_analysis_config_path):
        #     raise ValueError(f"The file {self._current_analysis_config_path} does not exist.")
        return self._current_analysis_config_path
    
    
if __name__ == "__main__":
    # this is the command to add the environment variable to .bashrc from the command line if you're in the package directory
    # echo "export WIDTH_ESTIMATION_PACKAGE_PATH=$(pwd)" >> ~/.bashrc
    
    # package_paths = PackagePaths(config_file="width_estimation_config_apple.yaml")
    package_paths = PackagePaths()
    
    # print off the paths and there values
    print(f"package_dir: {package_paths.package_dir}")
    print(f"data_dir: {package_paths.data_dir}")
    print(f"config_dir: {package_paths.config_dir}")
    print(f"startup_image_path: {package_paths.startup_image_path}")
    print(f"model_dir: {package_paths.model_dir}")
    print(f"rgb_test_images_dir: {package_paths.rgb_test_images_dir}")
    print(f"depth_test_images_dir: {package_paths.depth_test_images_dir}")
    print(f"config_file_path: {package_paths.config_file_path}")
    # print(f"rgb_test_image_paths: {package_paths.rgb_test_image_paths}")
    # print(f"depth_test_image_paths: {package_paths.depth_test_image_paths}")
