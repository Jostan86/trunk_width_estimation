import os

class PackagePaths:
    def __init__(self, config_file):
        
        self.package_dir = os.environ.get('WIDTH_ESTIMATION_PACKAGE_PATH')
        
        self.data_dir = os.path.join(self.package_dir, "data")
        self.config_dir = os.path.join(self.package_dir, "config")
        self.startup_image_path = os.path.join(self.data_dir, "startup_image.png")
        self.model_dir = os.path.join(self.data_dir, "models")
        self.rgb_test_images_dir = os.path.join(self.data_dir, "test_images", "rgb")
        self.depth_test_images_dir = os.path.join(self.data_dir, "test_images", "depth")
        
        self.config_file_path = os.path.join(self.config_dir, config_file)
        self.startup_image_path = os.path.join(self.data_dir, "startup_image.png")
        
        self.rgb_test_image_paths = [os.path.join(self.rgb_test_images_dir, f) for f in os.listdir(self.rgb_test_images_dir) if f.endswith('.png')]
        self.rgb_test_image_paths.sort()
        
        self.depth_test_image_paths = [os.path.join(self.depth_test_images_dir, f) for f in os.listdir(self.depth_test_images_dir) if f.endswith('.png')]
        self.depth_test_image_paths.sort()
        
        
if __name__ == "__main__":
    # this is the command to add the environment variable to .bashrc from the command line if you're in the package directory
    # echo "export WIDTH_ESTIMATION_PACKAGE_PATH=$(pwd)" >> ~/.bashrc
    
    package_paths = PackagePaths(config_file="width_estimation_config_apple.yaml")
    
    # print off the paths and there values
    for key, value in package_paths.__dict__.items():
        # if the key ends in 'dir' then print the key and the value
        if key.endswith('dir'):
            print(f"{key}: {value}")