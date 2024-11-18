
from pydantic import BaseModel, ValidationError
import yaml
import os
import logging

class ParametersWidthEstimation(BaseModel):
    yolo_model_to_use: str
    seg_model_confidence_threshold: float
    seg_model_nms_threshold: float
    seg_model_trunk_class: int
    seg_model_post_class: int

    camera_horizontal_fov: float
        
    pixel_width_segment_length: int
    pixel_width_percentile: float
    
    depth_calc_top_ignore: float
    depth_calc_bottom_ignore: float
    depth_calc_min_num_points: int
    depth_calc_percentile: float
    
    include_nms_filter: bool
    filter_nms_overlap_threshold: float
    
    filter_depth_max_depth: float
    include_depth_filter: bool
    
    filter_edge_edge_threshold: float
    filter_edge_size_threshold: float
    include_edge_filter: bool
    
    filter_position_bottom_threshold: float
    filter_position_top_threshold: float
    filter_position_score_threshold: float
    include_position_filter: bool
    
    width_correction_slope: float
    width_correction_intercept: float
    include_width_correction: bool
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'ParametersWidthEstimation':
        """
        Load parameters from a YAML file. Must have the same fields as the class.
        
        Args:
            file_path (str): Path to the YAML file.
            logging_level (int): Logging level to use.
        """

        logging.info(f"Loading parameters from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")
        
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        try:
            parameters = cls(**data)
        except ValidationError as e:
            logging.error(f"Parameter file {file_path} does not have the correct fields or types.")
            logging.error(f"{e}")
            raise ValueError(f"Invalid data in parameter file: {e}")
                
        # parameters.log_settings()

        # Override parameters with environment variables if they exist
        if "WIDTH_CORRECTION_SLOPE" in os.environ:
            logging.info("Overriding WIDTH_CORRECTION_SLOPE with environment variable")
            parameters.width_correction_slope = float(os.environ["WIDTH_CORRECTION_SLOPE"])
        else:
            logging.info("Using WIDTH_CORRECTION_SLOPE from parameter file")
        if "WIDTH_CORRECTION_INTERCEPT" in os.environ:
            logging.info("Overriding WIDTH_CORRECTION_INTERCEPT with environment variable")
            parameters.width_correction_intercept = float(os.environ["WIDTH_CORRECTION_INTERCEPT"])
        else:
            logging.info("Using WIDTH_CORRECTION_INTERCEPT from parameter file")
        
        return parameters
    
    def save_to_yaml(self, file_path: str):
        """
        Save the parameters to a YAML file without comments.
        
        Args:
            file_path (str): Path to save the YAML file.
        """
        # Convert the model to a dictionary
        yaml_data = self.dict()

        # Write to a YAML file
        with open(file_path, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        
        logging.info(f"Parameters saved to {file_path}")

    def log_settings(self) -> None:
        """
        Log the current settings.
        """
        logging.info("Current settings:")
        for name, value in self.model_dump().items():
            logging.info(f"{name}: {value}")
    
    def set_operation_usage(self, operation_name, usage):
        """
        Set the operation usage for the parameters.
        
        Args:
            operation_name (str): The display name of the operation.
            usage (str): The usage of the operation.
        """
        if operation_name == "Width Correction":
            self.include_width_correction = usage
        elif operation_name == "Position Filter":
            self.include_position_filter = usage
        elif operation_name == "Edge Filter":
            self.include_edge_filter = usage
        elif operation_name == "Depth Filter":
            self.include_depth_filter = usage
        elif operation_name == "Non-Maximum Suppression Filter":
            self.include_nms_filter = usage
        else:
            logging.error(f"Operation name {operation_name} not recognized.")
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # parameters = ParametersWidthEstimation.load_from_yaml("/trunk_width_estimation/config/width_estimation_config_apple.yaml")
        # parameters.save_to_yaml("/trunk_width_estimation/config/width_estimation_config_apple_test.yaml")
        parameters = ParametersWidthEstimation.load_from_yaml("/home/jostan/Documents/git_repos/trunk_width_estimation/config/width_estimation_config_apple.yaml")
        parameters.save_to_yaml("/home/jostan/Documents/git_repos/trunk_width_estimation/config/width_estimation_config_apple_test.yaml")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load parameters: {e}")
