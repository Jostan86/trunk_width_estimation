from pydantic import BaseModel, ValidationError
from typing import List
import yaml
import os
import logging

class ParametersWidthEstimation(BaseModel):
    yolo_model_to_use: str
    ignore_classes: List[int]
    trunk_class: int
    post_class: int
    
    camera_horizontal_fov: float
    
    
    pixel_width_segment_length: int
    
    depth_calc_top_ignore: float
    depth_calc_bottom_ignore: float
    depth_calc_min_num_points: int
    depth_calc_percentile: float
    
    filter_nms_overlap_threshold: float
    
    filter_depth_max_depth: float
    
    filter_edge_edge_threshold: float
    filter_edge_size_threshold: float
    
    filter_position_bottom_threshold: float
    filter_position_top_threshold: float
    filter_position_score_threshold: float
    
    width_correction_slope: float
    width_correction_intercept: float
    do_width_correction: bool
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'ParametersWidthEstimation':
        """Load parameters from a YAML file. Must have the same fields as the class."""
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
                
        parameters.log_settings()
        
        return parameters

    def log_settings(self) -> None:
        logging.info("Current settings:")
        for name, value in self.model_dump().items():
            logging.info(f"{name}: {value}")

    
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        parameters = ParametersWidthEstimation.load_from_yaml("/trunk_width_estimation/config/width_estimation_config_apple.yaml")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load parameters: {e}")
