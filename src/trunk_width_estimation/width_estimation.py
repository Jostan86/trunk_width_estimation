import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
from skimage.measure import label, regionprops
import os
import time
from .package_paths import PackagePaths
from .parameters import ParametersWidthEstimation
from typing import List, Tuple, Callable
from tqdm import tqdm
from datetime import datetime
import json
import pandas as pd
import logging

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from pf_orchard_interfaces.msg import TreeImageData, TreeInfo, TreePosition
# from cv_bridge import CvBridge
# from message_filters import ApproximateTimeSynchronizer, Subscriber

def import_ros2_modules():
    if not TrunkAnalyzerData.ros_2_modules_imported:
        try:
            from pf_orchard_interfaces.msg import TreeImageData
            from cv_bridge import CvBridge

            TrunkAnalyzerData.cv_bridge = CvBridge()
            TrunkAnalyzerData.ros_2_modules_imported = True
            TrunkAnalyzerData.tree_image_data_msg = TreeImageData()
            
        except ImportError as e:
            print("Error importing ROS2 modules: ", e)

class ProcessOrderError(Exception):
    pass


class TrunkAnalyzerData:
    """Class to hold the data for the trunk analyzer."""
    ros_2_modules_imported = False
    cv_bridge = None
    tree_image_data_msg = None

    def __init__(self):

        self.results: Results = None
       
        self.rgb_image: np.ndarray = None
        self.depth_image: np.ndarray = None

        self.confidences: np.ndarray = None
        self.masks: np.ndarray = None
        self.classes: np.ndarray = None
        self.results_kept: np.ndarray = None

        self.depth_estimates: np.ndarray = None
        self.cropped_width: int = None
        self.original_width: int = None
        self.height: int = None
        self.num_pixels: int = None

        self.object_widths: np.ndarray = None
        self.x_positions_in_image: np.ndarray = None
        self.object_locations: np.ndarray = None

        self.largest_segment_finder_ran = False
        self.image_segmented_flag = False
    
    @classmethod
    def from_images(cls, rgb_image: np.ndarray, depth_image: np.ndarray):
        """Create a new instance of the class from the results of the model and the rgb image."""
        new_trunk_data = cls()

        new_trunk_data.add_depth_image(depth_image)

        new_trunk_data.rgb_image = TrunkAnalyzerData.check_image_size(rgb_image)

        return new_trunk_data
    
    @classmethod
    def from_image_paths(cls, rgb_image_path: str, depth_image_path: str):
        """Create a new instance of the class from the paths to the images."""
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        rgb_image = cv2.imread(rgb_image_path)

        return cls.from_images(rgb_image, depth_image)

    @classmethod
    def from_results_dict(cls, results_dict: dict, rgb_image: np.ndarray):
        """Create a new instance of the class from the results dict and the rgb image."""
        new_trunk_data = cls()

        new_trunk_data._setup_arrays_from_dict(results_dict)

        new_trunk_data.rgb_image = rgb_image

        return new_trunk_data
    
    @classmethod 
    def from_ros2_image_msgs(cls, rgb_msg, depth_msg):
        """Create a new instance of the class from the ROS2 messages."""
        import_ros2_modules()
        
        cls.rgb_msg: Image = rgb_msg
        cls.depth_msg: Image = depth_msg

        rgb_image = cls.cv_bridge.imgmsg_to_cv2(cls.rgb_msg, desired_encoding='passthrough')
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = cls.cv_bridge.imgmsg_to_cv2(cls.depth_msg, desired_encoding='passthrough')

        return cls.from_images(rgb_image, depth_image)
    
    @staticmethod
    def create_results_dict(results: Results):
        """Create a results dict from the results of the model."""
        confidences, masks, classes, results_kept = TrunkAnalyzerData.extract_results(results)
        results_dict = {"confidences": confidences, "masks": masks, "classes": classes, "results_kept": results_kept}
        
        return results_dict
    
    # @staticmethod
    def create_ros2_tree_info_msg(self, header=None):
        """Create ROS2 messages from the trunk data."""
        import_ros2_modules()

        if self.rgb_msg is None and header is None:
            raise ValueError("The header must be provided if the rgb_msg is not set. (i.e. if the data was not created from ROS2 messages using the from_ros2_image_msgs method.)")
        
        tree_image_data = TrunkAnalyzerData.tree_image_data_msg

        tree_image_data.header = self.rgb_msg.header if self.rgb_msg is not None else header
        tree_image_data.segmented_image = TrunkAnalyzerData.cv_bridge.cv2_to_imgmsg(self.visualize_segmentation(), encoding='passthrough')

        if self.num_instances is None:
            tree_image_data.xs = []
            tree_image_data.ys = []
            tree_image_data.widths = []
            tree_image_data.classifications = []
            tree_image_data.confidences = []
            tree_image_data.object_seen = False
        else:
            tree_image_data.object_seen = True
            tree_image_data.xs = self.object_locations[:, 0].tolist()
            tree_image_data.ys = self.object_locations[:, 1].tolist()
            print("object locations before: ", self.object_locations)
            tree_image_data.widths = self.object_widths.tolist()
            tree_image_data.classifications = self.classes.tolist()
            tree_image_data.confidences = self.confidences.tolist()

        return tree_image_data

    @staticmethod
    def run_segmentation_only(trunk_segmenter, rgb_image: np.ndarray):
        """Run the segmentation only on the rgb image."""
        trunk_segmenter: TrunkSegmenter

        results_dict, results = trunk_segmenter.run(rgb_image)
        return results_dict, results
    
    @staticmethod
    def extract_results(results: Results):
        """Extract the results from the model."""
        num_instances = len(results.boxes) if len(results.boxes) > 0 else None
        confidences = results.boxes.conf.numpy() if num_instances is not None else None
        masks = results.masks.data.numpy().astype(np.uint8) if num_instances is not None else None
        classes = results.boxes.cls.numpy().astype(np.uint8) if num_instances is not None else None
        results_kept = np.arange(num_instances) if num_instances is not None else None
        return confidences, masks, classes, results_kept

    @staticmethod
    def check_image_size(image: np.ndarray) -> np.ndarray:
        """Check if the image is divisible by 32 and crop if it is not.
        
        Args:
            image (np.ndarray): Image to check the size of.
        
        Returns:
            np.ndarray: Image cropped to be divisible by 32.
        
        """
        
        if image.shape[0] % 32 != 0:
                image = image[:-(image.shape[0] % 32), :, :]
        if image.shape[1] % 32 != 0:
            image = image[:, :-(image.shape[1] % 32), :]
        
        return image
    
    def add_results(self, results: Results):
        """Add the results from the model to the class."""
        self.setup_arrays_from_results(results)

    def _setup_arrays_from_dict(self, results_dict: dict):
        """Setup the arrays for analysis using the results dict from the model.

        Args:
            results_dict (dict): Dictionary of the results from the model.
        """

        self.num_instances = results_dict["num_instances"]
        self.confidences: np.ndarray = results_dict["confidences"]
        self.masks: np.ndarray = results_dict["masks"]
        self.classes: np.ndarray = results_dict["classes"]
        self.results_kept: np.ndarray = results_dict["results_kept"]
            
    def setup_arrays_from_results(self, results: Results):
        """Setup the arrays for analysis using the results from the model.

        Args:
            results (YOLO): Results from the model.
        """
        self.results = results

        self.confidences, self.masks, self.classes, self.results_kept = self.extract_results(results)

    def add_depth_image(self, depth_image: np.ndarray):
        """Add a depth image to the class."""
        self.original_width = depth_image.shape[1]

        depth_image = self.check_image_size(depth_image)

        self.height = depth_image.shape[0]
        self.cropped_width = depth_image.shape[1]

        self.depth_image = depth_image

        self.num_pixels = self.cropped_width * self.height

    def remove_faulty_instances(self, keep_indices: np.ndarray):
        """Update all the active arrays based on the keep_indices array from an operation. Getting rid of
        instances that have been filtered out.

        Args:
            keep_indices (np.ndarray): Indices of the instances that are being kept after filtering.
        """

        self.confidences = self.confidences[keep_indices]
        self.masks = self.masks[keep_indices]
        self.results_kept = self.results_kept[keep_indices]
        self.classes = self.classes[keep_indices]

        if self.results is not None:
            self.results = self.results[keep_indices]

        if self.depth_estimates is not None:
            self.depth_estimates = self.depth_estimates[keep_indices]
        if self.object_widths is not None:
            self.object_widths = self.object_widths[keep_indices]
        if self.x_positions_in_image is not None:
            self.x_positions_in_image = self.x_positions_in_image[keep_indices]        
        if self.object_locations is not None:
            self.object_locations = self.object_locations[keep_indices]
        
        if self.num_instances == 0:
            self.confidences = None
            self.masks = None
            self.classes = None
            self.results_kept = None
            self.depth_estimates = None
            self.object_widths = None
            self.x_positions_in_image = None
            self.object_locations = None

    @property
    def num_instances(self):
        return None if self.confidences is None else len(self.confidences)
    
    
    def visualize_segmentation(self, add_mask_num: bool = False) -> np.ndarray:
        """Draw the segmentation on the rgb image."""
        rgb_image = self.rgb_image.copy()

        if self.num_instances is None:
            return rgb_image       

        for i, (object_class, mask) in enumerate(zip(self.classes, self.masks)):
            if add_mask_num:
                rgb_image = self.draw_mask_on_image(rgb_image, mask, object_class, mask_num=i + 1)
            else:
                rgb_image = self.draw_mask_on_image(rgb_image, mask, object_class)

        return rgb_image
    
    def draw_mask_on_image(self, 
                           image: np.ndarray, 
                           mask: np.ndarray, 
                           object_class: int, 
                           shade_amount: float = 0.4, 
                           mask_num: int = None) -> np.ndarray:
        """Draw the mask on the image.
        
        Args:
            image (np.ndarray): Image to draw the mask on.
            mask (np.ndarray): Mask to draw on the image.
            object_class (int): Class of the object in the mask.
            shade_amount (float, optional): Amount to shade the mask. Defaults to 0.4.
            mask_num (int, optional): Number of the mask. Defaults to None.
        """

        # Create a copy of the image to overlay the mask
        overlay = image.copy()

        # 0 is a tree and this makes it blue
        if object_class == 0:
            overlay[mask == 1] = [255, 0, 0]  # Set the mask area to blue
        # 1 is a trunk and this makes it red
        elif object_class == 1:
            overlay[mask == 1] = [0, 0, 255]  # Set the mask area to red
        else:
            overlay[mask == 1] = [0, 255, 0]

        # Blend the original image and the overlay
        blended = cv2.addWeighted(overlay, shade_amount, image, 1 - shade_amount, 0)

        if mask_num is not None:
            # Find the center point of the mask
            moments = cv2.moments(mask)
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            self.add_text_with_background(blended,
                                          str(mask_num),
                                          (center_x, center_y),
                                          text_scale=2,
                                          vertical_move_ratio=0.5,
                                          horizontal_move_ratio=-0.5,
                                          exclude_background=True)

        return blended
    
    @staticmethod
    def add_text_with_background(image: np.ndarray, 
                                 text: str, 
                                 position: Tuple[int, int], 
                                 text_scale: float=0.75, 
                                 background_color=(0, 0, 0), 
                                 text_color=(255, 255, 255), 
                                 font=cv2.FONT_HERSHEY_SIMPLEX,
                                 thickness: int=2,
                                 vertical_move_ratio: float=0,
                                 horizontal_move_ratio: float=0,
                                 exclude_background: bool=False):
        """Add text with a background to the image.
        
        Args:
            image (np.ndarray): Image to add the text to.
            text (str): Text to add to the image.
            position (Tuple[int, int]): Position to add the text.
            text_scale (float, optional): Scale of the text. Defaults to 0.75.
            background_color (Tuple[int, int, int], optional): Color of the background. Defaults to (0, 0, 0).
            text_color (Tuple[int, int, int], optional): Color of the text. Defaults to (255, 255, 255).
            font ([type], optional): Font of the text. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            thickness (int, optional): Thickness of the text. Defaults to 2.
            vertical_move_ratio (float, optional): Ratio to move the text vertically. Defaults to 0.
            horizontal_move_ratio (float, optional): Ratio to move the text horizontally. Defaults to 0.
            exclude_background (bool, optional): If True, doesn't draw the shaded background behind the text. Defaults to False.
        """
        try:
            # Get the size of the text (width, height)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, thickness=2)

            # Coordinates for the rectangle background
            x, y = position
            y += int(vertical_move_ratio * text_height)
            x += int(horizontal_move_ratio * text_width)

            rectangle_bgr = background_color  # Background color for the rectangle (black)

            if not exclude_background:
                # Draw the rectangle (slightly larger than the text size for padding)
                cv2.rectangle(image,
                            (x, y - text_height - baseline),
                            (x + text_width, y + baseline),
                            rectangle_bgr,
                            cv2.FILLED)
            
            # Now overlay the text on top of the rectangle
            cv2.putText(img=image,
                        text=text,
                        org=(x, y),
                        fontFace=font,
                        fontScale=text_scale,
                        color=text_color,
                        thickness=thickness)
        except Exception as e:
            print("Error adding text with background: ", e)
    
    def visualize_depth_image(self, max_depth: float = 3) -> np.ndarray:
        """Visualize the depth image.
        
        Args:
            max_depth (float, optional): Maximum depth to display. Defaults to 3.

        Returns:
            np.ndarray: Visualized depth image.
        """

        depth_image = self.depth_image.copy()

        # Convert depth values from mm to meters
        depth_image = depth_image / 1000.0
        
        depth_image[depth_image > max_depth] = max_depth

        # Record where values close to zero are
        zero_mask = depth_image < 0.0001

        # set zero values to the minimum value of the mask
        depth_image[zero_mask] = depth_image[~zero_mask].min()    

        # change the sign of the depth image (just to reverse the colormap)
        depth_image = -depth_image
        
        # Normalize depth values between 0 and 1
        depth_image = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255), cv2.COLORMAP_JET)

        # Where zero values were, set the color to white
        depth_colormap[zero_mask] = 255

        return depth_colormap
    
    @staticmethod
    def add_contour_to_depth_image(depth_colormap: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add the contour of the mask to the depth image.
        
        Args:
            depth_colormap (np.ndarray): Depth image to add the contour to.
            mask (np.ndarray): Mask to add the contour of.
            
        Returns:
            np.ndarray: Depth image with the contour added.
        """

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_colormap, contours, -1, (0, 0, 0), 3)

        return depth_colormap


class TrunkAnalyzerAbstractOperation:
    """Abstract class for the operations in the trunk analyzer."""
    def __init__(self, parameters: ParametersWidthEstimation, employ_operation: bool=True):
        """
        Args:
            parameters (ParametersWidthEstimation): Parameters for the trunk analyzer.
            employ_operation (bool, optional): If True, the operation will be run. Defaults to True.
        """

        self.parameters = parameters
        self.employ_operation = employ_operation
        self.name = self.__class__.__name__
        self.display_name = self.name
        self.visualizations = None
        self.reset_operation_data()
        self.reset_performance_analysis_data()

        self.create_vis_flag = False

        self.num_visualizations: int = 1
        self.num_extra_detail_visualizations = 0

        self.is_required = False

    def reset_operation_data(self):
        """Reset the data for the operation."""
        self.applied_filter = False
        self.operation_skipped = True
        self.operation_time = None
        self.visualizations: List[ProcessVisualization] = None
    
    def reset_performance_analysis_data(self):
        """Reset the operations's performance analysis data."""
        self.total_time = 0
        self.calls_list = []
        self.applied_list = []
    
    @property
    def total_calls(self):
        """Total number of times the operation was called."""
        return sum(self.calls_list)
    
    @property
    def total_times_applied(self):
        """Total number of times the operation was actually run/applied."""
        return sum(self.applied_list)
    
    @property
    def total_images(self):
        """Total number of images the operation was run on since the last reset."""
        return len(self.calls_list)

    def run_and_analyze(self, trunk_analyzer_data: TrunkAnalyzerData, create_vis_flag: bool=False):
        """Run the operation and time how long it takes.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to run the operation on.
            create_vis_flag (bool, optional): If True, create visualizations. Defaults to False.
            
        Returns:
            TrunkAnalyzerData: Updated data after running the operation.
        """

        start_time = time.time()
        trunk_analyzer_data = self.check_and_run(trunk_analyzer_data, create_vis_flag=create_vis_flag)
        self.operation_time = time.time() - start_time

        self.total_time += self.operation_time

        if not self.operation_skipped:
            self.calls_list.append(1)
        else: 
            self.calls_list.append(0)
        
        if self.applied_filter:
            self.applied_list.append(1)
        else:
            self.applied_list.append(0)

        return trunk_analyzer_data
    
    def check_and_run(self, trunk_analyzer_data: TrunkAnalyzerData, create_vis_flag: bool=False):
        """Check if the operation should be run and then run it.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to run the operation on.
            create_vis_flag (bool, optional): If True, create visualizations. Defaults to False.
            
        Returns:
            TrunkAnalyzerData: Updated data after running the operation.
        """
        self.create_vis_flag = create_vis_flag

        self.reset_operation_data()
        if trunk_analyzer_data.num_instances is not None or not trunk_analyzer_data.image_segmented_flag:
            self.operation_skipped = False
            num_instances_before = trunk_analyzer_data.num_instances
            trunk_analyzer_data = self.run(trunk_analyzer_data)
            num_instances_after = trunk_analyzer_data.num_instances
            if num_instances_before != num_instances_after:
                self.applied_filter = True
        
        if self.create_vis_flag:
            self.create_visualization(trunk_analyzer_data)

        return trunk_analyzer_data
    
    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Run the operation on the data.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to run the operation on.
            
        Returns:
            TrunkAnalyzerData: Updated data after running the operation.
        """
        
        raise NotImplementedError("The run method must be implemented in the subclass.")
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create a visualization of the operation.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to create the visualization from.
        """

        raise NotImplementedError("The create_visualization method must be implemented in the subclass.")
    
    def print_performance_data(self):
        """Print the performance analysis of the operation."""
        
        if not self.employ_operation:
            return
        
        print("--------------------")
        print(self.name)
        print("Total time: ", self.total_time)
        print("Total images: ", self.total_images)
        print("Total calls: ", self.total_calls)
        print("Total applied", self.total_times_applied)
    
    def add_performance_data(self, data_dict: dict):
        """Add the performance data to the dictionary.
        
        Args:
            data_dict (dict): Dictionary to add the performance data to.
        """
        if not self.employ_operation:
            return data_dict
        data_dict[self.name + "_time"] = self.operation_time
        data_dict[self.name + "_called"] = self.calls_list[-1]
        data_dict[self.name + "_applied"] = self.applied_list[-1]

        return data_dict
    
    @staticmethod
    def make_histogram(values: np.ndarray,
                    min_percentile: float,
                    max_percentile: float,
                    step_size: float,
                    mask_num: int,
                    title: str,
                    xlabel: str,
                    used_value: float=None,
                    used_value_label: str=None) -> np.ndarray:
        """Make a histogram of the values.

        Args:
            values (np.ndarray): Values to make the histogram of.
            min_percentile (float): Minimum percentile to include in the histogram.
            max_percentile (float): Maximum percentile to include in the histogram.
            step_size (float): Step size for the histogram.
            mask_num (int): Number of the mask.
            title (str): Title of the histogram.
            xlabel (str): Label for the x-axis.
            used_value ([type], optional): Value to add a line for. Defaults to None.
            used_value_label ([type], optional): Label for the used value. Defaults to None.

        Returns:
            np.ndarray: cv2 image of the histogram.
        """

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import io

        # Create a new figure and canvas
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Compute percentiles
        min_percentile = np.percentile(values, min_percentile) 
        max_percentile = np.percentile(values, max_percentile)

        # Round the values to the nearest step size
        max_value = np.ceil(max_percentile / step_size) * step_size
        min_value = np.floor(min_percentile / step_size) * step_size

        # Get the number of points over and under the min/max allowed depth
        over_max = np.sum(values > max_value)
        under_min = np.sum(values < min_value)

        # Filter values within the desired range
        values = values[(values >= min_value) & (values <= max_value)]

        # Adjust max and min to include the last bin
        max_value += step_size
        min_value -= step_size
        
        # Plot the histograms
        if len(values) > 0:
            ax.hist(values, bins=np.arange(min_value, max_value, step_size), alpha=0.5, label="Central Values", color='blue')
        if over_max > 0:
            over_max_representation = np.ones(over_max) * (max_value - step_size/2)
            ax.hist(over_max_representation,
                    bins=np.arange(min_value, max_value + step_size, step_size),
                    alpha=0.5, 
                    label=f"Values Over {max_value-step_size}mm",
                    color='orange')
        if under_min > 0:
            under_min_representation = np.ones(under_min) * (min_value + step_size/2)
            ax.hist(under_min_representation,
                    bins=np.arange(min_value, max_value, step_size),
                    alpha=0.5, 
                    label=f"Values Under {min_value+step_size}mm",
                    color='red')
        
        # Add a line for the used value
        if used_value is not None:
            ax.axvline(x=used_value, color='r', linestyle='dashed', linewidth=1, label=used_value_label)
        ax.set_xlim([min_value, max_value])
        ax.legend(loc='upper right')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        
        # Convert the figure to a cv2 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png') 
        buf.seek(0)
        buffer_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        cv2_image = cv2.imdecode(buffer_image, cv2.IMREAD_COLOR)
        buf.close()

        return cv2_image
    
    @staticmethod
    def add_mask_num(trunk_analyzer_data: TrunkAnalyzerData):
        """Helper function to determine whether a mask number should be added to the mask on the visualization. Returns true if there's
        more than 1 mask.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to add the mask number to.
        
        Returns:
            bool: True if there's more than 1 mask.
        """
        if trunk_analyzer_data.num_instances is not None:
            if trunk_analyzer_data.num_instances > 1:
                return True
        
        return False
    
    @property
    def operation_ran(self):
        """Check if the operation was run."""
        return self.applied_filter or not self.operation_skipped
    
    def set_parameters(self, parameters: ParametersWidthEstimation):
        """Set the parameters for the operation.

        Args:
            parameters (ParametersWidthEstimation): Parameters to set.
        """
        self.parameters = parameters
    

class ProcessVisualization:
    """Base class for the visualizations in the trunk analyzer."""
    def __init__(self, operation_name: str = None):
        """
        Args:
            operation_name (str, optional): Name of the operation. Defaults to None.
        """
        self.name = operation_name
        self.image = None
        self.extra_detail_only = False
        self.metrics: List[Tuple[str, str | int | float]] = []
        self.explanation = ""
    
    def set_image(self, image: np.ndarray):
        """Set the image for the visualization.

        Args:
            image (np.ndarray): Image to set.
        """
        self.image = image
    
    def add_metric(self, metric_name: str, metric_value, num_decimal_places: int=2):
        """Add a metric to the visualization.

        Args:
            metric_name (str): Name of the metric.
            metric_value ([type]): Value of the metric.
            num_decimal_places (int, optional): Number of decimal places to round the metric to. Defaults to 2.
        """
        if not isinstance(metric_value, (str, int, np.integer)):
            metric_value = round(float(metric_value), num_decimal_places)

        self.metrics.append((metric_name, metric_value))
       
    def add_metrics_to_image(self):
        """Add the metrics to the image itself. Makes an area on the right side of the image to display the metrics and adds them."""

        font_scale = 0.75
        if self.image is None:
            raise ValueError("The image must be set before adding metrics to it.")
        if len(self.metrics) == 0:
            return
        
        metric_texts = []
        text_lengths = []
        for metric_name, metric_value in self.metrics:
            metric_texts.append(f"{metric_name} {metric_value}")
            text_lengths.append(cv2.getTextSize(metric_texts[-1], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0])
        
        max_text_length = max(text_lengths)
        text_height = cv2.getTextSize(metric_texts[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, 2)[0][1]

        # add white space on the right side of the image
        self.image = np.concatenate((self.image, np.zeros((self.image.shape[0], max_text_length, 3), dtype=np.uint8)), axis=1)
        for i, metric_text in enumerate(metric_texts):
            cv2.putText(self.image, 
                        text=metric_text, 
                        org=(self.image.shape[1] - max_text_length, text_height + i * text_height),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=font_scale, 
                        color=(255, 255, 255), 
                        thickness=1)
        
    def display(self):
        """Display the visualization using cv2."""
        cv2.imshow(self.name, self.image)
        self.cv2_wait()
    
    def show_visualization(self):
        """Prepare the visualization and display it."""
        self.add_metrics_to_image()
        self.display()
    
    def get_image(self):
        """Get the image of the visualization."""
        return self.image

    @staticmethod
    def cv2_wait():
        """A helper function to wait for a key press in cv2 and also exit cleanly."""
        # This is needed because cv2.imshow isn't working in docker right
        print('select the image and press any key to continue')
        while True:
            key = cv2.waitKey(25) & 0xFF
            if key != 255:  # Any key pressed
                break


class TrunkSegmenter(TrunkAnalyzerAbstractOperation):
    """Class to segment the trunks in an image using the YOLO model."""
    
    def __init__(self, package_paths: PackagePaths, parameters: ParametersWidthEstimation = None):
        """
        Args:
            package_paths (PackagePaths): Class that holds the paths to the package directories.
            parameters (ParametersWidthEstimation, optional): Parameters for the trunk analyzer. Loads them using the package paths if none.
            combine_segmenter (bool, optional): If True, it assumes the segmenter is combined with the rest of the trunk analyzer. 
                                                Set False to run the segmenter independently.
        """
        if parameters is None:
            parameters = ParametersWidthEstimation.load_from_yaml(package_paths.config_file_path)

        super().__init__(parameters) 

        self.package_paths = package_paths
        self.display_name = "Trunk Segmenter"

        self.is_required = True

        self.resetting = False

        self.reset_model()

    def reset_model(self):
        """Reset the model to the one specified in the parameters."""
        # Load the model
        weight_path = os.path.join(self.package_paths.model_dir, self.parameters.yolo_model_to_use)
        self.yolo_model = YOLO(weight_path, task="segment")

        # Put an image through the model to initialize the model
        startup_image = cv2.imread(self.package_paths.startup_image_path)
        _ = self.run_segmentation(startup_image)
    
    def set_parameters(self, parameters: ParametersWidthEstimation):
        """Set the parameters for the segmentation operation.

        Args:
            parameters (ParametersWidthEstimation): Parameters to set.
        """
        old_yolo_model_name = self.parameters.yolo_model_to_use
        self.parameters = parameters

        if old_yolo_model_name != self.parameters.yolo_model_to_use:
            self.reset_model()
    
    def run(self, trunk_data: TrunkAnalyzerData):
        """Do the prediction on an image and return the results.

        Args:
            data (TrunkAnalyzerData): Trunk data to do the prediction on or the data object to add the results to.

        Returns:
            dict: Dictionary of the results from the model.
        """

        if trunk_data.image_segmented_flag:
            logging.warning("The trunk data passed to the segmenter already has the image segmented.")
            return trunk_data
        
        results = self.run_segmentation(trunk_data.rgb_image)

        trunk_data.setup_arrays_from_results(results)
        trunk_data.image_segmented_flag = True
        return trunk_data
    
    def run_segmentation(self, rgb_image: np.ndarray):
        """Run the segmentation on the rgb image."""
        # Do the prediction on the image
        

        results: Results = self.yolo_model.predict(rgb_image, 
                                          imgsz=(rgb_image.shape[0], rgb_image.shape[1]), 
                                          iou=self.parameters.seg_model_nms_threshold,
                                          conf=self.parameters.seg_model_confidence_threshold,
                                          verbose=False, 
                                          )
        
        results = results[0].cpu()

        return results
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create a visualization of the segmentation operation.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data to create the visualization from.    
        """
        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The image segmentation model has segmented out posts, trunks, and sprinklers in the image."
        visualization.set_image(vis_image)

        # add the confidence threshold and nms threshold to the visualization, and the actual confidences
        visualization.add_metric("Confidence Threshold:", self.parameters.seg_model_confidence_threshold)
        visualization.add_metric("NMS Threshold:", self.parameters.seg_model_nms_threshold, num_decimal_places=3)

        self.visualizations = [visualization]

        if trunk_analyzer_data.num_instances is None:
            return
        
        visualization.add_metric("", "")
        for i, confidence in enumerate(trunk_analyzer_data.confidences):
            visualization.add_metric(f"Mask {i + 1}:", "")
            visualization.add_metric(f"Confidence: ", confidence)


class FilterObjectType(TrunkAnalyzerAbstractOperation):
    """Filter out objects of unwanted classes and change the class to match the 'standard' used in the particle filter."""
    def __init__(self, parameters: ParametersWidthEstimation):
        """
        Args:
            parameters (ParametersWidthEstimation): Parameters for the trunk analyzer.
        """
        super().__init__(parameters)
        self.display_name = "Object Type Filter"

        self.is_required = True

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Change the class to match the 'standard' and filter out objects of unwanted classes."""

        trunk_analyzer_data.classes = np.where(trunk_analyzer_data.classes == self.parameters.seg_model_trunk_class, 0, 
                                        np.where(trunk_analyzer_data.classes == self.parameters.seg_model_post_class, 1, 2))

        keep_indices = np.isin(trunk_analyzer_data.classes, [0, 1])

        trunk_analyzer_data.remove_faulty_instances(keep_indices)

        return trunk_analyzer_data
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create a visualization of the object type filter operation."""
        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The object type filter has removed any sprinklers and sets the class numbers to match " \
                                    "what is expected by the rest of the algorithm."
        visualization.set_image(vis_image)

        visualization.add_metric("Trunk Class:", self.parameters.seg_model_trunk_class)
        visualization.add_metric("Post Class:", self.parameters.seg_model_post_class)

        self.visualizations = [visualization]

class LargestSegmentFinder(TrunkAnalyzerAbstractOperation):
    """Find the largest segment in the mask and keep only that segment."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Remove Multiple Segments"
        self.visualization_segments = []

        self.is_required = True

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Determine if there are multiple segments in the mask and if so keep only the largest one."""
        
        largest_segments = np.zeros_like(trunk_analyzer_data.masks)

        self.visualization_segments = []

        for i, mask in enumerate(trunk_analyzer_data.masks):
            # Label connected regions in the mask
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

            if num_labels == 1:
                largest_segments[i] = mask
            else:
                self.applied_filter = True
                largest_segments[i] = self._get_largest_segment(labeled_mask, i,)

        trunk_analyzer_data.masks = largest_segments

        trunk_analyzer_data.largest_segment_finder_ran = True

        return trunk_analyzer_data

    def _get_largest_segment(self, labeled_mask: np.ndarray, mask_num: int) -> np.ndarray:
        """Get the largest segment in a mask with multiple segments."""

        # Find properties of each connected region
        props = regionprops(labeled_mask)
        # Sort the regions by their area in descending order
        props.sort(key=lambda x: x.area, reverse=True)
        # Keep only the largest connected segment
        largest_segment_mask = labeled_mask == props[0].label

        if self.create_vis_flag:
            # add the other segments to the visualization list
            for i, prop in enumerate(props):

                if i == 0:
                    continue
                mask = labeled_mask == prop.label
                self.visualization_segments.append((mask, mask_num))

        return largest_segment_mask.astype(np.uint8)
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Display visualization of what the operation is doing."""
        
        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        for i, (mask, mask_num) in enumerate(self.visualization_segments):
            object_class = trunk_analyzer_data.classes[mask_num]
            vis_image = trunk_analyzer_data.draw_mask_on_image(vis_image, mask, object_class, shade_amount=1)


        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The largest segment finder ensures that the mask consists of only one segment."
        visualization.set_image(vis_image)

        self.visualizations = [visualization]


class FilterNMS(TrunkAnalyzerAbstractOperation):
    """Apply non-maximum suppression (NMS) to remove overlapping masks of different classes.
    
    Yolo applies NMS, but only to individual classes, so this is a second NMS to remove any overlapping masks from 
    different classes. There is technically an option to do this in Yolo, but it can't have a distinct iou threshold
    from the first NMS.
    """

    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Non-Maximum Suppression Filter"
        self.max_overlap = 0

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Apply non-maximum suppression to remove overlapping masks of different classes."""

        self.max_overlap = 0

        if trunk_analyzer_data.num_instances <= 1:
            self.operation_skipped = True
            return trunk_analyzer_data
        
        # TODO: is this necessary? Also, there has to be a package that does NMS efficiently already...
        mask_nms = trunk_analyzer_data.masks.copy()
        score_nms = trunk_analyzer_data.confidences.copy()

        indices = np.argsort(-score_nms)
        mask_nms = mask_nms[indices]

        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros(trunk_analyzer_data.num_instances, dtype=bool)

        # For each mask, compute overlap with other masks and suppress overlapping masks if their score is lower
        for i in range(len(mask_nms) - 1):
            if suppressed[i]:
                continue

            # Compute overlap with other masks
            overlap = np.sum(mask_nms[i] * mask_nms[i + 1:], axis=(1, 2)) / np.sum(mask_nms[i] + mask_nms[i + 1:], axis=(1, 2))
            # Suppress masks that are either already suppressed or have an overlap greater than the threshold
            suppressed[i + 1:] = np.logical_or(suppressed[i + 1:], overlap > self.parameters.filter_nms_overlap_threshold)

            if np.max(overlap) > self.max_overlap:
                self.max_overlap = np.max(overlap)

        # Get the indices of the masks that were not suppressed
        indices_revert = np.argsort(indices)
        suppressed = suppressed[indices_revert]
        not_suppressed = np.logical_not(suppressed)

        trunk_analyzer_data.remove_faulty_instances(not_suppressed)

        return trunk_analyzer_data
    
    def add_performance_data(self, data_dict: dict):
        data_dict = super().add_performance_data(data_dict)

        data_dict[self.name + "_max_overlap"] = self.max_overlap

        return data_dict

    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Display visualization of what the NMS filter operation is doing."""
        
        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The non-maximum suppression filter removes overlapping masks of different classes " \
                                    "(Same class masks are already suppressed by the YOLO model)."
        visualization.add_metric("Overlap Threshold:", self.parameters.filter_nms_overlap_threshold, num_decimal_places=3)

        if self.max_overlap > 0:
            visualization.add_metric("", "")
            visualization.add_metric("Max Overlap in Image:", self.max_overlap)
        
        visualization.set_image(vis_image)

        self.visualizations = [visualization]


class DepthCalculation(TrunkAnalyzerAbstractOperation):
    """Calculate the depth of the mask based on the aligned depth image. Also filters out masks with too valid few depth points."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Depth Calculation Filter"
        self.num_visualizations = 2

        self.is_required = True

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Calculates the depth estimate as the percentile depth for each mask. Also filters out masks that have less than
        min_num_points valid points in the region defined by top_ignore and bottom_ignore."""

        # Initialize array to store the depth values and the tree locations
        trunk_analyzer_data.depth_estimates = np.zeros(trunk_analyzer_data.num_instances)

        # Make boolean array of indices to keep
        keep = np.ones(trunk_analyzer_data.num_instances, dtype=bool)

        # Calculate the top and bottom ignore values in pixels
        self.top_ignore = int(self.parameters.depth_calc_top_ignore * trunk_analyzer_data.height)
        self.bottom_ignore = trunk_analyzer_data.height - int(self.parameters.depth_calc_bottom_ignore * trunk_analyzer_data.height)

        self.num_points = np.zeros(trunk_analyzer_data.num_instances, dtype=int)

        # Loop through each mask
        for i, mask in enumerate(trunk_analyzer_data.masks):
            mask: np.ndarray

            # Make copy of mask array
            mask_copy = mask.copy()

            # Zero out the top and bottom ignore regions
            mask_copy[:self.top_ignore, :] = 0
            mask_copy[self.bottom_ignore:, :] = 0

            # If there are no points in the mask, remove the segment
            if np.sum(mask_copy) == 0:
                keep[i] = False
                continue

            # Convert mask copy to a boolean array
            mask_copy = mask_copy.astype(bool)
            
            # Make a 1D array of the masked portions of the depth image
            masked_depth = trunk_analyzer_data.depth_image[mask_copy]

            # Remove zero values from the masked depth array and remove the mask if there are too few points left
            masked_depth = masked_depth[masked_depth != 0]
            self.num_points[i] = masked_depth.shape[0]
            if self.num_points[i] < self.parameters.depth_calc_min_num_points:
                keep[i] = False
                continue

            # Calculate percentile depth
            trunk_analyzer_data.depth_estimates[i] = np.percentile(masked_depth, self.parameters.depth_calc_percentile) / 1000

        # Update the arrays
        trunk_analyzer_data.remove_faulty_instances(keep)

        return trunk_analyzer_data
    
    def add_performance_data(self, data_dict: dict):
        """Add the performance data for the most recent image to the dictionary if the operation was run."""

        data_dict = super().add_performance_data(data_dict)

        if self.calls_list[-1] == 1:
            data_dict[self.name + "_num_points"] = self.num_points
        else:
            data_dict[self.name + "_num_points"] = None

        return data_dict
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create the visualization for what the depth calculation operation is doing."""

        vis_image = trunk_analyzer_data.visualize_segmentation()
        depth_image = trunk_analyzer_data.visualize_depth_image()

#        add a line at the top and bottom ignore values
        cv2.line(vis_image, (0, self.top_ignore), (trunk_analyzer_data.cropped_width, self.top_ignore), (0, 0, 255), 2)
        cv2.line(vis_image, (0, self.bottom_ignore), (trunk_analyzer_data.cropped_width, self.bottom_ignore), (0, 0, 255), 2)

        # Create visualization objects and add the metrics
        visualization_rgb = ProcessVisualization("Depth Calculation Mask Area")
        visualization_rgb.explanation = f"The object depth is determined by calculating the {self.parameters.depth_calc_percentile} percentile " \
                                        "depth of the points in the indicated region on the mask. If there are less than " \
                                        "{self.parameters.depth_calc_min_num_points} points in the region, then the mask is removed." 
        visualization_rgb.add_metric("Upper Threshold Position:", self.top_ignore)
        visualization_rgb.add_metric("Lower Threshold Position:", self.bottom_ignore)

        visualization_depth = ProcessVisualization("Colorized Depth Image")
        visualization_depth.explanation = "A visualization of the corresponding depth image with the mask contours added."
        visualization_depth.add_metric("Min Allowable Points:", self.parameters.depth_calc_min_num_points)
        visualization_depth.set_image(depth_image)
        
        self.visualizations = [visualization_rgb, visualization_depth]
       
        # If there are no instances, just show the rgb image
        if trunk_analyzer_data.num_instances is None:
            visualization_rgb.set_image(vis_image)
            return
                
        masks_copy: np.ndarray = trunk_analyzer_data.masks.copy()

        for i, (mask, object_class) in enumerate(zip(masks_copy, trunk_analyzer_data.classes)):
            mask[:self.top_ignore, :] = 0
            mask[self.bottom_ignore:, :] = 0
            
            vis_image = trunk_analyzer_data.draw_mask_on_image(vis_image, mask, object_class, shade_amount=0.5, mask_num=i + 1 if trunk_analyzer_data.num_instances > 1 else None)

            trunk_analyzer_data.add_contour_to_depth_image(depth_image, mask)

        visualization_rgb.set_image(vis_image)

        for mask_num, mask in enumerate(masks_copy):
            visualization_depth.add_metric("", "")
            visualization_depth.add_metric("Mask {}:".format(mask_num + 1), "")
            visualization_depth.add_metric("Num Points:", np.sum(mask))
           
        min_percentile = 5
        max_percentile = 90
        step_size = 5
        
        self.num_extra_detail_visualizations = trunk_analyzer_data.num_instances

        for i, mask in enumerate(masks_copy):
            
            mask: np.ndarray

            depth_image_copy = trunk_analyzer_data.depth_image.copy()

            masked_depth = depth_image_copy[mask.astype(bool)]
            masked_depth = masked_depth[masked_depth != 0] 

            histogram_image = self.make_histogram(masked_depth,
                                                  min_percentile,
                                                  max_percentile,
                                                  step_size,
                                                  i,
                                                  "Mask {} Depth Histogram".format(i + 1),
                                                  "Depth (m)",
                                                  used_value=trunk_analyzer_data.depth_estimates[i] * 1000,
                                                  used_value_label="{} percentile".format(self.parameters.depth_calc_percentile))
            
            visualization_hist = ProcessVisualization("Depth Histogram Mask {}".format(i + 1))
            visualization_hist.explanation = f"A histogram of the depth values for mask {i + 1}. The {self.parameters.depth_calc_percentile} " \
                                              "percentile is shown as a dashed red line. The tails are typically quite long due to noisy depth " \
                                              "values, so these are cut off at approximately the 5th and 90th percentiles." 
            visualization_hist.set_image(histogram_image)
            visualization_hist.extra_detail_only = True
            self.visualizations.append(visualization_hist)


class FilterDepth(TrunkAnalyzerAbstractOperation):
    """Filter out masks with a depth beyond a certain threshold."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Depth Filter"

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Sort out any outputs that are beyond a given depth threshold. Note that during depth calcuation any
        segments entirely in the top or bottom portions of the image are removed, and any segments with too few
        points in the point cloud are also removed."""

        if trunk_analyzer_data.depth_estimates is None:
            raise ProcessOrderError("Depth estimates must be calculated using the FilterDepthCalculation filter before using the FilterDepth filter.")

        keep = trunk_analyzer_data.depth_estimates < self.parameters.filter_depth_max_depth
        trunk_analyzer_data.remove_faulty_instances(keep)

        return trunk_analyzer_data
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create visualization of what the depth filter is doing."""

        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The depth filter removes any masks whose estimated depth is beyond the maximum depth threshold."
        visualization.set_image(vis_image)

        visualization.add_metric("Max Depth (m):", self.parameters.filter_depth_max_depth)

        self.visualizations = [visualization]

        if trunk_analyzer_data.num_instances is None:
            return
        
        visualization.add_metric("", "")
        for i, depth in enumerate(trunk_analyzer_data.depth_estimates):
            mask_num = i + 1
            visualization.add_metric("Mask {} Depth (m):".format(mask_num), depth, num_decimal_places=3)
            visualization.add_metric("", "")
        

class FilterEdge(TrunkAnalyzerAbstractOperation):
    """Filter out masks with too many pixels in the edge of the image."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Edge Filter"

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Sort out any outputs with masks that are too close to the edge of the image."""

        keep = np.zeros(trunk_analyzer_data.num_instances, dtype=bool)

        self.percent_in_edge = np.zeros(trunk_analyzer_data.num_instances, dtype=float)

        self.edge_threshold = int(self.parameters.filter_edge_edge_threshold * trunk_analyzer_data.cropped_width)

        masks_copy: np.ndarray = trunk_analyzer_data.masks.copy()

        for i, mask in enumerate(masks_copy):
            mask: np.ndarray

            left_edge_pixels = mask[:, :self.edge_threshold].sum()
            right_edge_pixels = mask[:, -self.edge_threshold:].sum()
            total_mask_pixels = np.sum(mask)
            self.percent_in_edge[i] = (left_edge_pixels + right_edge_pixels) / total_mask_pixels
            if self.percent_in_edge[i] > self.parameters.filter_edge_size_threshold:
                continue
            else:
                keep[i] = True

        trunk_analyzer_data.remove_faulty_instances(keep)

        return trunk_analyzer_data
    
    def add_performance_data(self, data_dict: dict):
        data_dict = super().add_performance_data(data_dict)

        if self.calls_list[-1] == 1:
            data_dict[self.name + "_percent_in_edge"] = self.percent_in_edge
        else:
            data_dict[self.name + "_percent_in_edge"] = None
        
        return data_dict
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Display visualization of what the edge filter operation is doing."""

        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))

        # add a line at the edge threshold
        self.edge_threshold_right = trunk_analyzer_data.cropped_width - self.edge_threshold

        cv2.line(vis_image, (self.edge_threshold, 0), (self.edge_threshold, trunk_analyzer_data.height), (0, 0, 255), 2)
        cv2.line(vis_image, (self.edge_threshold_right, 0), (self.edge_threshold_right, trunk_analyzer_data.height), (0, 0, 255), 2)

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = f"The edge filter removes any masks with greater than {self.parameters.filter_edge_size_threshold * 100}% of " \
                                     "their pixels in the edge region of the image."
        visualization.add_metric("Edge Threshold (pixels):", int(self.parameters.filter_edge_edge_threshold * trunk_analyzer_data.cropped_width))
        visualization.add_metric("Edge Size Threshold (% mask):", self.parameters.filter_edge_size_threshold)
        
        self.visualizations = [visualization]

        if trunk_analyzer_data.num_instances is None:    
            visualization.set_image(vis_image)
            return

        # get the pixels in a mask and in the edge are and color them dark blue or red
        masks_copy: np.ndarray = trunk_analyzer_data.masks.copy()
        edge_areas = []
        mask_areas = []

        for i, mask in enumerate(masks_copy):
            mask: np.ndarray
            
            mask_area = mask.sum()
            edge_area = 0
            
            left_edge_mask = mask.copy()
            left_edge_mask[:, self.edge_threshold:] = 0
            edge_area += left_edge_mask.sum()

            right_edge_mask = mask.copy()
            right_edge_mask[:, :self.edge_threshold_right] = 0
            edge_area += right_edge_mask.sum()

            edge_areas.append(edge_area)
            mask_areas.append(mask_area)

            vis_image = trunk_analyzer_data.draw_mask_on_image(vis_image, left_edge_mask, trunk_analyzer_data.classes[i], shade_amount=1)
            vis_image = trunk_analyzer_data.draw_mask_on_image(vis_image, right_edge_mask, trunk_analyzer_data.classes[i], shade_amount=1)
        
        visualization.set_image(vis_image)

        visualization.add_metric("", "")
        for i, (edge_area, mask_area) in enumerate(zip(edge_areas, mask_areas)):
            mask_num = i + 1
            visualization.add_metric("Mask {}:".format(mask_num), "")
            visualization.add_metric("Total Pixels:", int(mask_area))
            visualization.add_metric("Allowable Edge Pixels:", int(self.parameters.filter_edge_size_threshold * mask_area))
            visualization.add_metric("Actual Edge Pixels:", int(edge_area))
            visualization.add_metric("", "")

        
class FilterPosition(TrunkAnalyzerAbstractOperation):
    """Filter out masks based on their vertical position in the image."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)

        self.display_name = "Position Filter"

        self.bottom_position_threshold = None
        self.top_position_threshold = None

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Filter out masks based on thier vertical position in the image."""

        keep = np.zeros(trunk_analyzer_data.num_instances, dtype=bool)

        self.bottom_position_threshold = int(self.parameters.filter_position_bottom_threshold * trunk_analyzer_data.height)
        self.top_position_threshold = int(self.parameters.filter_position_top_threshold * trunk_analyzer_data.height)

        self.distances_from_bottom = np.zeros(trunk_analyzer_data.num_instances, dtype=int)
        self.distances_from_top = np.zeros(trunk_analyzer_data.num_instances, dtype=int)

        for i, mask in enumerate(trunk_analyzer_data.masks):
            mask: np.ndarray

            row_sums = np.any(mask, axis=1)

            # Find the first and last non-zero row
            top_pixel = np.argmax(row_sums)
            bottom_pixel = len(row_sums) - 1 - np.argmax(row_sums[::-1])

            self.distances_from_bottom[i] = trunk_analyzer_data.height - bottom_pixel
            self.distances_from_top[i] = top_pixel

            if self.distances_from_bottom[i] < self.bottom_position_threshold and self.distances_from_top[i] < self.top_position_threshold:
                keep[i] = True

        trunk_analyzer_data.remove_faulty_instances(keep)

        return trunk_analyzer_data
    
    def add_performance_data(self, data_dict: dict):
        data_dict = super().add_performance_data(data_dict)

        if self.calls_list[-1] == 1:
            data_dict[self.name + "_distances_from_bottom"] = self.distances_from_bottom
            data_dict[self.name + "_distances_from_top"] = self.distances_from_top
        else:
            data_dict[self.name + "_distances_from_bottom"] = None
            data_dict[self.name + "_distances_from_top"] = None
        
        return data_dict  
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Display visualization of what the filter is doing."""

        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data)) 

        self.bottom_position_threshold = int(self.parameters.filter_position_bottom_threshold * trunk_analyzer_data.height)
        self.top_position_threshold = int(self.parameters.filter_position_top_threshold * trunk_analyzer_data.height)

        bottom_line_position = trunk_analyzer_data.height - self.bottom_position_threshold 

        
        # add text "need some mask above this line" and "need some mask below this line"
        text_scale = 0.75
       
        trunk_analyzer_data.add_text_with_background(image=vis_image,
                                                        text="Need some mask below this line",
                                                        position=(10, bottom_line_position),
                                                        vertical_move_ratio=-0.3,
                                                        text_scale=text_scale)
        cv2.line(img=vis_image,
                 pt1=(0, bottom_line_position),
                 pt2=(trunk_analyzer_data.cropped_width, bottom_line_position),
                 color=(0, 0, 255), 
                 thickness=2)
        
        trunk_analyzer_data.add_text_with_background(image=vis_image,
                                                        text="Need some mask above this line",
                                                        position=(10, self.top_position_threshold),
                                                        vertical_move_ratio=1.3,
                                                        text_scale=text_scale)        
        cv2.line(img=vis_image,
                 pt1=(0, self.top_position_threshold),
                 pt2=(trunk_analyzer_data.cropped_width, self.top_position_threshold),
                 color=(0, 0, 255), 
                 thickness=2)

        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The position filter removes any masks that are either too high or too low in the image."
        visualization.set_image(vis_image)

        visualization.add_metric("Bottom Threshold (# pixels from bottom):", self.bottom_position_threshold)
        visualization.add_metric("Top Threshold (# pixels from top):", self.top_position_threshold)
        
        self.visualizations = [visualization]
        
        if trunk_analyzer_data.num_instances is None:
            return
        
        for i, (dist_from_bottom, dist_from_top) in enumerate(zip(self.distances_from_bottom, self.distances_from_top)):
            mask_num = i + 1
            visualization.add_metric("", "")
            visualization.add_metric("Mask {}:".format(mask_num), "")
            visualization.add_metric("Distance from Bottom (pixels):", dist_from_bottom)
            visualization.add_metric("Distance from Top (pixels):", dist_from_top)
            visualization.add_metric("", "")
        

class WidthEstimation(TrunkAnalyzerAbstractOperation):
    """Estimate the width of the object in meters."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Width Estimation"
        self.width_vis_data_all = []

        self.is_required = True

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Calculates the best estimate of the width of the tree in meters."""

        if trunk_analyzer_data.depth_estimates is None:
            raise ProcessOrderError("Depth estimates must be calculated using the ProcessDepthCalculation filter before the width can be calculated.")

        if not trunk_analyzer_data.largest_segment_finder_ran:
            raise ProcessOrderError("The LargestSegmentFinder operation should be run before calculating the width.")

        trunk_analyzer_data.x_positions_in_image = np.zeros(trunk_analyzer_data.num_instances, dtype=np.int32)
        trunk_analyzer_data.object_widths = np.zeros(trunk_analyzer_data.num_instances)
        trunk_analyzer_data.object_locations = np.zeros((trunk_analyzer_data.num_instances, 2))

        if self.create_vis_flag:
            self.width_vis_data_all = []

        # Loop through each mask
        for i, (mask, depth) in enumerate(zip(trunk_analyzer_data.masks, trunk_analyzer_data.depth_estimates)):               

            # Get the diameter of the tree in pixels
            pixel_width_estimate = self._calculate_pixel_width(mask)

            horz_fov = self.parameters.camera_horizontal_fov

            # Calculate the width of the image in meters at the depth of the tree
            image_width_m = depth * np.tan(np.deg2rad(horz_fov / 2)) * 2

            # Calculate the distance per pixel
            distperpix = image_width_m / trunk_analyzer_data.original_width

            # Calculate the diameter of the tree in meters
            diameter_m = pixel_width_estimate * distperpix
            
            trunk_analyzer_data.object_widths[i] = diameter_m

            # Calculate the x location of the tree in the image (in pixels) by taking the median of the mask points in x
            trunk_analyzer_data.x_positions_in_image[i] = np.median(np.where(mask)[1])
            trunk_analyzer_data.object_locations[i, 1] = trunk_analyzer_data.depth_estimates[i]
            trunk_analyzer_data.object_locations[i, 0] = (trunk_analyzer_data.x_positions_in_image[i] - (trunk_analyzer_data.original_width / 2)) * distperpix

        self.object_widths = trunk_analyzer_data.object_widths.copy()

        return trunk_analyzer_data

    def _calculate_pixel_width(self, mask: np.ndarray) -> np.ndarray:
        """Get the width of the segmentation in pixels. Note that this method is different than in the paper, but
        it produces a similar results and is significantly faster.
        
        Args:
            mask (np.ndarray): Mask of the trunk segmentation.
            
        Returns:
            np.ndarray: Width of the trunk in pixels.
        """

        # Get the column of the leftmost and rightmost pixel in each row of the mask
        leftmost_pixel_columns = mask.argmax(axis=1)
        rightmost_pixel_columns = mask.shape[1] - 1 - np.flip(mask, axis=1).argmax(axis=1)

        # Remove rows where no mask is present, which is rows where the leftmost pixel is not zero and the right most 
        # is not the last column
        valid_rows = np.logical_and(leftmost_pixel_columns != 0, rightmost_pixel_columns != mask.shape[1] - 1)
        leftmost_pixel_columns = leftmost_pixel_columns[valid_rows]
        rightmost_pixel_columns = rightmost_pixel_columns[valid_rows]

        # Calculate the x_coords as the midpoint between the leftmost and rightmost points, so the center of the trunk
        x_coords = ((leftmost_pixel_columns + rightmost_pixel_columns) / 2).astype(int)
        y_coords = np.arange(mask.shape[0])[valid_rows]


        # Calculate the angle at multiple small segments along the trunk and use them to calculate a corrected width
        segment_length = self.parameters.pixel_width_segment_length 
        widths = rightmost_pixel_columns - leftmost_pixel_columns # Width of the trunk at each row
        corrected_widths = np.zeros_like(widths, dtype=float)
        # TODO: can this be vectorized?
        for i in range(0, len(y_coords) - segment_length, segment_length):
            dy = y_coords[i + segment_length] - y_coords[i]
            dx = x_coords[i + segment_length] - x_coords[i]
            angle = np.arctan2(dy, dx)
            # Correct the width based on the angle of the tree at that location
            corrected_widths[i:i + segment_length] = widths[i:i + segment_length] * np.abs(np.sin(angle))

        width_estimate = np.percentile(corrected_widths, self.parameters.pixel_width_percentile)

        if self.create_vis_flag:
            visualization_data = {}
            visualization_data["leftmost_pixel_columns"] = leftmost_pixel_columns
            visualization_data["rightmost_pixel_columns"] = rightmost_pixel_columns
            visualization_data["x_coords"] = x_coords
            visualization_data["y_coords"] = y_coords
            visualization_data["widths"] = widths
            visualization_data["corrected_widths"] = corrected_widths
            visualization_data["width_estimate"] = width_estimate
            self.width_vis_data_all.append(visualization_data)

        return width_estimate   
    
    def add_performance_data(self, data_dict: dict):
        data_dict = super().add_performance_data(data_dict)

        if self.calls_list[-1] == 1:
            data_dict[self.name + "_widths"] = self.object_widths
        else:
            data_dict[self.name + "_widths"] = None

        return data_dict
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Create visualization of how the width estimation operation works."""
        
        vis_image1 = trunk_analyzer_data.visualize_segmentation(add_mask_num=TrunkAnalyzerAbstractOperation.add_mask_num(trunk_analyzer_data))

        self.visualizations = []

        if trunk_analyzer_data.num_instances is None:
            visualization = ProcessVisualization("Width Estimation")
            visualization.explanation = "No instances were found in the image so the width estimation could not be performed."
            visualization.set_image(vis_image1)
            visualization.extra_detail_only = False 
            self.visualizations.append(visualization)
            self.num_extra_detail_visualizations = 0
            return            

        for visualization_data in self.width_vis_data_all:
            leftmost_pixel_columns = visualization_data["leftmost_pixel_columns"]
            rightmost_pixel_columns = visualization_data["rightmost_pixel_columns"]
            x_coords = visualization_data["x_coords"]
            y_coords = visualization_data["y_coords"]

            for i in [-1, 0, 1]:
                # vis_image1[y_coords, leftmost_pixel_columns+i] = [0, 255, 0]
                # vis_image1[y_coords, rightmost_pixel_columns+i] = [0, 255, 0]
                vis_image1[y_coords, x_coords+i] = [255, 0, 0]
        
        vis_image_2 = vis_image1.copy()

        for visualization_data in self.width_vis_data_all:
            y_coords = visualization_data["y_coords"]
            x_coords = visualization_data["x_coords"]
            corrected_widths = visualization_data["corrected_widths"]
            
            segment_length = self.parameters.pixel_width_segment_length
            for i in range(0, len(y_coords) - segment_length, segment_length):
                cv2.line(vis_image_2, (x_coords[i], y_coords[i]), (x_coords[i + segment_length], y_coords[i + segment_length]), (255, 0, 255), 2)
                
                dy = y_coords[i + segment_length] - y_coords[i]
                dx = x_coords[i + segment_length] - x_coords[i]
                segment_angle = np.arctan2(dy, dx)   

                segment_center_x = (x_coords[i] + np.cos(segment_angle) * segment_length / 2).astype(int)
                segment_center_y = (y_coords[i] + np.sin(segment_angle) * segment_length / 2).astype(int)
                segment_mean_width = np.mean(corrected_widths[i:i + segment_length]).astype(int)

                segment_left_x = (segment_center_x - np.sin(segment_angle) * segment_mean_width / 2).astype(int)
                segment_right_x = (segment_center_x + np.sin(segment_angle) * segment_mean_width / 2).astype(int)
                segment_top_y = (segment_center_y + np.cos(segment_angle) * segment_mean_width / 2).astype(int)
                segment_bottom_y = (segment_center_y - np.cos(segment_angle) * segment_mean_width / 2).astype(int)
               
                cv2.line(vis_image_2, 
                         (segment_left_x, segment_top_y),
                         (segment_right_x, segment_bottom_y),
                         (0, 255, 0),
                         2)
                
        visualization = ProcessVisualization("Width Estimation: Segmentation Edges")
        visualization.explanation = "The mask edges are shown in green, and the center of the mask is shown in blue."
        visualization.set_image(vis_image1)
        visualization.extra_detail_only = True
        self.visualizations.append(visualization)

        visualization = ProcessVisualization("Width Estimation: Width Calculation")
        visualization.explanation = f"The center-line of the segmentation (blue line) is the points between the right and left side of " \
                                    f"the mask at each row. The width is first calculated as the number of pixels between the right and " \
                                    f"left side of the mask along a particular row of the image. Then the angle between each " \
                                    f"{self.parameters.pixel_width_segment_length}th pixel along the center-line (red lines) is used to " \
                                    f"adjust the original width, these adjusted widths are shown in green." 
        visualization.set_image(vis_image_2)
        self.visualizations.append(visualization)

        for i, visualization_data in enumerate(self.width_vis_data_all):
            corrected_widths = visualization_data["corrected_widths"]


            histogram_image = self.make_histogram(corrected_widths,
                                                  min_percentile=5,
                                                  max_percentile=95,
                                                  step_size=0.5,
                                                  mask_num=i+1,
                                                  title="Width Histogram Mask {}".format(i + 1),
                                                  xlabel="Width (pixels)",
                                                  used_value=visualization_data["width_estimate"],
                                                  used_value_label="{} percentile".format(self.parameters.pixel_width_percentile))
            
            visualization_hist = ProcessVisualization("Width Estimates Histogram {}".format(i + 1))
            visualization_hist.explanation = f"A histogram of the width estimates for mask {i + 1}. The {self.parameters.pixel_width_percentile} " \
                                                "percentile is shown as a dashed red line. The tails can be long due to noisy width " \
                                                "values, so these are cut off at approximately the 5th and 95th percentiles."
            visualization_hist.set_image(histogram_image)
            visualization_hist.extra_detail_only = True
            self.visualizations.append(visualization_hist)
        
        self.num_extra_detail_visualizations = 1 + len(self.width_vis_data_all)
    

class WidthCorrection(TrunkAnalyzerAbstractOperation):
    """Apply a linear correction to the width estimates based on the x position of the tree in the image."""
    def __init__(self, parameters: ParametersWidthEstimation):
        super().__init__(parameters)
        self.display_name = "Width Correction"

    def run(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Apply a linear correction to the width estimates based on the x position of the tree in the image."""
        
        x_positions_in_image_rel_middle = abs(trunk_analyzer_data.x_positions_in_image - int(trunk_analyzer_data.original_width / 2))
        trunk_analyzer_data.object_widths = trunk_analyzer_data.object_widths + self.parameters.width_correction_slope * x_positions_in_image_rel_middle \
                                + self.parameters.width_correction_intercept

        return trunk_analyzer_data
    
    def create_visualization(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Display visualization of what the operation is doing."""
        
        vis_image = trunk_analyzer_data.visualize_segmentation(add_mask_num=self.add_mask_num(trunk_analyzer_data))       
        
        visualization = ProcessVisualization(self.display_name)
        visualization.explanation = "The width correction operation applies a linear correction to the width estimates based on the x position " \
                                    "of the tree in the image, which is indicated by the red line in the image."

        visualization.set_image(vis_image)

        visualization.add_metric("Width Correction Slope (mm/px):", self.parameters.width_correction_slope * 1000, num_decimal_places=4)
        visualization.add_metric("Width Correction Intercept (mm):", self.parameters.width_correction_intercept * 1000, num_decimal_places=4)

        self.visualizations = [visualization]

        if trunk_analyzer_data.num_instances is None:
            return
        
        for i, (x_pos, width) in enumerate(zip(trunk_analyzer_data.x_positions_in_image, trunk_analyzer_data.object_widths)):
            mask_num = i + 1
            visualization.add_metric("", "")
            visualization.add_metric("Mask {}:".format(mask_num), "")
            visualization.add_metric("X Position (pixels):", x_pos)
            visualization.add_metric("Width (cm):", round(width*100, 2))
            visualization.add_metric("", "")
            cv2.line(vis_image, (x_pos, 0), (x_pos, trunk_analyzer_data.height), (0, 0, 255), 2)


class TrunkAnalyzer:
    """Post segmentation analyzer to filter the masks and calculate the depth and width of the trunks/posts"""
    
    def __init__(self, package_paths: PackagePaths, combine_segmenter: bool=True, create_vis_flag: bool=False):
        """
        Args:
            package_paths (PackagePaths): Class that holds the paths to the package directories.
            combine_segmenter (bool): Whether to incorporate the TrunkSegmenter class into the TrunkAnalyzer class.
            create_vis_flag (bool): Whether to create visualizations for each of the operations.
        """  
        
        self.package_paths = package_paths

        self.parameters = ParametersWidthEstimation.load_from_yaml(self.package_paths.config_file_path)

        self.combine_segmenter = combine_segmenter

        self.create_vis_flag = create_vis_flag
        
        # Create all the operations 
        self.object_type_filter = FilterObjectType(self.parameters)
        self.largest_seg_finder = LargestSegmentFinder(self.parameters)
        self.filter_nms = FilterNMS(self.parameters)
        self.depth_calculation = DepthCalculation(self.parameters)
        self.filter_depth = FilterDepth(self.parameters)
        self.filter_edge = FilterEdge(self.parameters)
        self.filter_position = FilterPosition(self.parameters)
        self.width_estimation = WidthEstimation(self.parameters)
        self.width_correction = WidthCorrection(self.parameters)

        self.operations: List[TrunkAnalyzerAbstractOperation] = []
        
        if self.combine_segmenter:
            self.trunk_segmenter = TrunkSegmenter(self.package_paths, parameters=self.parameters)
            self.operations.append(self.trunk_segmenter)
        else:
            self.trunk_segmenter = None
        
        self.operations.extend([self.object_type_filter,
                               self.largest_seg_finder,
                               self.filter_nms, 
                               self.depth_calculation, 
                               self.filter_depth, 
                               self.filter_edge, 
                               self.filter_position,
                               self.width_estimation,
                               self.width_correction])
        
        self.set_operation_usage()
    
    def set_operation_usage(self):
        """Set the employ_operation attribute for each operation based on the parameters."""
        self.filter_nms.employ_operation = self.parameters.include_nms_filter
        self.filter_depth.employ_operation = self.parameters.include_depth_filter
        self.filter_edge.employ_operation = self.parameters.include_edge_filter
        self.filter_position.employ_operation = self.parameters.include_position_filter
        self.width_correction.employ_operation = self.parameters.include_width_correction
    
    def get_parameters(self):
        """Get the parameters for the width estimation."""
        return self.parameters
    
    def set_parameters(self, parameters: ParametersWidthEstimation):
        """Set the parameters for the width estimation."""
        self.parameters = parameters
        for operation in self.operations:
            operation.set_parameters(parameters)   
        self.set_operation_usage()

    def reload_parameters(self):
        """Reload the parameters from the config file."""
        self.parameters = ParametersWidthEstimation.load_from_yaml(self.package_paths.config_file_path)
        self.set_parameters(self.parameters)
    
    def get_width_estimation(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Run the full algorithm to get the width estimation on a new image.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data class with the image to process.
            
        Returns:
            TrunkAnalyzerData: Data class with the results of the width estimation.
        """
        for operation in self.operations:
            if operation.employ_operation:
                trunk_analyzer_data = operation.check_and_run(trunk_analyzer_data)    

        return trunk_analyzer_data

    def get_width_estimation_pf(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Helper function to run the particle filter code on a new image."""

        trunk_analyzer_data = self.get_width_estimation(trunk_analyzer_data)
        
        if trunk_analyzer_data.num_instances is not None:
            # Switch sign on x_pos and y_pos to match the coordinate system of the particle filter
            trunk_analyzer_data.object_locations[:, 0] = -trunk_analyzer_data.object_locations[:, 0]
            trunk_analyzer_data.object_locations[:, 1] = -trunk_analyzer_data.object_locations[:, 1]

        return trunk_analyzer_data

    # def get_width_estimation_map_making(self, rgb_img: np.ndarray, depth_img: np.ndarray, show_seg=False, apply_width_correction=False):
    #     """Helper function to run the map making code on a new image."""

    #     tree_locations, tree_widths, classes, img_x_positions, seg_img = self.get_width_estimation(depth_img, rgb_image=rgb_img, apply_width_correction=apply_width_correction)

    #     if show_seg:
    #         return tree_locations, tree_widths, classes, img_x_positions, seg_img
    #     else:
    #         return tree_locations, tree_widths, classes, img_x_positions


class TrunkAnalyzerAnalyzer(TrunkAnalyzer):
    """Expands the TrunkAnalyzer class to include the ability to run a performance analysis on the each of the operations and the full pipeline."""
    def __init__(self, package_paths: PackagePaths, combine_segmenter: bool=True, create_vis_flag: bool=False):
        super().__init__(package_paths, combine_segmenter, create_vis_flag)

        self.package_paths = package_paths

        self.image_num = 0

    def reset_for_analysis(self):
        """Reset the variables for the performance analysis."""
        self.analysis_data = None
        
        self.save_model = False
        self.save_segmented_images = False
        self.gt_datasets = False

    def set_visualization_flag(self, create_vis_flag: bool):
        """Set the create_vis_flag for the operations."""
        self.create_vis_flag = create_vis_flag
        for operation in self.operations:
            operation.create_vis_flag = create_vis_flag
        

    def get_width_estimation(self, trunk_analyzer_data: TrunkAnalyzerData):
        """Get the width estimation on a new image.
        
        Args:
            trunk_analyzer_data (TrunkAnalyzerData): Data class with the image to process.
            
        Returns:
            TrunkAnalyzerData: Data class with the results of the width estimation.
        """

        for operation in self.operations:
            if operation.employ_operation:                
                trunk_analyzer_data = operation.run_and_analyze(trunk_analyzer_data, create_vis_flag=self.create_vis_flag)
        
        return trunk_analyzer_data  
    
    def get_visualizations(self):
        """Get the visualizations for the operations.
        
        Returns:
            List[ProcessVisualization]: List of visualizations for the operations.
        """
        visualizations = []
        for operation in self.operations:
            if not operation.employ_operation:
                continue
            if operation.visualizations is None:
                for i in range(operation.num_visualizations):
                    visualizations.append(None)
            else:
                for visualization in operation.visualizations:
                    if visualization.extra_detail_only:
                        continue
                    visualizations.append(visualization)
        return visualizations

    def do_performance_analysis(self, 
                                gt_datasets: bool=False,
                                save_segmented_images: bool=False, 
                                save_model: bool=False,
                                hardware_notes: str="",
                                only_nth_images: int=1,
                                from_app: bool=False,
                                progress_callback: Callable[[int, int], None] = None):
        """Run the performance analysis on the operations.
        
        Args:
            gt_datasets (bool): Whether the datasets have ground truth widths.
            save_segmented_images (bool): Whether to save the segmented images.
            save_model (bool): Whether to save the model in the results directory.
            hardware_notes (str): Notes about the hardware used for the analysis.
            only_nth_images (int): Only run the analysis on every nth image.
            from_app (bool): Whether the analysis is being run from the app.
            progress_callback (Callable[[int, int], None]): Callback function to update the progress bar.
            """

        if hardware_notes == "":
            logging.warning("Please provide hardware notes for the analysis.")

        self.reset_for_analysis()

        self.save_model = save_model
        self.save_segmented_images = save_segmented_images
        self.gt_datasets = gt_datasets

        self.only_nth_images = only_nth_images
        self.from_app = from_app

        results_save_dir_path = self.get_save_dir_for_analysis(self.package_paths.analysis_results_dir)
        self.package_paths.set_current_analysis_data(results_save_dir_path)

        logging.basicConfig(filename=os.path.join(self.package_paths.current_analysis_results_dir, "analysis_log.log"), level=logging.INFO)

        self.settings_dict = {}

        self.log_analysis_info()
        self.save_analysis_info()
        self.save_settings_json(hardware_notes=hardware_notes)

        self.total_num_images = int(self.get_total_num_files() / self.only_nth_images)
        self.current_image_num = 0
        self.progress_callback = progress_callback
       
        self.abort_analysis_flag = False

        for operation in self.operations:
            operation.reset_performance_analysis_data()
        
        for dataset_base_dir in self.package_paths.analysis_gt_data_dirs:
            if self.abort_analysis_flag:
                return
            self.run_analysis_on_dataset(dataset_base_dir)

        self.save_analysis_results()

        for operation in self.operations:
            operation.print_performance_data()       
    
    def abort_analysis(self):
        self.abort_analysis_flag = True

    def log_analysis_info(self):
        """Log the information about the analysis."""

        logging.info("Starting width estimation performance analysis on the following datasets: ")

        for dataset_base_dir in self.package_paths.analysis_gt_data_dirs:
            logging.info(dataset_base_dir.split("/")[-1] + ": ({} images)".format(len(os.listdir(os.path.join(dataset_base_dir, "rgb")))))

        self.log_seperator()


        logging.info("Using the following operations: ")
        for operation in self.operations:
            if operation.employ_operation:
                logging.info(operation.name)
        
        self.log_seperator()

        logging.info("Using the following parameters: ")
        self.parameters.log_settings()

        self.log_seperator()

        logging.info("Results will be saved in: " + self.package_paths.current_analysis_results_dir)
        logging.info("Saving segmented images: " + str(self.save_segmented_images))
        logging.info("Saving the model: " + str(self.save_model))
        logging.info("Using ground truth datasets: " + str(self.gt_datasets))

    def log_seperator(self):
        """Log a seperator in the log file."""
        for _ in range(3):
            logging.info("--------------------------------------------------")

    def save_analysis_info(self):
        """Save the information about the analysis in the results directory."""

        self.settings_dict["analysis_name"] = os.path.basename(self.package_paths.current_analysis_results_dir)
        
        self.settings_dict["save_model"] = self.save_model
        if self.save_model:
            # save the yolo model in the save_dir
            yolo_model_path = os.path.join(self.package_paths.model_dir, self.parameters.yolo_model_to_use)
            yolo_model_save_path = os.path.join(self.package_paths.current_analysis_results_dir, self.parameters.yolo_model_to_use)
            os.system(f"cp {yolo_model_path} {yolo_model_save_path}")
            self.settings_dict["yolo_model_name"] = self.parameters.yolo_model_to_use

        #save this file in the save_dir
        os.system(f"cp {__file__} {self.package_paths.current_analysis_results_dir}")
        self.settings_dict["code_file_name"] = os.path.basename(__file__)

        # save the parameters in the save_dir
        self.parameters.save_to_yaml(self.package_paths.current_analysis_config_path)
        self.settings_dict["config_file_name"] = os.path.basename(self.package_paths.current_analysis_config_path)

        self.settings_dict["save_segmented_images"] = self.save_segmented_images

        if self.save_segmented_images:
            self.settings_dict["segmented_images_dir_name"] = "segmented_images"
            self.segmented_images_dir = os.path.join(self.package_paths.current_analysis_results_dir, "segmented_images")
            os.makedirs(self.segmented_images_dir)

    def save_settings_json(self, hardware_notes: str = ""):
        """Add any other settings to the json file and save it in the save_dir."""

        self.settings_dict["operations"] = [operation.name if operation.employ_operation else None for operation in self.operations]
        self.settings_dict["dataset_names"] = [os.path.basename(dataset_base_dir) for dataset_base_dir in self.package_paths.analysis_gt_data_dirs]
        self.settings_dict["gt_datasets"] = self.gt_datasets
        self.settings_dict["log_file_name"] = "analysis_log.log"
        self.settings_dict["results_data_file_name"] = os.path.basename(self.package_paths.analysis_results_data_path)
        self.settings_dict["results_summary_file_name"] = os.path.basename(self.package_paths.analysis_results_summary_path)
        self.settings_dict["hardware_notes"] = hardware_notes
        self.settings_dict["only_nth_images"] = self.only_nth_images

        settings_json_path = os.path.join(self.package_paths.current_analysis_results_dir, "settings.json")
        with open(settings_json_path, "w") as f:
            json.dump(self.settings_dict, f, indent=4)

    def run_analysis_on_dataset(self, dataset_base_dir: str):
        """Run the analysis on a dataset."""

        gt_data_dir = os.path.join(dataset_base_dir, "data")
        rgb_dir = os.path.join(dataset_base_dir, "rgb")
        depth_dir = os.path.join(dataset_base_dir, "depth")

        rgb_image_names = os.listdir(rgb_dir)
        rgb_image_names.sort()

        if self.only_nth_images != 1:
            rgb_image_names = rgb_image_names[::self.only_nth_images]

        for rgb_image_name in tqdm(rgb_image_names, desc="Processing images for {}".format(dataset_base_dir.split("/")[-1]), total=len(rgb_image_names)):
        # for rgb_image_name in rgb_image_names:
            if self.abort_analysis_flag:
                return
            self.run_analysis_on_image(rgb_image_name, rgb_dir, depth_dir, gt_data_dir)
        
    def run_analysis_on_image(self, rgb_image_name: str, rgb_dir: str, depth_dir: str, gt_data_dir: str):
        """Run the analysis on a single image."""

        self.current_image_num += 1
        if self.progress_callback is not None:
            self.progress_callback(self.current_image_num, self.total_num_images)
        
        rgb_image_path = os.path.join(rgb_dir, rgb_image_name)
        depth_image_path = os.path.join(depth_dir, rgb_image_name)

        if self.gt_datasets:
            gt_data_path = os.path.join(gt_data_dir, rgb_image_name.split(".")[0] + ".json")
            with open(gt_data_path, "r") as f:
                gt_data = json.load(f)
        else:
            gt_data = None
        
        trunk_data = TrunkAnalyzerData.from_image_paths(rgb_image_path, depth_image_path)

        start_time = time.time()
        trunk_data = self.get_width_estimation(trunk_data)
        run_time = time.time() - start_time

        # cv2.imshow("Segmentation", seg_image)
        # cv2.waitKey(20)

        if trunk_data.num_instances is None:
            # tqdm.write("No tree detected for image: " + rgb_image_name)
            # logging.info("No tree detected for image: " + rgb_image_name)  
            self.save_results_for_image(rgb_image_name, 
                                        rgb_dir, 
                                        trunk_data,
                                        run_time,
                                        gt_data)
            return
            
        if self.gt_datasets:
            gt_x_position = gt_data["x_position_in_image"]
            
            if trunk_data.num_instances > 1:
                distance_error = np.abs(trunk_data.x_positions_in_image - gt_x_position)
                best = np.argmin(distance_error)
                matched_width = trunk_data.object_widths[best]
                matched_ground_truth_x_position = trunk_data.x_positions_in_image[best]
            else:
                matched_width = trunk_data.object_widths[0]
                matched_ground_truth_x_position = trunk_data.x_positions_in_image[0]

            if abs(matched_ground_truth_x_position - gt_x_position) > 50:
                matched_width = None
                matched_ground_truth_x_position = None
                
        else:
            matched_width = None
            matched_ground_truth_x_position = None
        
        self.save_results_for_image(rgb_image_name, 
                                    rgb_dir, 
                                    trunk_data,
                                    run_time,
                                    gt_data, 
                                    matched_width, 
                                    matched_ground_truth_x_position)        
    
    def save_results_for_image(self, 
                               rgb_image_name: str, 
                               rgb_dir: str,
                               trunk_data: TrunkAnalyzerData,
                               run_time: float,
                               gt_data: dict = None,
                               matched_width: float = None,
                               matched_ground_truth_x_position: int = None,):
        """Save the results for an image in the results dataframe."""

        data_dict = {}
        
        data_dict["dataset"] = rgb_dir.split("/")[-2]
        data_dict["image"] = rgb_image_name
        data_dict["estimated_widths"] = trunk_data.object_widths
        data_dict["x_positions_in_image"] = trunk_data.x_positions_in_image
        data_dict["classes"] = trunk_data.classes
        data_dict["object_x_positions"] = trunk_data.object_locations[:, 0] if trunk_data.num_instances is not None else None
        data_dict["object_y_positions"] = trunk_data.object_locations[:, 1] if trunk_data.num_instances is not None else None
        data_dict["run_time"] = run_time

        if gt_data:
            data_dict["matched_width"] = matched_width
            data_dict["matched_ground_truth_x_position"] = matched_ground_truth_x_position
            data_dict["ground_truth_width"] = gt_data["ground_truth_width"]
            data_dict["gt_x_position_in_image"] = gt_data["x_position_in_image"]
        
        for operation in self.operations:
            data_dict = operation.add_performance_data(data_dict)

        if self.analysis_data is None:
            self.analysis_data = pd.DataFrame([data_dict])
        else:
            # remove nones
            data_dict = {k: v for k, v in data_dict.items() if v is not None}
            self.analysis_data = pd.concat([self.analysis_data, pd.DataFrame([data_dict])], ignore_index=True)

        if self.save_segmented_images:
            segmented_image_path = os.path.join(self.segmented_images_dir, rgb_image_name)
            cv2.imwrite(segmented_image_path, trunk_data.visualize_segmentation())

    def save_analysis_results(self): 
        """Save the results of the analysis to a csv file."""

        summary_data = {}
        summary_data["mean_run_time"] = self.analysis_data["run_time"].mean()
        summary_data["total_images"] = len(self.analysis_data)

        if self.gt_datasets:
            summary_data["mean_width_error"] = np.mean(np.abs(self.analysis_data["matched_width"] - self.analysis_data["ground_truth_width"]))
            summary_data["mean_x_position_error"] = np.mean(np.abs(self.analysis_data["matched_ground_truth_x_position"] - self.analysis_data["gt_x_position_in_image"]))
            summary_data["num_images_missing"] = len(self.analysis_data[self.analysis_data["matched_width"].isnull()])
        
        for operation in self.operations:
            if not operation.employ_operation:
                continue
            summary_data[operation.name + "_total_time"] = operation.total_time
            summary_data[operation.name + "_mean_time"] = operation.total_time / operation.total_calls
            summary_data[operation.name + "_total_calls"] = operation.total_calls
            summary_data[operation.name + "_total_applied"] = operation.total_times_applied

        summary_df = pd.DataFrame([summary_data])
        
        self.analysis_data.to_csv(self.package_paths.analysis_results_data_path, index=False)

        summary_df.to_csv(self.package_paths.analysis_results_summary_path, index=False)

    @staticmethod
    def get_save_dir_for_analysis(results_base_dir: str) -> str:
        """
        Get the directory to save the results of the analysis in. The directory is created in the data directory of the package. 
        The directory is named "performance_test_results" and then a subdirectory with the current date and time is created.
        """

        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        results_base_dir = os.path.join(results_base_dir, "results_" + date_time)

        # make save_dir if it does not exist, although i have no idea why this would ever happen
        if not os.path.exists(results_base_dir):
            os.makedirs(results_base_dir)
        else:
            print("Directory already exists")
            exit()
        
        return results_base_dir
    
    def get_total_num_files(self):
        total_files = 0
        for dataset_base_dir in self.package_paths.analysis_gt_data_dirs:
            rgb_dir = os.path.join(dataset_base_dir, "rgb")
            total_files += len(os.listdir(rgb_dir))
        return total_files



