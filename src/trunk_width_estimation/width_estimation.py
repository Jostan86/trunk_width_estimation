
import numpy as np
from ultralytics import YOLO
import cv2
from skimage.measure import label, regionprops
import os
import yaml
import time
from .package_paths import PackagePaths
from .parameters import ParametersWidthEstimation

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
    
class TrunkSegmenter:
    """Class to segment the trunks in an image using the YOLO model."""
    
    def __init__(self, package_paths: PackagePaths):
        """
        Args:
            package_paths (PackagePaths): Class that holds the paths to the package directories.
        """
        
        parameters = ParametersWidthEstimation.load_from_yaml(package_paths.config_file_path)    

        # Load the model
        weight_path = os.path.join(package_paths.model_dir, parameters.yolo_model_to_use)
        self.yolo_model = YOLO(weight_path, task="segment")

        # Put an image through the model to initialize the model
        startup_image = cv2.imread(package_paths.startup_image_path)
        results_dict, results = self.get_results(startup_image)
    
    
    def get_results(self, image: np.ndarray) -> dict:
        """Do the prediction on an image and return the results.

        Args:
            image (np.ndarray): Image that was sent to the model.

        Returns:
            dict: Dictionary of the results from the model.
        """
        
        image = check_image_size(image)

        # Do the prediction 
        # iou:  Intersection Over Union threshold for Non-Maximum Suppression (NMS). Lower values result in fewer 
        #       detections by eliminating overlapping boxes, useful for reducing duplicates.
        # conf: Object confidence threshold. Lower values will result in more detections but also lower confidence.
        results = self.yolo_model.predict(image, 
                                          imgsz=(image.shape[0], image.shape[1]), 
                                          iou=0.01, 
                                          conf=0.7,
                                          verbose=False, 
                                          )
        
        # Move the results to the cpu
        results = results[0].cpu()

        # Get the data from the results
        num_instances = len(results.boxes) if len(results.boxes) > 0 else None
        confidences = results.boxes.conf.numpy() if num_instances is not None else None
        masks = results.masks.data.numpy().astype(np.uint8) if num_instances is not None else None
        classes = results.boxes.cls.numpy() if num_instances is not None else None

        # Initialize the array that will hold the indices of the instances that are being kept after filtering
        results_kept = np.arange(num_instances) if num_instances is not None else None

        # package the results into a dictionary
        results_dict = {"confidences": confidences, "masks": masks, "classes": classes,
                        "results_kept": results_kept, "num_instances": num_instances}

        return results_dict, results

class TrunkAnalyzer:
    """Post segmentation analyzer to filter the masks and calculate the depth and width of the trunks"""
    
    def __init__(self, package_paths: PackagePaths, combine_segmenter: bool=True):
        """
        Args:
            package_paths (PackagePaths): Class that holds the paths to the package directories.
            combine_segmenter (bool): Whether to incorporate the TrunkSegmenter class into the TrunkAnalyzer class.
        """  
        
        self._parameters = ParametersWidthEstimation.load_from_yaml(package_paths.config_file_path)

        self._ignore_classes = np.array(self._parameters.ignore_classes)

        self._rgb_image = None
        self._depth_image = None

        self._confidences = None
        self._masks = None
        self._classes = None
        self._results_kept = None

        self._depth_calculated = False
        self._width_calculated = False
        
        self._depth_median = None
        self._depth_percentile = None

        self._num_instances = None
        self._cropped_width = None
        self._original_width = None
        self._height = None
        self._num_pixels = None
        
        self._tree_widths = None
        self._img_x_positions = None
        self._tree_locations = None

        self._combine_segmenter = combine_segmenter

        if self._combine_segmenter:
            self._trunk_segmenter = TrunkSegmenter(package_paths)
        else:
            self._trunk_segmenter = None
    
    def _calculate_depth(self, top_ignore: float, bottom_ignore: float, min_num_points: int, 
                        depth_filter_percentile: float):
        """
        Calculates the best estimate of the distance between the tree and camera. Calculates the median depth and 
        percentile depth for each mask. Also filters out masks that have less than min_num_points valid points 
        in the region defined by top_ignore and bottom_ignore.

        Args:
            top_ignore (float): Proportion of the top of the image to ignore mask points in.
            bottom_ignore (float): Proportion of the bottom of the image to ignore mask points in.
            min_num_points (int): Minimum number of valid depth points needed to keep the mask.
            depth_filter_percentile (float): Percentile of the depth values to use for the percentile depth estimate.
           
        """

        # Initialize arrays to store the depth values and the tree locations
        self._depth_median = np.zeros(self._num_instances)
        self._depth_percentile = np.zeros(self._num_instances)

        # Make boolean array of indices to keep
        keep = np.ones(self._num_instances, dtype=bool)

        # Calculate the top and bottom ignore values in pixels
        top_ignore = int(top_ignore * self._height)
        bottom_ignore = self._height - int(bottom_ignore * self._height)

        # Loop through each mask
        for i, mask in enumerate(self._masks):

            # Make copy of mask array
            # TODO: is this necessary?
            mask_copy = mask.copy()

            # Zero out the top and bottom ignore regions
            mask_copy[:top_ignore, :] = 0
            mask_copy[bottom_ignore:, :] = 0

            # If there are no points in the mask, remove the segment
            if np.sum(mask_copy) == 0:
                keep[i] = False
                continue

            # Convert mask copy to a boolean array
            mask_copy = mask_copy.astype(bool)

            # Make a 1D array of the masked portions of the depth image
            masked_depth = self._depth_image[mask_copy]

            # Remove zero values from the masked depth array and remove the mask if there are too few points left
            masked_depth = masked_depth[masked_depth != 0]
            if masked_depth.shape[0] < min_num_points:
                keep[i] = False
                continue

            # Calculate median and percentile depth
            self._depth_median[i] = np.median(masked_depth) / 1000
            self._depth_percentile[i] = np.percentile(masked_depth, depth_filter_percentile) / 1000

        # Update the arrays
        self._depth_calculated = True
        self._remove_faulty_instances(keep)
            
    def _calculate_width(self):
        """Calculates the best estimate of the width of the tree in meters.
        """

        self._tree_widths = np.zeros(self._num_instances)
        self._tree_locations = np.zeros((self._num_instances, 2))

        # Loop through each mask
        for i, (mask, depth) in enumerate(zip(self._masks, self._depth_median)):

            # Get the diameter of the tree in pixels
            pixel_width = self._calculate_pixel_width(mask)

            horz_fov = self._parameters.camera_horizontal_fov

            # Calculate the width of the image in meters at the depth of the tree
            image_width_m = depth * np.tan(np.deg2rad(horz_fov / 2)) * 2

            # Calculate the distance per pixel
            distperpix = image_width_m / self._original_width

            # Calculate the diameter of the tree in meters
            diameter_m = pixel_width * distperpix

            # If there are no valid widths, set the width to 0, otherwise set it to the max width
            if len(diameter_m) == 0:
                self._tree_widths[i] = 0
            else:
                self._tree_widths[i] = np.max(diameter_m)

            # Calculate the x location of the tree in the image (in pixels) by taking the median of the mask points in x
            # TODO: This value should be able to be used for the img_x_positions instead of recalculating it later
            self._img_x_positions[i] = np.median(np.where(mask)[1])
            self._tree_locations[i, 1] = self._depth_median[i]
            self._tree_locations[i, 0] = (self._img_x_positions[i] - (self._original_width / 2)) * distperpix

        self._width_calculated = True

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
        y_coords = np.arange(mask.shape[0])[valid_rows]
        
        # Calculate the x_coords as the midpoint between the leftmost and rightmost points, so the center of the trunk
        x_coords = ((leftmost_pixel_columns + rightmost_pixel_columns) / 2).astype(int)

        # Calculate the angle at multiple small segments along the trunk and use them to calculate a corrected width
        segment_length = self._parameters.pixel_width_segment_length 
        widths = rightmost_pixel_columns - leftmost_pixel_columns # Width of the trunk at each row
        corrected_widths = np.zeros_like(widths, dtype=float)
        # TODO: can this be vectorized?
        for i in range(0, len(y_coords) - segment_length, segment_length):
            dy = y_coords[i + segment_length] - y_coords[i]
            dx = x_coords[i + segment_length] - x_coords[i]
            angle = np.arctan2(dy, dx)
            # Correct the width based on the angle of the tree at that location
            corrected_widths[i:i + segment_length] = widths[i:i + segment_length] * np.abs(np.sin(angle))

        return corrected_widths

    def _mask_filter_nms(self, overlap_threshold: float):
        """Apply non-maximum suppression (NMS) to remove overlapping masks of different classes. 
        
        Yolo applies NMS, but only to individual classes, so this is a second NMS to remove any overlapping masks from 
        different classes. There is technically an option to do this in Yolo, but it can't have a distinct iou threshold
        from the first NMS.

        Args:
            overlap_threshold (float): Overlap threshold for NMS. If the overlap between two masks is greater than this
            value, the mask with the lower score will be removed.
        """

        # TODO: is this necessary? Also, there has to be a package that does NMS efficiently already
        mask_nms = self._masks.copy()
        score_nms = self._confidences.copy()

        # Sort masks by score
        indices = np.argsort(-score_nms)
        mask_nms = mask_nms[indices]

        # Array to keep track of whether an instance is suppressed or not
        suppressed = np.zeros((len(mask_nms)), dtype=bool)

        # For each mask, compute overlap with other masks and suppress overlapping masks if their score is lower
        for i in range(len(mask_nms) - 1):
            # If already suppressed, skip
            if suppressed[i]:
                continue
            # Compute overlap with other masks
            overlap = np.sum(mask_nms[i] * mask_nms[i + 1:], axis=(1, 2)) / np.sum(mask_nms[i] + mask_nms[i + 1:],
                                                                                   axis=(1, 2))
            # Suppress masks that are either already suppressed or have an overlap greater than the threshold
            suppressed[i + 1:] = np.logical_or(suppressed[i + 1:], overlap > overlap_threshold)

        # Get the indices of the masks that were not suppressed
        indices_revert = np.argsort(indices)
        suppressed = suppressed[indices_revert]
        not_suppressed = np.logical_not(suppressed)

        # Update the arrays
        self._remove_faulty_instances(not_suppressed)

    def _mask_filter_depth(self, depth_threshold: float):
        """Sort out any outputs that are beyond a given depth threshold. Note that during depth calcuation any
        segments entirely in the top or bottom portions of the image are removed, and any segments with too few
        points in the point cloud are also removed.

        Args:
            depth_threshold (float): Depth threshold in meters. Any masks with a percentile depth greater than this 
            value wll be removed.

        Returns:
            Updates the class arrays to only include the masks that are within the depth threshold.
        """
        keep = self._depth_percentile < depth_threshold
        self._remove_faulty_instances(keep)

    def _mask_filter_edge(self, edge_threshold: float, size_threshold: float):
        """Sort out any outputs with masks that are too close to the edge of the image.
        
        Args:
            edge_threshold (float): Proportion of the image width that is considered the edge.
            size_threshold (float): Proportion of the mask that must be beyond the edge threshold for the mask to be
            removed.
        """

        keep = np.zeros(self._num_instances, dtype=bool)

        edge_threshold = int(edge_threshold * self._cropped_width)

        masks_copy = self._masks.copy()

        for i, mask in enumerate(masks_copy):
            left_edge_pixels = mask[:, :edge_threshold].sum()
            right_edge_pixels = mask[:, -edge_threshold:].sum()
            total_mask_pixels = np.sum(mask)
            if left_edge_pixels / total_mask_pixels > size_threshold or right_edge_pixels / total_mask_pixels > size_threshold:
                continue
            else:
                keep[i] = True

        self._remove_faulty_instances(keep)

    def _mask_filter_position(self, bottom_position_threshold: float, score_threshold: float, 
                             top_position_threshold: float):
        """Filter out masks based on thier vertical position in the image. 
        
        Args:
            bottom_position_threshold (float): Proportion of the image height from the bottom. Masks with their lowest
            point above this threshold will be removed.
            top_position_threshold (float): Proportion of the image height from the top. Masks with their highest point
            above this threshold will be removed.
        """
        
        # TODO: double check the documentation for this function matches the code
        
        
        keep = np.zeros(self._num_instances, dtype=bool)

        bottom_position_threshold = int(bottom_position_threshold * self._height)
        top_position_threshold = int(top_position_threshold * self._height)

        # TODO: is this necessary?
        masks_copy = self._masks.copy()

        for i, mask in enumerate(masks_copy):
            # if self._confidences[i] > score_threshold:
            #     keep[i] = True
            #     continue

            bottom_pixels = mask[-bottom_position_threshold:].sum()
            if bottom_pixels > 0:
                keep[i] = True

            top_pixels = mask[:top_position_threshold].sum()
            if np.isclose(top_pixels, 0):
                keep[i] = False

        self._remove_faulty_instances(keep)

    def _mask_filter_multi_segs(self):
        """Determine if there are multiple segments in the mask and if so keep only the largest one."""

        
        largest_segments = np.zeros_like(self._masks)

        # Loop through each mask
        for i, mask in enumerate(self._masks):
            # Label connected regions in the mask
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

            # If there's only one connected region, no need to process further
            if num_labels == 1:
                largest_segments[i] = mask
            else:
                # Find properties of each connected region
                props = regionprops(labeled_mask)

                # Sort the regions by their area in descending order
                props.sort(key=lambda x: x.area, reverse=True)

                # Keep only the largest connected segment
                largest_segment_mask = labeled_mask == props[0].label

                # Store the largest segment in the result array
                largest_segments[i] = largest_segment_mask.astype(np.uint8)

        self._masks = largest_segments
    
    def _remove_faulty_instances(self, keep_indices: np.ndarray):
        """Update all the active arrays based on the keep_indices array from a filtering operation. Gettting rid of
        instances that have been filtered out.

        Args:
            keep_indices (np.ndarray): Indices of the instances that are being kept after filtering.
        """

        self._confidences = self._confidences[keep_indices]
        self._masks = self._masks[keep_indices]
        self._results_kept = self._results_kept[keep_indices]
        self._classes = self._classes[keep_indices]
        self._num_instances = len(self._masks)
        self._img_x_positions = self._img_x_positions[keep_indices]

        # Update the depth and width arrays if they have been calculated
        if self._depth_calculated:
            self._depth_median = self._depth_median[keep_indices]
            self._depth_percentile = self._depth_percentile[keep_indices]
        if self._width_calculated:
            self._tree_widths = self._tree_widths[keep_indices]
            self._tree_locations = self._tree_locations[keep_indices]

    def _setup_arrays(self, results_dict: dict):
        """Setup the arrays for analysis using the results dict from the model.
        
        Args:
            results_dict (dict): Dictionary of the results from the model.
        """

        self._num_instances = results_dict["num_instances"]
        self._confidences = results_dict["confidences"]
        self._masks = results_dict["masks"]
        self._classes = results_dict["classes"]
        self._results_kept = results_dict["results_kept"]
        
        self._img_x_positions = np.zeros(self._num_instances, dtype=np.int32)

        # Remove any sprinklers if desired
        if self._num_instances is not None:
            keep_indices = np.isin(self._classes, self._ignore_classes, invert=True)
            self._remove_faulty_instances(keep_indices)
            
    def _new_image_reset(self, results_dict: dict, depth_image: np.ndarray):
        """ Resets the class variables for a new image.

        Args:
            results_dict (dict): Results from the image segmentation model
            depth_image (np.ndarray): aligned depth image that corresponds to the image
        """

        self._confidences = None
        self._masks = None
        self._results_kept = None

        self._depth_calculated = False
        self._depth_median = None
        self._depth_percentile = None
        self._tree_locations = None

        self._width_calculated = False
        self._tree_widths = None

        self._setup_arrays(results_dict)

        self._original_width = depth_image.shape[1]

        depth_image = check_image_size(depth_image)

        self._height = depth_image.shape[0]
        self._cropped_width = depth_image.shape[1]

        self._depth_image = depth_image

        self._num_pixels = self._cropped_width * self._height

    def _process_image(self):

        if self._masks is None:
            return

        # Send the masks through all the filters, skipping to the end if the number of instances is 0
        if self._num_instances > 0:
             self._mask_filter_multi_segs()
        if self._num_instances > 1:
            self._mask_filter_nms(overlap_threshold=self._parameters.filter_nms_overlap_threshold)
        if self._num_instances > 0:
            # self.calculate_depth(top_ignore=0.50, bottom_ignore=0.20, min_num_points=500,
            #                      depth_filter_percentile_upper=65, depth_filter_percentile_lower=35)
            self._calculate_depth(top_ignore=self._parameters.depth_calc_top_ignore,
                                 bottom_ignore=self._parameters.depth_calc_bottom_ignore,
                                 min_num_points=self._parameters.depth_calc_min_num_points,
                                 depth_filter_percentile=self._parameters.depth_calc_percentile,
                                 )
        if self._num_instances > 0:
            self._mask_filter_depth(depth_threshold=self._parameters.filter_depth_max_depth)
        if self._num_instances > 0:
            self._mask_filter_edge(edge_threshold=self._parameters.filter_edge_edge_threshold,
                                   size_threshold=self._parameters.filter_edge_size_threshold)
        if self._num_instances > 0:
            self._mask_filter_position(bottom_position_threshold=self._parameters.filter_position_bottom_threshold,
                                       top_position_threshold=self._parameters.filter_position_top_threshold,
                                       score_threshold=self._parameters.filter_position_score_threshold)
            self._calculate_width()

        # # Calculate the x position of the tree in the image
        # self._img_x_positions = np.zeros(self._num_instances, dtype=np.int32)
        # for i, mask in enumerate(self._masks):
        #     # Figure out the x location of the mask
        #     self._img_x_positions[i] = int(np.mean(np.where(mask)[1]))


    def get_width_estimation_pf(self, depth_image: np.ndarray, results_dict: dict = None, rgb_image: np.ndarray = None):
        """Get the width estimation on a new image.
        
        Args:
            depth_image (np.ndarray): Depth image to process.
            results_dict (dict): Results from the image segmentation model. (needed if the segmenter is separate)
            rgb_image (np.ndarray): RGB image to process. (needed if the segmenter is combined)
            
        Returns:
            np.ndarray: Array of the tree locations in the image.
            np.ndarray: Array of the tree widths in the image.
            np.ndarray: Array of the classes of the segmentations.
            np.ndarray: Array of the x positions of the trees in the image.
            np.ndarray: Image with the segmentations plotted on it.
        """

        if self._combine_segmenter:
            if rgb_image is None:
                raise ValueError("rgb_image must be provided if combine_segmenter is True")
            results_dict, results = self._trunk_segmenter.get_results(rgb_image)
        elif results_dict is None:
            raise ValueError("results_dict must be provided if combine_segmenter is False")
        
        # Reset the class variables for a new image
        self._new_image_reset(results_dict, depth_image)

        # Process the image
        self._process_image()

        # If no instances are left, return None for all the values
        if self._num_instances == 0 or self._num_instances is None:
            return None, None, None, None, None

        # Switch sign on x_pos and y_pos to match the coordinate system of the particle filter
        self._tree_locations[:, 0] = -self._tree_locations[:, 0]
        self._tree_locations[:, 1] = -self._tree_locations[:, 1]
        
        # Values obtained from calibrate_widths.py, sept = True
        # self.tree_widths = -0.00625 + (-2.088e-05 * img_x_positions_rel_middle) + (1.058 * self.tree_widths)
        # self.tree_widths = -2.088e-05 * img_x_positions_rel_middle + self.tree_widths
        if self._parameters.do_width_correction:
            img_x_positions_rel_middle = abs(self._img_x_positions - int(self._original_width / 2))
            self._tree_widths = self._tree_widths - self._parameters.width_correction_slope * img_x_positions_rel_middle \
                                - self._parameters.width_correction_intercept

        if self._combine_segmenter:
            results = results[self._results_kept]
            return self._tree_locations, self._tree_widths, self._classes, self._img_x_positions, results.plot()
        else:
            return self._tree_locations, self._tree_widths, self._classes, self._img_x_positions, self._results_kept

    def get_width_estimation_map_making(self, rgb_img, depth_img, show_seg=False):
        """Helper function to run the map making code on a new image."""

        # Run the model on the image
        results_dict, results = self._trunk_segmenter.get_results(rgb_img)

        # Reset the class variables for a new image
        self._new_image_reset(results_dict, depth_img)

        # Process the image
        self._process_image()

        results = results[self._results_kept]

        # If no instances are left, return None for all the values
        if self._num_instances == 0 or self._num_instances is None:
            return None, None, None, None, None

        # Adjust the width based on the x position of the tree in the image
        # img_x_positions_rel_middle = abs(self.img_x_positions - 320)
        # Values obtained from calibrate_widths.py, sept = True
        # self.tree_widths = -0.006246 + (-2.0884248893265422e-05 * img_x_positions_rel_middle) + (1.057907234666699 * self.tree_widths)

        if show_seg:
            return self._tree_locations, self._tree_widths, self._classes, self._img_x_positions, results.plot()
        else:
            return self._tree_locations, self._tree_widths, self._classes, self._img_x_positions



