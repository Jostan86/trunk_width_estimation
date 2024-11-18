
import numpy as np
from ultralytics import YOLO
import cv2
from skimage.measure import label, regionprops
import os
from importlib_resources import files
import yaml
import time

class TrunkSegmenter:
    def __init__(self, config_file_path):
        """Class to segment the trunks in an image using the YOLO model."""
        
        # with open(config_file_path) as f:
        #     config_data = yaml.safe_load(f)     

        # # Load the model
        # weight_path = os.path.join(config_data["project_root_dir"], "data", "models", config_data["model_to_use"])
        self.yolo_model = YOLO("/trunk_width_estimation/models/jazz_s_v8.engine", task="segment")
        # self.yolo_model = YOLO("best_s_500_v8.pt")

        # Put an image through the model to initialize the model
        startup_img_path = "/trunk_width_estimation/data/startup_image.png"
        startup_img_path = str(startup_img_path)
        startup_image = cv2.imread(startup_img_path)
        results1, results = self.get_results(startup_image)

    def get_results(self, image):
        """
        Do the prediction on an image and return the results.

        Args:
            image (): Image that was sent to the model.

        Returns:
            does inference and returns the results
        """
        if image.shape[0] % 32 != 0:
            image = image[:-(image.shape[0] % 32), :, :]
        if image.shape[1] % 32 != 0:
            image = image[:, :-(image.shape[1] % 32), :]

        # Do the prediction and convert to cpu. Agnostic NMS is set to false, which is the default, because I do nms
        # later also, and i need to use a different iou for the 2. Ideally i could send 2 different ious here, but alas.
        # self.start_time = time.time()
        results = self.yolo_model.predict(image, imgsz=(image.shape[0], image.shape[1]), iou=0.01, conf=0.7,
                                          verbose=False, agnostic_nms=False)
        # print("Time to do inference: ", time.time() - self.start_time)

        results = results[0].cpu()

        # Save the number of instances
        num_instances = len(results.boxes) if len(results.boxes) > 0 else None

        # Save the mask and score arrays to the class variables
        confidences = results.boxes.conf.numpy() if num_instances is not None else None
        masks = results.masks.data.numpy().astype(np.uint8) if num_instances is not None else None
        classes = results.boxes.cls.numpy() if num_instances is not None else None

        # Initialize the array that will hold the indices of the instances that are being kept
        results_kept = np.arange(num_instances) if num_instances is not None else None

        # package the results into a dictionary
        results_dict = {"confidences": confidences, "masks": masks, "classes": classes,
                        "results_kept": results_kept, "num_instances": num_instances}

        return results_dict, results

class TrunkAnalyzer:
    def __init__(self, config_file_path="/trunk_width_estimation/config/width_estimation_jazz_config.yaml", combine_segmenter=False):
        """Post segmentation analyzer to filter the masks and calculate the depth and width of the trunks"""

        with open(config_file_path) as f:
            self.config_data = yaml.safe_load(f)    

        self.ignore_classes = np.array(self.config_data["ignore_classes"])

        # Initialize the class variables
        self.results = None
        self.image = None
        self.depth_img = None

        self.confidences = None
        self.masks = None
        self.classes = None
        self.results_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile_upper = None
        self.depth_percentile_lower = None

        self.width_calculated = False
        self.tree_widths = None
        self.img_x_positions = None

        self.num_instances = None
        self.cropped_width = None
        self.original_width = None
        self.height = None
        self.num_pixels = None

        self.tree_locations = None

        self.combine_segmenter = combine_segmenter

        if self.combine_segmenter:
            self.trunk_segmenter = TrunkSegmenter(config_file_path)
        else:
            self.trunk_segmenter = None



    def setup_arrays(self):
        """Setup the arrays for analysis using the results dict from the model."""

        self.num_instances = self.results["num_instances"]
        self.confidences = self.results["confidences"]
        self.masks = self.results["masks"]
        self.classes = self.results["classes"]
        self.results_kept = self.results["results_kept"]

        # Remove any sprinklers if desired
        if self.num_instances is not None:
            keep_indices = np.isin(self.classes, self.ignore_classes, invert=True)
            self.update_arrays(keep_indices)


    def update_arrays(self, keep_indices):
        """Update all the arrays based on the keep_indices array from a filtering operation.

        Args:
            keep_indices (): Indices of the instances that are being kept.

        Returns:
            Filters all the active arrays based on the keep_indices array.
        """

        self.confidences = self.confidences[keep_indices]
        self.masks = self.masks[keep_indices]
        self.results_kept = self.results_kept[keep_indices]
        self.classes = self.classes[keep_indices]
        self.num_instances = len(self.masks)

        if self.depth_calculated:
            self.depth_median = self.depth_median[keep_indices]
            self.depth_percentile_upper = self.depth_percentile_upper[keep_indices]
            self.depth_percentile_lower = self.depth_percentile_lower[keep_indices]

        if self.width_calculated:
            self.tree_widths = self.tree_widths[keep_indices]
            self.tree_locations = self.tree_locations[keep_indices]

    def get_width_pix(self, mask):
        """Get the width of the segmentation in pixels. Note that this method is different than in the paper, but
        it produces a similar results and is significantly faster."""

        # Get the leftmost and rightmost points in the mask for each row
        leftmost = mask.argmax(axis=1)
        rightmost = mask.shape[1] - 1 - np.flip(mask, axis=1).argmax(axis=1)

        # Remove rows where no mask is present, which is rows where the leftmost pixel is not zero and right most is not
        # the last column
        valid_rows = np.logical_and(leftmost != 0, rightmost != mask.shape[1] - 1)
        y_coords = np.arange(mask.shape[0])[valid_rows]
        leftmost = leftmost[valid_rows]
        rightmost = rightmost[valid_rows]

        # Calculate the midpoint between the leftmost and rightmost points, so the center of the trunk
        midpoints = ((leftmost + rightmost) / 2).astype(int)
        x_coords = midpoints

        # Calculate local angles use them to calculate a corrected width
        segment_length = 15 # Length of the segment to use for calculating the local angle, 15 was arbitrarily chosen
        widths = rightmost - leftmost # Width of the trunk at each row
        corrected_widths = np.zeros_like(widths, dtype=float)
        for i in range(0, len(y_coords) - segment_length, segment_length):
            dy = y_coords[i + segment_length] - y_coords[i]
            dx = x_coords[i + segment_length] - x_coords[i]
            angle = np.arctan2(dy, dx)
            # Correct the width based on the angle of the tree at that location
            corrected_widths[i:i + segment_length] = widths[i:i + segment_length] * np.abs(np.sin(angle))

        # Cut off the top 30% of the values and the bottom 30% of the values
        cut_off_dist = int(corrected_widths.shape[0] * 0.3)
        corrected_widths_trimmed = np.cumsum(corrected_widths)[:-cut_off_dist]
        corrected_widths_trimmed = corrected_widths_trimmed[cut_off_dist:]

        # Calculate the difference between elements separated by a window of 20 in return_distance_axis_trimmed (effectively a discrete derivative)
        return_distance_axis_derv = corrected_widths_trimmed[20:] - corrected_widths_trimmed[:-20]

        # Determine how many of the smallest elements in return_distance4 to consider (40% of the length of the array)
        k = int(return_distance_axis_derv.shape[0] * 0.4)

        # Find the indices of the k smallest elements in return_distance_axis_derv
        idx1 = np.argpartition(return_distance_axis_derv, k)[:k]

        # Calculate the real indices in the original return_distance_axis array
        real_idx = idx1 + 10 + cut_off_dist

        # Retrieve the distances at the calculated indices
        diameter_pixels = corrected_widths[real_idx]

        return diameter_pixels

    def calculate_depth(self, top_ignore=0.4, bottom_ignore=0.20, min_num_points=300,
                        depth_filter_percentile_upper=65, depth_filter_percentile_lower=35):
        """
        Calculates the best estimate of the distance between the tree and camera.

        Args:
            top_ignore (): Proportion of the top of the image to ignore mask points in.
            bottom_ignore (): Proportion of the bottom of the image to ignore mask points in.
            min_num_points (): Minimum number of valid pointcloud points needed to keep the mask, if less than this,
            disregard the mask.
            depth_filter_percentile_upper (): Percentile of the depth values to use for the upper percentile depth
            estimate. So at 65, the percentile depth will be farther than 65% of the points in the mask.
            depth_filter_percentile_lower (): Percentile of the depth values to use for the lower percentile depth
            estimate. So at 35, the percentile depth will be closer than 35% of the points in the mask.

        Returns:
            Calculates the median depth and percentile depth for each mask. Also filters out masks that have less than
            min_num_points valid points in the region defined by top_ignore and bottom_ignore.
        """

        # Initialize arrays to store the depth values and the tree locations
        self.depth_median = np.zeros(self.num_instances)
        self.depth_percentile_upper = np.zeros(self.num_instances)
        self.depth_percentile_lower = np.zeros(self.num_instances)

        # Make boolean array of indices to keep
        keep = np.ones(self.num_instances, dtype=bool)

        # Calculate the top and bottom ignore values in pixels
        top_ignore = int(top_ignore * self.height)
        bottom_ignore = self.height - int(bottom_ignore * self.height)

        # Loop through each mask
        for i, mask in enumerate(self.masks):

            # Make copy of mask array
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

            # Make a 1D array of the masked portions of the depth image, which is currently a 2d array
            masked_depth = self.depth_img[mask_copy]

            # Remove zero values from the masked depth array
            masked_depth = masked_depth[masked_depth != 0]

            # If there are less than the min number of points, remove the mask
            if masked_depth.shape[0] < min_num_points:
                keep[i] = False
                continue

            # Calculate median depth
            self.depth_median[i] = np.median(masked_depth) / 1000
            # Calculate the percentile depth
            self.depth_percentile_upper[i] = np.percentile(masked_depth, depth_filter_percentile_upper) / 1000
            self.depth_percentile_lower[i] = np.percentile(masked_depth, depth_filter_percentile_lower) / 1000

        # Update the arrays
        self.depth_calculated = True
        self.update_arrays(keep)

    def calculate_width(self):
        """
        Calculates the best estimate of the width of the tree in meters.
        """

        self.tree_widths = np.zeros(self.num_instances)
        self.tree_locations = np.zeros((self.num_instances, 2))

        # Loop through each mask
        for i, (mask, depth) in enumerate(zip(self.masks, self.depth_median)):

            # Get the diameter of the tree in pixels
            diameter_pixels = self.get_width_pix(mask)

            # Check image width in pixels depending on the fov of the camera, based on the realsense d435
            if self.original_width == 640:
                horz_fov = 55.0
            elif self.original_width == 848 or self.original_width == 1280:
                horz_fov = 69.4
            else:
                print("Image width not supported, using default 69.4 degrees")
                horz_fov = 69.4

            # Calculate the width of the image in meters at the depth of the tree
            image_width_m = depth * np.tan(np.deg2rad(horz_fov / 2)) * 2

            # Calculate the distance per pixel
            distperpix = image_width_m / self.original_width

            # Calculate the diameter of the tree in meters
            diameter_m = diameter_pixels * distperpix

            # If there are no valid widths, set the width to 0, otherwise set it to the max width
            if len(diameter_m) == 0:
                self.tree_widths[i] = 0
            else:
                self.tree_widths[i] = np.max(diameter_m)

            # Calculate the x location of the tree in the image by taking the median of the mask points in x
            x_median_pixel = np.median(np.where(mask)[1])
            self.tree_locations[i, 1] = self.depth_median[i]
            self.tree_locations[i, 0] = (x_median_pixel - (self.original_width / 2)) * distperpix

        self.width_calculated = True


    def mask_filter_nms(self, overlap_threshold=0.5):
        """
        Apply non-maximum suppression (NMS) to a set of masks and scores. Yolo applies NMS, but only to individual
        classes, so this is a second NMS to remove any overlapping masks from different classes. There is technically an
        option to do this in Yolo, but it can't have a distinct iou threshold from the first NMS.

        Args:
            overlap_threshold (): Overlap threshold for NMS. If the overlap between two masks is greater than this
            value, the mask with the lower score will be suppressed.

        Returns:
            Updates the class arrays to only include the masks that were not suppressed.
        """

        mask_nms = self.masks.copy()
        score_nms = self.confidences.copy()

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
        self.update_arrays(not_suppressed)

    def mask_filter_depth(self, depth_threshold=1.5):
        """Sort out any outputs that are beyond a given depth threshold. Note that during depth calcuation any
        segments entirely in the top or bottom portions of the image are removed, and any segments with too few
        points in the point cloud are also removed.

        Args:
            depth_threshold (): Depth threshold in meters. Any masks with a percentile depth greater than this value
            will be removed.

        Returns:
            Updates the class arrays to only include the masks that are within the depth threshold.
        """
        keep = self.depth_percentile_upper < depth_threshold
        self.update_arrays(keep)

    def mask_filter_edge(self, edge_threshold=0.05, size_threshold=0.1):
        """Sort out any outputs with masks that are too close to the edge of the image. Edge threshold is how close
        the mask can be to the edge, as a proportion of the image width. Size threshold is the proportion of the mask
        that must be beyond the edge threshold for the mask to be removed. """

        keep = np.zeros(self.num_instances, dtype=bool)

        edge_threshold = int(edge_threshold * self.cropped_width)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            left_edge_pixels = mask[:, :edge_threshold].sum()
            right_edge_pixels = mask[:, -edge_threshold:].sum()
            total_mask_pixels = np.sum(mask)
            if left_edge_pixels / total_mask_pixels > size_threshold or right_edge_pixels / total_mask_pixels > size_threshold:
                continue
            else:
                keep[i] = True

        self.update_arrays(keep)

    def mask_filter_position(self, bottom_position_threshold=0.33, score_threshold=0.9, top_position_threshold=0.3):
        """Filter out any masks whose lowest point is above the bottom position threshold. Position threshold is the
        proportion of the image height from the bottom. Also filter out any masks that are entirely below the top
        position threshold."""

        keep = np.zeros(self.num_instances, dtype=bool)

        bottom_position_threshold = int(bottom_position_threshold * self.height)
        top_position_threshold = int(top_position_threshold * self.height)

        masks_copy = self.masks.copy()

        for i, mask in enumerate(masks_copy):
            # if self.confidences[i] > score_threshold:
            #     keep[i] = True
            #     continue

            bottom_pixels = mask[-bottom_position_threshold:].sum()
            if bottom_pixels > 0:
                keep[i] = True

            top_pixels = mask[:top_position_threshold].sum()
            if np.isclose(top_pixels, 0):
                keep[i] = False

        self.update_arrays(keep)

    def mask_filter_multi_segs(self):
        """Determine if there are multiple segments in the mask and if so keep only the largest one."""

        # Initialize an empty array to store the largest segments
        largest_segments = np.zeros_like(self.masks)

        # Loop through each mask
        for i, mask in enumerate(self.masks):
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

        self.masks = largest_segments

    def new_image_reset(self, results, depth_image):
        """ Resets the class variables for a new image.

        Args:
            results (): Results from the image segmentation model
            depth_image (): aligned depth image that corresponds to the image
        """

        self.confidences = None
        self.masks = None
        self.results_kept = None

        self.depth_calculated = False
        self.depth_median = None
        self.depth_percentile_upper = None
        self.tree_locations = None

        self.width_calculated = False
        self.tree_widths = None

        self.results = results

        self.setup_arrays()

        self.original_width = depth_image.shape[1]

        # Crop the depth image to be divisible by 32
        if depth_image.shape[0] % 32 != 0:
            depth_image = depth_image[:-(depth_image.shape[0] % 32), :]
        if depth_image.shape[1] % 32 != 0:
            depth_image = depth_image[:, :-(depth_image.shape[1] % 32)]

        self.height = depth_image.shape[0]
        self.cropped_width = depth_image.shape[1]

        self.depth_img = depth_image

        self.num_pixels = self.cropped_width * self.height

    def process_image(self):

        if self.masks is None:
            return

        # Send the masks through all the filters, skipping to the end if the number of instances is 0
        if self.num_instances > 0:
             self.mask_filter_multi_segs()
        if self.num_instances > 1:
            self.mask_filter_nms(overlap_threshold=0.3)
        if self.num_instances > 0:
            # self.calculate_depth(top_ignore=0.50, bottom_ignore=0.20, min_num_points=500,
            #                      depth_filter_percentile_upper=65, depth_filter_percentile_lower=35)
            self.calculate_depth(top_ignore=0.50, bottom_ignore=0.20, min_num_points=500,
                                 depth_filter_percentile_upper=70, depth_filter_percentile_lower=35)
        if self.num_instances > 0:
            self.mask_filter_depth(depth_threshold=2.0)
        if self.num_instances > 0:
            self.mask_filter_edge(edge_threshold=0.05)
        if self.num_instances > 0:
            self.mask_filter_position(bottom_position_threshold=0.5, top_position_threshold=0.65, score_threshold=0.9)
            self.calculate_width()

        # Calculate the x position of the tree in the image
        self.img_x_positions = np.zeros(self.num_instances, dtype=np.int32)
        for i, mask in enumerate(self.masks):
            # Figure out the x location of the mask
            self.img_x_positions[i] = int(np.mean(np.where(mask)[1]))


    def pf_helper(self, depth_image, results_dict=None, rgb_image=None):
        """Helper function to run the particle filter on a new image."""

        if self.combine_segmenter:
            if rgb_image is None:
                raise ValueError("rgb_image must be provided if combine_segmenter is True")
            results_dict, results = self.trunk_segmenter.get_results(rgb_image)

        # Reset the class variables for a new image
        self.new_image_reset(results_dict, depth_image)

        # Process the image
        self.process_image()

        # If no instances are left, return None for all the values
        if self.num_instances == 0 or self.num_instances is None:
            return None, None, None, None, None

        # Switch sign on x_pos and y_pos
        self.tree_locations[:, 0] = -self.tree_locations[:, 0]
        self.tree_locations[:, 1] = -self.tree_locations[:, 1]

        # Adjust the width based on the x position of the tree in the image
        img_x_positions_rel_middle = abs(self.img_x_positions - 320)
        # Values obtained from calibrate_widths.py, sept = True
        # self.tree_widths = -0.00625 + (-2.088e-05 * img_x_positions_rel_middle) + (1.058 * self.tree_widths)
        # self.tree_widths = -2.088e-05 * img_x_positions_rel_middle + self.tree_widths
        self.tree_widths = self.tree_widths - 2.8e-05 * img_x_positions_rel_middle - 0.001

        if self.combine_segmenter:
            results = results[self.results_kept]
            return self.tree_locations, self.tree_widths, self.classes, self.img_x_positions, results.plot()
        else:
            return self.tree_locations, self.tree_widths, self.classes, self.img_x_positions, self.results_kept

    def map_making_helper(self, rgb_img, depth_img, show_seg=False):
        """Helper function to run the map making code on a new image."""

        # Run the model on the image
        results_dict, results = self.trunk_segmenter.get_results(rgb_img)

        # Reset the class variables for a new image
        self.new_image_reset(results_dict, depth_img)

        # Process the image
        self.process_image()

        results = results[self.results_kept]

        # If no instances are left, return None for all the values
        if self.num_instances == 0 or self.num_instances is None:
            return None, None, None, None, None

        # Adjust the width based on the x position of the tree in the image
        # img_x_positions_rel_middle = abs(self.img_x_positions - 320)
        # Values obtained from calibrate_widths.py, sept = True
        # self.tree_widths = -0.006246 + (-2.0884248893265422e-05 * img_x_positions_rel_middle) + (1.057907234666699 * self.tree_widths)

        if show_seg:
            return self.tree_locations, self.tree_widths, self.classes, self.img_x_positions, results.plot()
        else:
            return self.tree_locations, self.tree_widths, self.classes, self.img_x_positions


