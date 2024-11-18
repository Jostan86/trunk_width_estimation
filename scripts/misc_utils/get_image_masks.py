from trunk_width_estimation import TrunkSegmenter, PackagePaths
import cv2
import os
from tqdm import tqdm
import numpy as np

# This script is used to generate masks for individual trunks in the images. It creates a folder for each pair of depth/rgb images and saves the masks in the folder as:
# - timestamp
#     - seg_img.png
#     - rgb_img.png
#     - depth_img.png
#     - mask_0_post.png
#     - mask_1_trunk.png
#     - ...


# ------------------------------
# Set the paths here

# path to the config file
config_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'width_estimation_config_apple.yaml')

# path to the rgb images, a folder with the depth images should be in the same directory named "depth", with the same filenames for the corresponding images
rgb_img_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'test_images', 'rgb')
# rgb_img_dir = "/media/imml/portabits/map_making_data/cherry_orchard/images/row_6/rgb/"

# save location
save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mask_data')

# Save every nth image
n = 10
# ------------------------------


# Initialize the segmenter
trunk_segmenter = TrunkSegmenter(PackagePaths(config_file_path))

# Get the paths to the images
rgb_img_paths = [os.path.join(rgb_img_dir, f) for f in os.listdir(rgb_img_dir)]
rgb_img_paths.sort()

# Only use every nth image
rgb_img_paths = rgb_img_paths[::n]

# Create the save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for rgb_img_path in tqdm(rgb_img_paths):

    # Load the images
    depth_path = rgb_img_path.replace("rgb", "depth")
    rgb_image = cv2.imread(rgb_img_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Get the results
    results_dict, results = trunk_segmenter.get_results(rgb_image)
    masks = results_dict["masks"]
    seg_img = results.plot()
    classes = results_dict["classes"]

    # If no trunks are detected, skip the image
    if masks is None:
        print("No trunk detected")
        continue
    
    # Create a folder for the image
    time_stamp_str = os.path.basename(rgb_img_path).split(".")[0]
    os.makedirs(os.path.join(save_dir, time_stamp_str), exist_ok=True)

    # Save the masks
    seg_image_custom = rgb_image.copy()
    for idx, (mask, mask_class) in enumerate(zip(masks, classes)):
        mask = mask.astype(int)
        mask = mask * 255
        mask = mask.astype('uint8')
        if mask_class == 0:
            class_name = "post"
            seg_mask = cv2.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
        else:
            class_name = "trunk"
            seg_mask = cv2.merge([mask, np.zeros_like(mask), np.zeros_like(mask)])
        
        
        # apply mask to rgb image
        seg_image_custom = cv2.addWeighted(seg_image_custom, 1, seg_mask, 0.5, 0)

        
        mask_path = os.path.join(save_dir, time_stamp_str, "mask" + "_" + str(idx) + "_" + class_name + ".png")

        cv2.imwrite(mask_path, mask)

    rgb_path = os.path.join(save_dir, time_stamp_str, "rgb_img.png")
    depth_path = os.path.join(save_dir, time_stamp_str, "depth_img.png")
    segmenation_path = os.path.join(save_dir, time_stamp_str, "seg_img.png")
    cv2.imwrite(rgb_path, rgb_image)
    cv2.imwrite(depth_path, depth_image)
    cv2.imwrite(segmenation_path, seg_image_custom)


