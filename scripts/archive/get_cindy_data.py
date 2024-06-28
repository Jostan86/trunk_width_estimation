from trunk_width_estimation import TrunkAnalyzer
import cv2
import os
from tqdm import tqdm
import numpy as np

config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'width_estimation_config.yaml')

trunk_analyzer = TrunkAnalyzer(config_file_path, combine_segmenter=True)

rgb_img_dir = "/media/imml/portabits/map_making_data/cherry_orchard/images/row_6/rgb/"

rgb_img_paths = [os.path.join(rgb_img_dir, f) for f in os.listdir(rgb_img_dir)]
rgb_img_paths.sort()

# rgb_img_paths = rgb_img_paths[600:]
# keep only every 5th image

rgb_img_paths = rgb_img_paths[::10]
save_dir = "/media/imml/portabits/cindy_data/"

for rgb_img_path in tqdm(rgb_img_paths):

    depth_path = rgb_img_path.replace("rgb", "depth")
    rgb_image = cv2.imread(rgb_img_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    masks, seg_img, classes = trunk_analyzer.get_mask_and_seg(rgb_image, depth_image)

    if seg_img is None:
        print("No trunk detected")
        continue

    
    time_stamp_str = os.path.basename(rgb_img_path).split(".")[0]

    os.makedirs(os.path.join(save_dir, "image_data", time_stamp_str), exist_ok=True)

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
        
        
        # apply mask to rgb image as red shading on mask
        seg_image_custom = cv2.addWeighted(seg_image_custom, 1, seg_mask, 0.5, 0)

        
        mask_path = os.path.join(save_dir, "image_data", time_stamp_str, "mask" + "_" + str(idx) + "_" + class_name + ".png")

        cv2.imwrite(mask_path, mask)

    rgb_path = os.path.join(save_dir, "image_data", time_stamp_str, "rgb_img.png")
    depth_path = os.path.join(save_dir, "image_data", time_stamp_str, "depth_img.png")
    segmenation_path = os.path.join(save_dir, "segmentations", time_stamp_str + ".png")
    cv2.imwrite(rgb_path, rgb_image)
    cv2.imwrite(depth_path, depth_image)
    cv2.imwrite(segmenation_path, seg_image_custom)


