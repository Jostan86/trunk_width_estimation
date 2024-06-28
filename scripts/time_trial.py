import cv2
import os
import time
from trunk_width_estimation import TrunkAnalyzer, PackagePaths

package_path = "/trunk_width_estimation"
os.environ['WIDTH_ESTIMATION_PACKAGE_PATH'] = package_path

package_paths = PackagePaths(config_file="width_estimation_config_apple.yaml")

times_overall = []
times_with_seg = []

trunk_segmenter = TrunkAnalyzer(package_paths, combine_segmenter=True)

for rgb_image_path, depth_image_path in zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths):
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    
    start_time = time.time()
    
    locations, widths, classes, img_x_positions, seg_img = trunk_segmenter.get_width_estimation_pf(depth_image, rgb_image=rgb_image)
    
    times_overall.append(time.time() - start_time)
    
    if seg_img is not None:
        times_with_seg.append(time.time() - start_time)
        cv2.imshow('output.png', seg_img)
        cv2.waitKey(50)
    
cv2.destroyAllWindows()

avg_overall = sum(times_overall) / len(times_overall)
print('Average Overall:', avg_overall)

avg_with_seg = sum(times_with_seg) / len(times_with_seg)
print('Average of images with segmentations:', avg_with_seg)