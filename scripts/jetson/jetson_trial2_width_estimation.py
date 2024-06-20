import cv2
import os
import time
from trunk_width_estimation import PackagePaths, TrunkAnalyzer

package_paths = PackagePaths()

times = []

trunk_segmenter = TrunkAnalyzer(package_paths, combine_segmenter=True)

for rgb_image_path, depth_image_path in zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths):
    
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    
    start_time = time.time()
    
    locations, widths, classes, img_x_positions, seg_img = trunk_segmenter.pf_helper(depth_image, rgb_image=rgb_image)
    
    # print("Time taken:", time.time() - start_time)
    
    if seg_img is not None:
        times.append(time.time() - start_time)
        cv2.imshow('output', seg_img)
        cv2.waitKey(25)
    
cv2.destroyAllWindows()

avg = sum(times) / len(times)
print('Average:', avg)

