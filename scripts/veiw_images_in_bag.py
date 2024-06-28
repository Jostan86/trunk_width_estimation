from pf_orchard_localization.recorded_data_loaders import BagDataLoader
import cv2
from trunk_width_estimation import TrunkSegmenter, PackagePaths

bag_file = '/media/jostan/portabits/Smart Orchard Mattawa-selected/legs.bag'
video_save_path = bag_file.replace('.bag', '.mp4')
depth_topic = '/camera/depth/image_rect_raw'
rgb_topic = '/camera/color/image_rect_color'
odom_topic = '/odom'

trunk_segmenter = TrunkSegmenter(PackagePaths())

bag_data = BagDataLoader(bag_file, depth_topic, rgb_topic, odom_topic)

img_msg = bag_data.get_next_img_msg()

seg_images = []
while img_msg is not None:
    rgb_img = img_msg['rgb_image']
    
    # tree_locations, tree_widths, classes, img_x_positions, seg_image = trunk_analyzer.pf_helpber(rgb_img, depth_img)
    results_dict, results = trunk_segmenter.get_results(rgb_img)
    
    seg_images.append(results.plot())
    
    cv2.imshow('image', seg_images[-1])
    
    # if seg_image is not None:
    #     cv2.imshow('segmentation', seg_image)
       
    cv2.waitKey(10)
    
    img_msg = bag_data.get_next_img_msg()

bag_data.close()

# cv2.destroyAllWindows()

# make the images into a video
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(video_save_path, fourcc, 15.0, (seg_images[0].shape[1], seg_images[0].shape[0]))

# Write each frame to the video
for frame in seg_images:
    output_video.write(frame)

# Release the video writer
output_video.release()


