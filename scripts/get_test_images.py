import rosbag
import cv2
import os
from cv_bridge import CvBridge
from trunk_width_estimation import PackagePaths

package_paths = PackagePaths()

bag_path = "/media/jostan/MOAD/research_data/apple_orchard_data/2023_orchard_data/uncompressed/synced/pcl_mod/envy-trunks-02_5_converted_synced_pcl-mod.bag"
rgb_topic = "/registered/rgb/image"
depth_topic = "/registered/depth/image"

rgb_save_dir = package_paths.rgb_test_images_dir
depth_save_dir = package_paths.depth_test_images_dir

if not os.path.exists(rgb_save_dir):
    os.makedirs(rgb_save_dir)
    
if not os.path.exists(depth_save_dir):
    os.makedirs(depth_save_dir)

bridge = CvBridge()

bag = rosbag.Bag(bag_path)

for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic]):
    
    if topic == depth_topic:
        time_stamp = msg.header.stamp
        time_stamp_str = str(time_stamp.secs) + "_" + str(time_stamp.nsecs).zfill(9)
        depth_image = bridge.imgmsg_to_cv2(msg, "passthrough")
        depth_path = os.path.join(depth_save_dir, time_stamp_str + ".png")
        cv2.imwrite(depth_path, depth_image)
    elif topic == rgb_topic:
        time_stamp = msg.header.stamp
        time_stamp_str = str(time_stamp.secs) + "_" + str(time_stamp.nsecs).zfill(9)
        rgb_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_path = os.path.join(rgb_save_dir, time_stamp_str + ".png")
        cv2.imwrite(rgb_path, rgb_image)
    
    

bag.close()

