import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from collections import deque
from cv_bridge import CvBridge
from pf_orchard_interfaces.msg import TreeImageData
from trunk_width_estimation import TrunkAnalyzer, PackagePaths, TrunkAnalyzerData
import time
import os
from message_filters import ApproximateTimeSynchronizer, Subscriber

class TrunkWidthEstimationNode(Node):
    """
    This class subscribes to the depth and rgb image topics and publishes the data from the trunk width estimation.
    """

    def __init__(self):
        super().__init__('trunk_width_estimation')
        
        self.depth_queue = deque()
        self.rgb_queue = deque()

        self.cv_bridge = CvBridge()
        
        self.depth_sub = Subscriber(self, Image, os.environ['DEPTH_IMAGE_TOPIC'])
        self.rgb_sub = Subscriber(self, Image, os.environ['RGB_IMAGE_TOPIC'])
        
        self.time_sync = ApproximateTimeSynchronizer([self.depth_sub, self.rgb_sub], queue_size=10, slop=0.01)
        self.time_sync.registerCallback(self.sync_callback)
        
        self.trunk_analyzer = TrunkAnalyzer(PackagePaths('width_estimation_config_apple.yaml'), combine_segmenter=True)
        
        self.publisher = self.create_publisher(TreeImageData, 'tree_image_data', 10)
        
        # self.img_pub = self.create_publisher(Image, 'segmented_image', 10)
        
    def sync_callback(self, depth_msg: Image, rgb_msg: Image):
        self.do_work(depth_msg, rgb_msg)

    def do_work(self, depth_msg: Image, rgb_msg: Image):

        # Simulate a delay
        # time.sleep(0.2)
              
        start_time = time.time()
        trunk_analyzer_data = TrunkAnalyzerData.from_ros2_image_msgs(rgb_msg, depth_msg)
        trunk_analyzer_data = self.trunk_analyzer.get_width_estimation_pf(trunk_analyzer_data)
        tree_image_data = trunk_analyzer_data.create_ros2_tree_info_msg()

        self.get_logger().info(f"Time taken: {time.time() - start_time}")
              
        self.publisher.publish(tree_image_data)
        # self.img_pub.publish(tree_image_data.segmented_image)
        

def main(args=None):
    rclpy.init(args=args)
    realsense_processor = TrunkWidthEstimationNode()
    rclpy.spin(realsense_processor)
    realsense_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # os.environ['DEPTH_IMAGE_TOPIC'] = '/camera/camera/aligned_depth_to_color/image_raw'
    # os.environ['RGB_IMAGE_TOPIC'] = '/camera/camera/color/image_raw'
    main()
