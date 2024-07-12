import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from collections import deque
import message_filters
# from .optical_flow_odom import OpticalFlowOdometer
from cv_bridge import CvBridge
from pf_orchard_interfaces.msg import TreeImageData, TreeInfo, TreePosition
from trunk_width_estimation import TrunkAnalyzer, PackagePaths
import time

class TrunkWidthEstimationNode(Node):

    def __init__(self):
        super().__init__('trunk_width_estimation')
        
        self.depth_queue = deque()
        self.rgb_queue = deque()
        
        self.depth_sub = message_filters.Subscriber(self, Image, '/registered/depth/image')
        self.rgb_sub = message_filters.Subscriber(self, Image, '/registered/rgb/image')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.rgb_sub], 
            queue_size=20, 
            slop=0.01
        )
        self.ts.registerCallback(self.callback)
        
        self.trunk_analyzer = TrunkAnalyzer(PackagePaths('width_estimation_config_apple.yaml'), combine_segmenter=True)
        self.cv_bridge = CvBridge()
        
        self.publisher = self.create_publisher(TreeImageData, 'tree_image_data', 10)
        self.img_pub = self.create_publisher(Image, 'segmented_image', 10)
        self.timer = self.create_timer(0.001, self.process_images)
        
        self.processing_images = False

    def callback(self, depth_msg: Image, rgb_msg: Image):
        self.depth_queue.append(depth_msg)
        self.rgb_queue.append(rgb_msg)
        # self.get_logger().info('Images received and added to queue')


    def process_images(self):
        if self.depth_queue and self.rgb_queue:
            if self.processing_images:
                self.get_logger().warn('Images are still being processed. Skipping this iteration. Also, wtf 14636')
                return
            
            self.processing_images = True
            
            depth_msg = self.depth_queue.popleft()
            rgb_msg = self.rgb_queue.popleft()
            
            if len(self.depth_queue) > 20:
                self.get_logger().warn('Queue too large. Reducing queue size.')
                self.depth_queue.popleft()
                self.rgb_queue.popleft()

            if depth_msg.header.stamp == rgb_msg.header.stamp:
                # self.get_logger().info(f'Processing images with timestamp: {depth_msg.header.stamp}')
                self.do_work(depth_msg, rgb_msg)
            else:
                self.get_logger().warn('Timestamps do not match. WTF Why 2356432')
                # self.find_matching_images(depth_msg, rgb_msg)

            self.processing_images = False
    # def find_matching_images(self, depth_msg, rgb_msg):
    #     if len(self.depth_queue) > 1 and len(self.rgb_queue) > 1:
    #         next_depth_msg = self.depth_queue[0]
    #         next_rgb_msg = self.rgb_queue[0]

    #         if next_depth_msg.header.stamp == next_rgb_msg.header.stamp:
    #             self.depth_queue.popleft()
    #             self.rgb_queue.popleft()
    #             self.do_work(next_depth_msg, next_rgb_msg)
    #         else:
    #             self.get_logger().warn('Could not find matching timestamps. Dropping images.')
    #             self.depth_queue.popleft()
    #             self.rgb_queue.popleft()

    def do_work(self, depth_msg: Image, rgb_msg: Image):
        
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        
        if rgb_image is None or depth_image is None:
            print("HELLO!")
        
        start_time = time.time()
        locations, widths, classes, img_x_positions, seg_img = self.trunk_analyzer.get_width_estimation_pf(depth_image, rgb_image=rgb_image)
        self.get_logger().info("Time taken: {}".format(time.time() - start_time))
        
        tree_image_data = TreeImageData()
        
        if locations is None:
            tree_image_data.header = rgb_msg.header
            tree_image_data.segmented_image = rgb_msg
            tree_image_data = tree_image_data
            tree_image_data.object_seen = False
            self.publisher.publish(tree_image_data)
            return 
            
        tree_image_data.object_seen = True
        for i in range(len(locations)):
            tree_position = TreePosition()
            tree_position.x = locations[i][0]
            tree_position.y = locations[i][1]
            
            tree_info = TreeInfo()
            tree_info.position = tree_position
            tree_info.width = widths[i]
            tree_info.classification = int(classes[i])
            
            tree_image_data.trees.append(tree_info)
        
        tree_image_data.segmented_image = self.cv_bridge.cv2_to_imgmsg(seg_img, encoding='passthrough')
        tree_image_data.header = rgb_msg.header
              
        self.publisher.publish(tree_image_data)
        self.img_pub.publish(tree_image_data.segmented_image)
        

def main(args=None):
    rclpy.init(args=args)
    realsense_processor = TrunkWidthEstimationNode()
    rclpy.spin(realsense_processor)
    realsense_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
