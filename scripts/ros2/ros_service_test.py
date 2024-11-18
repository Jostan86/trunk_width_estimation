import rclpy
from rclpy.node import Node
from pf_orchard_interfaces.msg import TreeImageData
from pf_orchard_interfaces.srv import TreeImageProcessing
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np

class TrunkDataTrial(Node):
    def __init__(self):
        super().__init__('trunk_data_trial')

        # Initialize Subscribers directly
        self.rgb_subscription = Subscriber(self, Image, "/camera/camera/color/image_raw")
        self.depth_subscription = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")

        # Initialize time synchronizer with a slightly larger slop
        self.time_sync = ApproximateTimeSynchronizer([self.rgb_subscription, self.depth_subscription], 10, 0.01)
        self.time_sync.registerCallback(self.sync_callback)

        self.cv_bridge = CvBridge()

        self.trunk_analyzer_client = self.create_client(TreeImageProcessing, 'trunk_width_estimation')

        
        while not self.trunk_analyzer_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def sync_callback(self, rgb_msg, depth_msg):
        # self.get_logger().info("Received RGB and Depth images")
        tree_data_request = TreeImageProcessing.Request()
        tree_data_request.rgb_msg = rgb_msg
        tree_data_request.depth_msg = depth_msg

        self.start_time = self.get_clock().now()
        future = self.trunk_analyzer_client.call_async(tree_data_request)
        future.add_done_callback(self.service_response_callback)
    
    def service_response_callback(self, future):
        try:
            response: TreeImageProcessing.Response = future.result()
            
            tree_image_data: TreeImageData = response.tree_image_data
            segmented_image = self.cv_bridge.imgmsg_to_cv2(tree_image_data.segmented_image, desired_encoding="passthrough")

            if tree_image_data.object_seen:
                widths = np.array(tree_image_data.widths)
                object_locations = np.array([tree_image_data.xs, tree_image_data.ys]).T
                classifications = np.array(tree_image_data.classifications)
                self.get_logger().info(f"Widths: {widths}")
                self.get_logger().info(f"Object Locations: {object_locations}")
                self.get_logger().info(f"Classifications: {classifications}")
            else:
                self.get_logger().info("No object detected")


            total_time = self.get_clock().now() - self.start_time
            total_time_sec = total_time.nanoseconds / 1e9
            self.get_logger().info(f"Time taken: {total_time_sec} seconds")
            
            # Display the segmented image
            cv2.imshow("Segmented Image", segmented_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().info(f"Service call failed: {e}")

        

def main(args=None):
    rclpy.init(args=args)
    trunk_data_trial = TrunkDataTrial()
    rclpy.spin(trunk_data_trial)
    trunk_data_trial.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
