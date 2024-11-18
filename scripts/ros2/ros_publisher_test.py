import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pf_orchard_interfaces.msg import TreeImageData
from cv_bridge import CvBridge
import cv2
import numpy as np

class TrunkWidthSubTest(Node):
    """Subscribe to the data from the trunk width data publisher and display the data."""
    def __init__(self):
        super().__init__('trunk_width_sub_test')
        self.cv_bridge = CvBridge()
        self.trunk_info_subscription = self.create_subscription(TreeImageData, 'tree_image_data', self.listener_callback, 10)
        self.get_logger().info('Subscribed to the trunk width data publisher')

    def listener_callback(self, tree_image_data: TreeImageData):
        
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
        
        # Display the segmented image
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    trunk_width_sub_test = TrunkWidthSubTest()
    rclpy.spin(trunk_width_sub_test)
    trunk_width_sub_test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()