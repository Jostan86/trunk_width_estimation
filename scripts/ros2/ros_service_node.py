import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pf_orchard_interfaces.msg import TreeImageData
from pf_orchard_interfaces.srv import TreeImageProcessing
from trunk_width_estimation import TrunkAnalyzer, PackagePaths, TrunkAnalyzerData

class TrunkWidthEstimationService(Node):
    """
    This class provides a the width estimation as a ros service.
    """
    def __init__(self):
        super().__init__('trunk_width_estimation_service')
        self.trunk_analyzer = TrunkAnalyzer(PackagePaths('width_estimation_config_apple.yaml'), combine_segmenter=True)
        self.bridge = CvBridge()
        self.width_estimation_service = self.create_service(TreeImageProcessing, 'trunk_width_estimation', self.width_estimation_service_callback)

        self.get_logger().info('Trunk width estimation service has been started')
        
    def width_estimation_service_callback(self, req: TreeImageProcessing.Request, res: TreeImageProcessing.Response):
        
        trunk_analyzer_data = TrunkAnalyzerData.from_ros2_image_msgs(req.rgb_msg, req.depth_msg)
        trunk_analyzer_data = self.trunk_analyzer.get_width_estimation_pf(trunk_analyzer_data)
        tree_image_data = trunk_analyzer_data.create_ros2_tree_info_msg()

        res.tree_image_data = tree_image_data
        res.success = True
        
        return res

def main(args=None):
    rclpy.init(args=args)

    trunk_width_service = TrunkWidthEstimationService()

    rclpy.spin(trunk_width_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()