import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pf_orchard_interfaces.msg import TreeImageData, TreeInfo, TreePosition
from pf_orchard_interfaces.srv import TreeImageProcessing
import time
from trunk_width_estimation import TrunkAnalyzer, PackagePaths
from std_msgs.msg import Bool
#!/usr/bin/env python

class TrunkWidthEstimationService(Node):
    def __init__(self):
        super().__init__('trunk_width_estimation_service')
        self.trunk_analyzer = TrunkAnalyzer(PackagePaths('width_estimation_config_apple.yaml'), combine_segmenter=True)
        self.bridge = CvBridge()
        self.width_estimation_service = self.create_service(TreeImageProcessing, 'trunk_width_estimation', self.width_estimation_service_callback)

        self.get_logger().info('Trunk width estimation service has been started')
        
    def width_estimation_service_callback(self, req, res):
        
        rgb_image = self.bridge.imgmsg_to_cv2(req.color_image, desired_encoding='passthrough')
        depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
        
        start_time = time.time()
        locations, widths, classes, img_x_positions, seg_img = self.trunk_analyzer.get_width_estimation_pf(depth_image, rgb_image=rgb_image)
        self.get_logger().info("Time taken: {}".format(time.time() - start_time))
        tree_image_data = TreeImageData()
        
        if locations is None:
            tree_image_data.header = req.color_image.header
            tree_image_data.segmented_image = req.color_image
            res.tree_image_data = tree_image_data
            res.success = False
            res.message = 'No trees detected'
            res.tree_image_data.object_seen = False
            return res
            
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
        
        tree_image_data.segmented_image = self.bridge.cv2_to_imgmsg(seg_img, encoding='passthrough')
        tree_image_data.header = req.color_image.header
         
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