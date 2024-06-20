import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pf_orchard_interfaces.msg import TreeImageData, TreeInfo, TreePosition
from pf_orchard_interfaces.srv import TreeImageProcessing

from trunk_width_estimation import TrunkAnalyzer, PackagePaths
#!/usr/bin/env python

class TrunkWidthEstimationService(Node):
    def __init__(self):
        super().__init__('trunk_width_estimation_service')
        self.trunk_analyzer = TrunkAnalyzer(PackagePaths(), combine_segmenter=True)
        self.bridge = CvBridge()
        self.width_estimation_service = self.create_service(TreeImageProcessing, 'trunk_width_estimation', self.width_estimation_service_callback)
        
    def width_estimation_service_callback(self, req, res):
        rgb_image = self.bridge.imgmsg_to_cv2(req.color_image, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(req.depth_image, desired_encoding='passthrough')
        
        if rgb_image is None or depth_image is None:
            print("HELLO!")
        
        locations, widths, classes, img_x_positions, seg_img = self.trunk_analyzer.pf_helper(depth_image, rgb_image=rgb_image)
                
        tree_image_data = TreeImageData()
        
        if locations is None:
            tree_image_data.header = req.color_image.header
            tree_image_data.segmented_image = req.color_image
            res.tree_image_data = tree_image_data
            res.success = False
            res.message = 'No trees detected'
            return res
            
            
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

    minimal_service = TrunkWidthEstimationService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()