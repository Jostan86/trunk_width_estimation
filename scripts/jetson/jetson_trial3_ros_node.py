from trunk_width_estimation import PackagePaths
from pf_orchard_interfaces.srv import TreeImageProcessing
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

class TrunkWidthEstimationClient(Node):
    def __init__(self):
        super().__init__('trunk_width_estimation_client')
        self.bridge = CvBridge()
        self.client = self.create_client(TreeImageProcessing, 'trunk_width_estimation')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        self.request = TreeImageProcessing.Request()
    
    def send_request(self, rgb_image, depth_image):
        color_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
        depth_image = self.bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')
        
        self.request.color_image = color_image
        self.request.depth_image = depth_image
        
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        
        return response
            
                                   
def main(args=None):
    rclpy.init(args=args)
    trunk_width_estimation_client = TrunkWidthEstimationClient()
    package_paths = PackagePaths()
    times = []
    for rgb_image_path, depth_image_path in zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths):
        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        
        time_start = time.time()
        response = trunk_width_estimation_client.send_request(rgb_image, depth_image)
        
        if response.success:
            times.append(time.time() - time_start)
        
        seg_img = trunk_width_estimation_client.bridge.imgmsg_to_cv2(response.tree_image_data.segmented_image, desired_encoding='passthrough')
        cv2.imshow('segmented image', seg_img)
        cv2.waitKey(25)
    
    print('Average time taken: ', np.mean(times))
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()
