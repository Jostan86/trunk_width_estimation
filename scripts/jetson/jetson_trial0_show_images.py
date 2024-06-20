import cv2
import os
from trunk_width_estimation import PackagePaths

package_paths = PackagePaths()


times_gpu = []
times_total = []

        
for image_path in package_paths.rgb_test_image_paths:
    image = cv2.imread(image_path)
    
    print(os.path.basename(image_path))
    
    cv2.imshow('image', image)
    
    cv2.waitKey(50)
    # cv2.imwrite('output.png', image)

cv2.destroyAllWindows()