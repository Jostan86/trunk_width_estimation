import cv2
import os
import time
from trunk_width_estimation import TrunkAnalyzer, TrunkSegmenter, PackagePaths
import copy


def display_images(start_img=0):
    """This function will display the segmented images in the test images folder one at a time. """
    
    package_paths = get_package_paths()

    trunk_analyzer = TrunkAnalyzer(package_paths, combine_segmenter=True)

    for i, (rgb_image_path, depth_image_path) in enumerate(zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths)):
        
        if i < start_img:
            continue
        
        print(i)
        
        # Load images
        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        
        # Get results
        positions, widths, classes, img_x_positions, seg_img = trunk_analyzer.get_width_estimation_pf(depth_image, rgb_image=rgb_image)
        
        print("--------------------")
        print("Object Positions (m):", positions)
        print("Object Widths (m):", widths)
        print("Object Classes:", classes)
        print("Image X Positions (pixels):", img_x_positions)
        
        cv2.imshow("Depth Image", visualize_depth_image(depth_image))
        
        if seg_img is not None:
            cv2.imshow('Segmented and Filtered', seg_img)
        else:
            cv2.imshow('Segmented and Filtered', rgb_image)
            
        cv2_wait()
        
    cv2.destroyAllWindows()


def display_images_segmenter_separate(start_img=0):
    """This function will also display the segmented images in the test images folder one at a time, but the 
    segmenter and analyzer are separate. It also displays the segmented image before and after the analyzer."""

    
    package_paths = get_package_paths()

    trunk_analyzer = TrunkAnalyzer(package_paths, combine_segmenter=False)
    trunk_segmenter = TrunkSegmenter(package_paths)

    for i, (rgb_image_path, depth_image_path) in enumerate(zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths)):
        
        if i < start_img:
            continue
        
        print(i)
        
        # Load images
        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        
        # Get segmentation data and the unfiltered segmentation image
        results_dict, results = trunk_segmenter.get_results(rgb_image)   
        og_seg_img = results.plot()
        
        # Get results from the analyzer
        positions, widths, classes, img_x_positions, results_kept = trunk_analyzer.get_width_estimation_pf(depth_image, results_dict=results_dict)  
        if results_kept is not None:
            results = results[results_kept]
            seg_img = results.plot()
        else:
            seg_img = rgb_image
        
        print("--------------------")
        print("Object Positions (m):", positions)
        print("Object Widths (m):", widths)
        print("Object Classes:", classes)
        print("Image X Positions (pixels):", img_x_positions)
        
        cv2.imshow("Depth Image", visualize_depth_image(depth_image))
        cv2.imshow("Original Segmentation", og_seg_img)
        cv2.imshow('Segmented and Filtered', seg_img)
        
        cv2_wait()
        
    cv2.destroyAllWindows()

def get_package_paths():
    # Set the package path environment variable
    package_path = "/trunk_width_estimation"
    os.environ['WIDTH_ESTIMATION_PACKAGE_PATH'] = package_path

    # Load the package paths
    package_paths = PackagePaths(config_file="width_estimation_config_apple.yaml")
    
    return package_paths

def cv2_wait():
    # This is needed because cv2.imshow isn't working in docker right
    print('select the image and press any key to continue')
    while True:
        key = cv2.waitKey(25) & 0xFF
        if key != 255:  # Any key pressed
            break


def visualize_depth_image(depth_image):
    # Convert depth values from mm to meters
    depth_image = depth_image / 1000.0
    
    # Set values over 6 meters to 6 meters
    depth_image[depth_image > 6] = 6
    
    # Normalize depth values between 0 and 1
    depth_image = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply colormap to depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255), cv2.COLORMAP_JET)
    
    return depth_colormap
    
    
if __name__ == "__main__":
    
    # There are two ways to use the trunk analyzer, it can be combined with the segmenter or kept separate. The
    # ability to seperate was mainly added to allow for parallel processing of the analysis on the previous reults while
    # segmenting the next image. The combined version is slightly easier to use and is the default. 
    
    display_images(start_img=200)
    # display_images_segmenter_separate(start_img=200)