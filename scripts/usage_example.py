import cv2
import os
import time
from trunk_width_estimation import TrunkAnalyzer, TrunkSegmenter, PackagePaths, TrunkAnalyzerData


def display_images(package_paths: PackagePaths, start_img=0):
    """This function will display the segmented images in the test images folder one at a time. """

    trunk_analyzer = TrunkAnalyzer(package_paths, combine_segmenter=True)

    for i, (rgb_image_path, depth_image_path) in enumerate(zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths)):
        
        if i < start_img:
            continue
        
        print(i)
        
        start_time = time.time()

        # Load images
        trunk_data = TrunkAnalyzerData.from_image_paths(rgb_image_path, depth_image_path)

        # Get results
        trunk_data = trunk_analyzer.get_width_estimation_pf(trunk_data)
        
        print("--------------------")
        print("Object Positions (m):", trunk_data.object_locations)
        print("Object Widths (m):", trunk_data.object_widths)
        print("Object Classes:", trunk_data.classes)
        print("Image X Positions (pixels):", trunk_data.x_positions_in_image)
        print("Time taken: ", time.time() - start_time)
        
        cv2.imshow("Depth Image", trunk_data.visualize_depth_image())
        cv2.imshow('Segmented and Filtered', trunk_data.visualize_segmentation())
        cv2_wait()
        #wait 50ms
        # cv2.waitKey(50)
        
    cv2.destroyAllWindows()

    for filter in trunk_analyzer.operations:
        filter.print_performance_data()

def display_images_segmenter_separate(package_paths: PackagePaths, start_img=0):
    """This function will also display the segmented images in the test images folder one at a time, but the 
    segmenter and analyzer are separate. It also displays the segmented image before and after the analyzer."""

    trunk_analyzer = TrunkAnalyzer(package_paths, combine_segmenter=False)
    trunk_segmenter = TrunkSegmenter(package_paths)

    for i, (rgb_image_path, depth_image_path) in enumerate(zip(package_paths.rgb_test_image_paths, package_paths.depth_test_image_paths)):
        
        if i < start_img:
            continue
        
        print(i)

        start_time = time.time()
        
        # Load images
        trunk_data = TrunkAnalyzerData.from_image_paths(rgb_image_path, depth_image_path)
        
        # Get segmentation data and the unfiltered segmentation image
        trunk_data: TrunkAnalyzerData = trunk_segmenter.run(trunk_data)  
        og_seg_img = trunk_data.visualize_segmentation()
        
        # Get results from the analyzer
        trunk_data: TrunkAnalyzerData = trunk_analyzer.get_width_estimation_pf(trunk_data)  
        
        
        print("--------------------")
        print("Object Positions (m):", trunk_data.object_locations)
        print("Object Widths (m):", trunk_data.object_widths)
        print("Object Classes:", trunk_data.classes)
        print("Image X Positions (pixels):", trunk_data.x_positions_in_image)
        print("Time taken: ", time.time() - start_time)
        
        cv2.imshow("Depth Image", trunk_data.visualize_depth_image())
        
        cv2.imshow('Segmented and Filtered', trunk_data.visualize_segmentation())
        
        cv2.imshow("Original Segmentation", og_seg_img)
        
        cv2_wait()
        
    cv2.destroyAllWindows()


def cv2_wait():
    # This is needed because cv2.imshow isn't working in docker right
    print('select the image and press any key to continue')
    while True:
        key = cv2.waitKey(25) & 0xFF
        if key != 255:  # Any key pressed
            break


if __name__ == "__main__":
    
    # ----- Set the package paths -----
    # These are environment variables that can be set to the paths of the package and package data. The PackagePaths object needs these to exist.
    
    # Package path is the path to the trunk_width_estimation folder.
    # package_path = "/path/to/trunk_width_estimation"
    # os.environ['WIDTH_ESTIMATION_PACKAGE_PATH'] = package_path

    # Package data path is the path to the trunk_width_estimation_package_data folder.
    # package_data_path = "/path/to/trunk_width_estimation_package_data"
    # os.environ['WIDTH_ESTIMATION_PACKAGE_DATA_PATH'] = package_data_path

    # Load the package paths
    package_paths = PackagePaths(config_file="width_estimation_config_apple.yaml")
    
    # ----- Display images -----
    # There are two ways to use the trunk analyzer, it can be combined with the segmenter or kept separate. The
    # ability to separate was mainly added to allow for parallel processing of the analysis on the previous results while
    # segmenting the next image. The combined version is slightly easier to use and is the default. 
    
    display_images(package_paths, start_img=1)
    # display_images_segmenter_separate(package_paths, start_img=1)
