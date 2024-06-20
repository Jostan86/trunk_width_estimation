from ultralytics import YOLO
import cv2
import os
import time
from trunk_width_estimation import PackagePaths

package_paths = PackagePaths()

model_path = package_paths.model_dir + "/jazz_s_float16_v8.engine"
# model_path = package_paths.model_dir + "/jazz_s_v8.pt"
yolo_model = YOLO(model_path, task="segment")
# yolo_model = YOLO("jazz_s_v8.pt")

image = cv2.imread(package_paths.startup_image_path)

if image.shape[0] % 32 != 0:
    image = image[:-(image.shape[0] % 32), :, :]
if image.shape[1] % 32 != 0:
    image = image[:, :-(image.shape[1] % 32), :]

# results = yolo_model.predict(image, imgsz=(480, 640), int8=True, task="segment")
results = yolo_model.predict(image, imgsz=(480, 640))

results = results[0].cpu()

times_gpu = []
times_total = []

        
for image_path in package_paths.rgb_test_image_paths:
    
    image = cv2.imread(image_path)
    
    start_time = time.time()
    
    if image is None:
        print('Image not found:', image_path)
        continue
    
    results = yolo_model.predict(image, imgsz=(image.shape[0], image.shape[1]), iou=0.01, conf=0.7,
                                          verbose=False, agnostic_nms=False)
    # results = yolo_model.predict(image, imgsz=(480, 640), int8=True, task="segment")
    
    times_gpu.append(time.time() - start_time)
    
    results = results[0].cpu()
    
    times_total.append(time.time() - start_time)
    
    print(os.path.basename(image_path))
    
    cv2.imshow('image', results.plot())
    
    # cv2.imwrite('output.png', results.plot())

    cv2.waitKey(10)
    
cv2.destroyAllWindows()

gpu_avg = sum(times_gpu) / len(times_gpu)
print('GPU Average:', gpu_avg)

total_avg = sum(times_total) / len(times_total)
print('Total Average:', total_avg)
