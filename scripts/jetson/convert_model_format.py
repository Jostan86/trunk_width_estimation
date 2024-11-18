from ultralytics import YOLO
import os
from trunk_width_estimation import PackagePaths

package_paths = PackagePaths()

# Load a YOLOv8n PyTorch model
model = YOLO(os.path.join(package_paths.model_dir, "jazz_s_v8.pt"))


# Export the model

# Float16 model
model.export(format="engine", imgsz=(480, 640), half=True, workspace=8.0)  

# Float32 model
# model.export(format="engine", imgsz=(480, 640), workspace=8.0)  

# Int8 model (i deleted the dataset from here, so that would need to be added back for this to work) (but it sucks so don't bother)
# model.export(format="engine", imgsz=(480, 640), int8=True, workspace=6, data="/trunk_width_estimation/jazz_dataset_yolov8/data.yaml")
# model.export(format="engine", imgsz=(480, 640), int8=True)

# Load the exported TensorRT model
# trt_model = YOLO("jazz_s_v8.engine")

# Run inference
# results = trt_model("https://ultralytics.com/images/bus.jpg")
