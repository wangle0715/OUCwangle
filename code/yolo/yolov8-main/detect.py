from ultralytics import YOLO
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["ULTRALYTICS_HUB_DISABLE"] = "1"
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = YOLO("/media/disk1/wl/yolov8-main/runs/train/best1.pt")
# model = YOLO("/media/disk1/wl/yolov8-main/runs/train/best2.pt")
input_dir = r"/media/disk1/wl/yolov8-main/datasets/tactile/test/images"
# input_dir = r"/media/disk1/wl/yolov8-main/datasets/true/test/images"
output_dir = r"/media/disk1/wl/yolov8-main/runs/detect/tactile"
# output_dir = r"/media/disk1/wl/yolov8-main/runs/detect/true"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)

        results = model.predict(img_path)

        save_path = os.path.join(output_dir, f"{filename}")
        results[0].save(save_path)