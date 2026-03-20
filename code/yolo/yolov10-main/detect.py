from ultralytics import YOLOv10
import os

# model = YOLOv10("/media/disk1/wl/yolov10-main/runs/train/exp_tactile/weights/best.pt")
model = YOLOv10("/media/disk1/wl/yolov10-main/runs/train/exp_true/weights/best.pt")
# input_dir = r"/media/disk1/wl/yolov10-main/datasets/tactile/test/images"
input_dir = r"/media/disk1/wl/yolov10-main/datasets/true/test/images"
# output_dir = r"/media/disk1/wl/yolov10-main/runs/detect/tactile"
output_dir = r"/media/disk1/wl/yolov10-main/runs/detect/true"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)

        results = model.predict(img_path)

        save_path = os.path.join(output_dir, f"{filename}")
        results[0].save(save_path)