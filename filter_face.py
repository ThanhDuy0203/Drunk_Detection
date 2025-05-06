from ultralytics import YOLO
import cv2
import os
import shutil


model = YOLO('yolo11n.pt')
source_folder = r"D:\FPTUniversity\Capstone_Project\Code_tesst\out3"
destination_folder = r"D:\FPTUniversity\Capstone_Project\Code_tesst\out3_test"

os.makedirs(destination_folder, exist_ok=True)
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        results = model(image)
        if results[0].boxes:
            for result in results:
                if result.names:
                    names = result.names
                    for c in result.boxes.cls:
                        if names[int(c)] == "person":

                            shutil.copy(image_path, os.path.join(destination_folder, filename))
                            print(f"Đã lưu ảnh: {filename}")
                            break
                    else:
                        continue
                    break
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {filename}")

print("Hoàn tất!")


