import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
def initialize_face_mesh():
    #Khởi tạo FaceMesh cho ảnh tĩn
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True)


def get_connected_landmarks(face_oval):
    #Tạo danh sách các điểm kết nối liên tục từ FACEMESH_FACE_OVAL
    df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
    routes_idx = []

    # Lấy điểm bắt đầu
    p1, p2 = df.iloc[0]["p1"], df.iloc[0]["p2"]

    # Tìm các điểm nối liên tục
    for _ in range(df.shape[0]):
        routes_idx.append([p1, p2])
        obj = df[df["p1"] == p2]
        p1, p2 = obj["p1"].values[0], obj["p2"].values[0]

    return routes_idx


def get_face_contour(landmarks, routes_idx, img_shape):
    #Chuyển đổi chỉ số landmark thành tọa độ pixel
    routes = []
    for source_idx, target_idx in routes_idx:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        # Tính tọa độ tương đối
        relative_source = (int(img_shape[1] * source.x), int(img_shape[0] * source.y))
        relative_target = (int(img_shape[1] * target.x), int(img_shape[0] * target.y))

        routes.append(relative_source)
        routes.append(relative_target)

    return routes


def crop_face(image, routes):
    #Tạo mask và cắt khuôn mặt
    # Tạo mask rỗng
    mask = np.zeros(image.shape[:2])
    # Vẽ đa giác lồi bao quanh khuôn mặt
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    # Áp dụng mask lên ảnh gốc
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out

def crop_face_square(image, routes, target_size=480):
    #Cắt vùng khuôn mặt và resize về 480x480
    # Chuyển routes thành numpy array
    routes = np.array(routes, dtype=np.int32)

    # Lấy bounding box chứa toàn bộ khuôn mặt
    x_min, y_min = np.min(routes, axis=0)
    x_max, y_max = np.max(routes, axis=0)

    # Tính kích thước hiện tại của khuôn mặt
    width, height = x_max - x_min, y_max - y_min

    # Xác định tâm của khuôn mặt
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # Tạo bounding box hình vuông
    half_size = target_size // 2
    x1, x2 = center_x - half_size, center_x + half_size
    y1, y2 = center_y - half_size, center_y + half_size

    # Đảm bảo không bị out-of-bounds
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    # Cắt ảnh
    face_cropped = image[y1:y2, x1:x2]

    # Resize về target_size nếu cần
    if face_cropped.shape[0] != target_size or face_cropped.shape[1] != target_size:
        face_cropped = cv2.resize(face_cropped, (target_size, target_size))

    return face_cropped



def process_image(image_path):
    #Xử lý ảnh và hiển thị kết quả
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return

    # Khởi tạo FaceMesh và xử lý
    face_mesh = initialize_face_mesh()
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("Không phát hiện khuôn mặt")
        return

    # Lấy landmarks của khuôn mặt đầu tiên
    landmarks = results.multi_face_landmarks[0]
    face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL

    # Tạo đường viền và cắt ảnh
    routes_idx = get_connected_landmarks(face_oval)
    routes = get_face_contour(landmarks, routes_idx, img.shape)
    face_cropped = crop_face(img, routes)

    # Hiển thị bằng matplotlib
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(face_cropped[:, :, ::-1])  # Chuyển BGR sang RGB
    plt.show()
"""


def initialize_face_mesh():
    """Khởi tạo FaceMesh cho ảnh tĩnh"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def get_connected_landmarks(face_oval):
    """Tạo danh sách các điểm kết nối liên tục từ FACEMESH_FACE_OVAL"""
    df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])
    routes_idx = []

    # Lấy điểm bắt đầu
    p1, p2 = df.iloc[0]["p1"], df.iloc[0]["p2"]

    # Tìm các điểm nối liên tục
    for _ in range(df.shape[0]):
        routes_idx.append([p1, p2])
        obj = df[df["p1"] == p2]
        p1, p2 = obj["p1"].values[0], obj["p2"].values[0]

    return routes_idx


def get_face_contour(landmarks, routes_idx, img_shape):
    """Chuyển đổi chỉ số landmark thành tọa độ pixel"""
    routes = []
    for source_idx, target_idx in routes_idx:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        # Tính tọa độ tương đối
        relative_source = (int(img_shape[1] * source.x), int(img_shape[0] * source.y))
        relative_target = (int(img_shape[1] * target.x), int(img_shape[0] * target.y))

        routes.append(relative_source)
        routes.append(relative_target)

    return routes


def crop_face(image, routes):
    """Cắt vùng khuôn mặt từ ảnh gốc"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(routes, dtype=np.int32)], 255)
    face_extracted = cv2.bitwise_and(image, image, mask=mask)

    x_min, y_min = np.min(routes, axis=0)
    x_max, y_max = np.max(routes, axis=0)
    return face_extracted[y_min:y_max, x_min:x_max]


def process_image_folder(input_folder, output_folder):
    """Xử lý tất cả ảnh trong folder và lưu kết quả"""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Khởi tạo FaceMesh
    face_mesh = initialize_face_mesh()

    # Duyệt qua tất cả file trong folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Chỉ xử lý file ảnh
            input_path = os.path.join(input_folder, filename)

            # Đọc ảnh
            img = cv2.imread(input_path)
            if img is None:
                print(f"Không thể đọc ảnh: {input_path}")
                continue

            # Chuyển sang RGB và xử lý
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)

            if results.multi_face_landmarks:
                # Lấy khuôn mặt đầu tiên
                landmarks = results.multi_face_landmarks[0]
                face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL

                # Lấy danh sách điểm kết nối và tọa độ
                routes_idx = get_connected_landmarks(face_oval)
                routes = get_face_contour(landmarks, routes_idx, img.shape)
                face_cropped = crop_face(img, routes)

                # Tạo tên file đầu ra
                output_filename = f"cropped_{filename}"
                output_path = os.path.join(output_folder, output_filename)

                # Lưu ảnh đã cắt
                if face_cropped is None or face_cropped.size == 0:
                    print('No FaceCropped found!')
                    continue
                else:
                    cv2.imwrite(output_path, face_cropped)
                    print(f"Đã xử lý và lưu: {output_filename}")
            else:
                print(f"Không tìm thấy khuôn mặt trong: {filename}")

    # Giải phóng FaceMesh
    face_mesh.close()

if __name__ == "__main__":
    #img_path = r"D:\FPTUniversity\Capstone_Project\Preprocess_Data\frame_0005.jpg"
    inp_path = r"D:\FPTUniversity\Capstone_Project\Code_tesst\out3"
    out_path = r"D:\FPTUniversity\Capstone_Project\Drunk_Detection\test\drunk"
    process_image_folder(inp_path, out_path)