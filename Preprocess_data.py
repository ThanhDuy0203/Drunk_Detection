import cv2
import os
import numpy as np


class ImagePreprocessor:
    def __init__(self, input_folder, output_folder, size=(96, 170), grayscale=False, method="clahe"):
        """
        Khởi tạo class với thư mục ảnh đầu vào và đầu ra.
        :param input_folder: Thư mục chứa ảnh gốc.
        :param output_folder: Thư mục để lưu ảnh sau khi xử lý.
        :param size: Kích thước resize (mặc định: 128x128).
        :param grayscale: Có chuyển ảnh về grayscale không? (Mặc định: False).
        :param method: Phương pháp chuẩn hóa sáng ('hist_eq' hoặc 'clahe').
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.size = size
        self.grayscale = grayscale
        self.method = method

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # Tạo thư mục đầu ra nếu chưa có

    def process_image(self, image_path, output_path):
        """
        Xử lý một ảnh: Resize, có thể chuyển grayscale, chuẩn hóa độ sáng & tương phản.
        :param image_path: Đường dẫn ảnh đầu vào.
        :param output_path: Đường dẫn ảnh đầu ra.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi khi đọc ảnh: {image_path}")
            return

        # Resize ảnh
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        # Nếu grayscale=True, chuyển sang ảnh xám
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Chuẩn hóa độ sáng & tương phản chỉ áp dụng khi ảnh là grayscale
            if self.method == "hist_eq":
                image = cv2.equalizeHist(image)
            elif self.method == "clahe":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(image)

        # Lưu ảnh đã xử lý
        cv2.imwrite(output_path, image)
        print(f"Đã xử lý: {output_path}")

    def process_all_images(self):
        """
        Xử lý tất cả ảnh trong thư mục input_folder.
        """
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Chỉ xử lý ảnh
                input_path = os.path.join(self.input_folder, filename)
                output_path = os.path.join(self.output_folder, filename)
                self.process_image(input_path, output_path)