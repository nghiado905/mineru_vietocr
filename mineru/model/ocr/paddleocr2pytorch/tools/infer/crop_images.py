import os
import cv2
import numpy as np
from predict_det import TextDetector  # Giả sử bạn đã định nghĩa lớp TextDetector
import torch
def crop_and_save_text_regions(image_path, output_dir, args):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    # Khởi tạo TextDetector
    detector = TextDetector(args)

    # Phát hiện các vùng văn bản
    dt_boxes, elapse = detector(img)

    # Lặp qua từng hộp giới hạn và crop ảnh
    for i, box in enumerate(dt_boxes):
        # Chuyển đổi tọa độ hộp giới hạn sang định dạng integer
        box = box.astype(np.int32)

        # Lấy tọa độ để crop
        x_min = max(0, min(box[:, 0]))
        x_max = min(img.shape[1], max(box[:, 0]))
        y_min = max(0, min(box[:, 1]))
        y_max = min(img.shape[0], max(box[:, 1]))

        # Crop vùng văn bản từ ảnh gốc
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Đặt tên file đầu ra
        output_path = os.path.join(output_dir, f"text_region_{i}.png")

        # Lưu ảnh đã crop
        cv2.imwrite(output_path, cropped_img)
        print(f"Đã lưu vùng văn bản {i} vào: {output_path}")

# Cấu hình tham số cho TextDetector
class Args:
    det_algorithm = "DB"  # Có thể thay đổi thành EAST, SAST, PSE, FCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_limit_side_len = 960
    det_limit_type = "max"
    det_db_thresh = 0.3
    det_db_box_thresh = 0.6
    det_db_unclip_ratio = 1.5
    use_dilation = False
    det_db_score_mode = "fast"
    # det_model_path = "path/to/det_model.pth"  # Đường dẫn tới mô hình
    # det_yaml_path = "path/to/det_config.yaml"  # Đường dẫn tới file cấu hình

# Thực thi
args = Args()
image_path = "path/to/your/image.jpg"  # Đường dẫn tới ảnh đầu vào
output_dir = "path/to/output/folder"   # Thư mục lưu các ảnh đã crop
crop_and_save_text_regions(image_path, output_dir, args)