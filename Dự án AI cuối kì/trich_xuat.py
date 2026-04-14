import cv2
import numpy as np

def extract_features(image):
    """
    Hàm trích xuất đặc trưng hình ảnh cho mô hình phân loại trái cây.
    
    Tham số:
        image (numpy.ndarray): Ma trận ảnh đọc bằng cv2.imread() hoặc từ camera frame.
        
    Trả về:
        list: Vector chứa 5 giá trị đặc trưng [H, S, V, Edge_Density, Texture_Std].
              Trả về None nếu ảnh đầu vào bị lỗi.
    """
    # 1. Kiểm tra đầu vào 
    if image is None or not isinstance(image, np.ndarray):
        print("Lỗi: Dữ liệu ảnh đầu vào không hợp lệ.")
        return None

    try:
        # 2. Chuẩn hóa kích thước 
        img_resized = cv2.resize(image, (200, 200))
        
        # 3. Trích xuất HSV (Màu sắc)
        hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv_img[:, :, 0]) # Kênh Hue
        s_mean = np.mean(hsv_img[:, :, 1]) # Kênh Saturation
        v_mean = np.mean(hsv_img[:, :, 2]) # Kênh Value
        
        # 4. Trích xuất Canny (Đường biên/Vết dập nứt)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blurred_for_canny = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred_for_canny, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 5. Trích xuất Local Mean (Kết cấu bề mặt / Độ mốc)
        local_mean_img = cv2.blur(gray, (5, 5))
        texture_score = np.std(local_mean_img) 

        # 6. Đóng gói thành Vector 1 chiều
        feature_vector = [h_mean, s_mean, v_mean, edge_density, texture_score]
        
        return feature_vector

    except Exception as e:
        print(f"Lỗi trong quá trình trích xuất đặc trưng: {e}")
        return None