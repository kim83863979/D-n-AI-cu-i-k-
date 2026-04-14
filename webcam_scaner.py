import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ========== 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG ==========
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


# ========== 2. TẠO MODEL GIẢ LẬP (ĐỂ TEST NGAY) ==========
def create_demo_model():
    """
    Tạo model Random Forest với dữ liệu giả lập (vì chưa có ảnh thật).
    Sau này khi có dataset thật, thay bằng train từ file ảnh.
    """
    np.random.seed(42)
    X = []
    y = []
    
    # 100 mẫu trái cây tươi (label 0)
    for _ in range(100):
        X.append([
            np.random.uniform(20, 50),   # H
            np.random.uniform(40, 80),   # S
            np.random.uniform(100, 200), # V
            np.random.uniform(0.01, 0.05), # edge_density thấp
            np.random.uniform(30, 60)   # texture_score vừa
        ])
        y.append(0)
    
    # 100 mẫu trái cây hư (label 1)
    for _ in range(100):
        X.append([
            np.random.uniform(10, 30),   # H tối hơn
            np.random.uniform(10, 50),   # S thấp
            np.random.uniform(30, 100),  # V tối
            np.random.uniform(0.05, 0.15), # edge_density cao
            np.random.uniform(50, 90)    # texture_score cao
        ])
        y.append(1)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model


# ========== 3. HÀM CHÍNH QUÉT WEBCAM ==========
def scan_webcam():
    print("📸 Đang khởi tạo webcam...")
    
    # Tạo model (sau này có thể load từ file .pkl)
    model = create_demo_model()
    print("✅ Model đã sẵn sàng (dữ liệu giả lập)")
    
    # Mở webcam (0 là camera mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở webcam. Kiểm tra lại camera.")
        return
    
    print("\n" + "="*50)
    print("🎯 WEBCAM SCANNER ĐÃ SẴN SÀNG")
    print("="*50)
    print("   Nhấn SPACE (phím cách) để quét trái cây")
    print("   Nhấn ESC để thoát")
    print("="*50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lỗi đọc khung hình từ webcam")
            break
        
        # Lật ảnh theo chiều ngang cho giống gương
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Vẽ khung vuông ở giữa để người dùng đặt trái cây
        x1, y1 = w//4, h//4
        x2, y2 = 3*w//4, 3*h//4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Hiển thị hướng dẫn
        cv2.putText(frame, "DAT TRAI CAY VAO KHUNG", (w//2 - 150, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: Scan | ESC: Thoat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "CHO QUET...", (w//2 - 60, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hiển thị khung hình webcam
        cv2.imshow("FRUIT SCANNER", frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # Phím SPACE
            # Cắt vùng ảnh trong khung
            fruit_roi = frame[y1:y2, x1:x2]
            if fruit_roi.size == 0:
                continue
            
            # Gọi hàm trích xuất đặc trưng của bạn
            features = extract_features(fruit_roi)
            if features is None:
                print("❌ Không trích xuất được đặc trưng từ vùng ảnh này")
                continue
            
            # Dự đoán
            result = model.predict([features])[0]
            proba = model.predict_proba([features])[0]
            
            # Hiển thị kết quả
            if result == 0:
                label = f"TUOI - Do tin cay: {proba[0]*100:.1f}%"
                color = (0, 255, 0)  # Xanh
                print(f"✅ KẾT QUẢ: {label}")
            else:
                label = f"HU - Do tin cay: {proba[1]*100:.1f}%"
                color = (0, 0, 255)  # Đỏ
                print(f"⚠️ KẾT QUẢ: {label}")
            
            # Vẽ kết quả lên frame và hiển thị trong 2 giây
            result_frame = frame.copy()
            cv2.putText(result_frame, label, (50, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            cv2.imshow("FRUIT SCANNER", result_frame)
            cv2.waitKey(2000)  # 2 giây
            
        elif key == 27:  # ESC
            print("\n👋 Đã thoát chương trình")
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()


# ========== 4. CHẠY CHƯƠNG TRÌNH ==========
if __name__ == "__main__":
    scan_webcam()