import cv2
import numpy as np
import joblib

from trich_xuat import extract_features 

# ========== HÀM CHÍNH QUÉT WEBCAM ==========
def scan_webcam():
    print("📸 Đang khởi tạo webcam...")
    
    try:
        # Load mô hình thật đã được huấn luyện bằng CNN
        model = joblib.load("random_forest.pkl")
        print("✅ Đã tải model thật (random_forest.pkl) thành công!")
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy file 'random_forest.pkl'. Hãy chạy file train_cnn_rf.py trước!")
        return
    
    # Mở webcam (0 là camera mặc định)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ LỖI: Không thể mở webcam. Kiểm tra lại camera.")
        return
    
    print("\n" + "="*50)
    print("🎯 WEBCAM SCANNER ĐÃ SẴN SÀNG (Powered by CNN)")
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
            # Cắt vùng ảnh trong khung vuông
            fruit_roi = frame[y1:y2, x1:x2]
            if fruit_roi.size == 0:
                continue
            
            # Gọi hàm trích xuất CNN
            features = extract_features(fruit_roi)
            
            if features is None:
                print("❌ Không trích xuất được đặc trưng từ vùng ảnh này")
                continue
            
            try:
                # Đưa 1280 đặc trưng vào dự đoán
                result = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                
                # Hiển thị kết quả (0: Tươi, 1: Hư)
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
                cv2.waitKey(2000)  # Dừng 2 giây để xem kết quả
                
            except Exception as e:
                print(f"❌ Lỗi khi dự đoán: {e}")
                
        elif key == 27:  # ESC
            print("\n👋 Đã thoát chương trình")
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

# ========== CHẠY CHƯƠNG TRÌNH ==========
if __name__ == "__main__":
    scan_webcam()
