import numpy as np
import cv2
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Tắt bớt các cảnh báo rác của TensorFlow cho đỡ rối màn hình
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("⏳ Đang tải mô hình CNN (MobileNetV2)...")
# Tải CNN bỏ lớp phân loại (include_top=False) và gộp đặc trưng (pooling='avg')
cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("✅ CNN đã sẵn sàng!")

def extract_features(image):
    """
    Hàm này dùng CNN quét ảnh và trả về 1280 đặc trưng (thay vì 5 như trước)
    """
    if image is None or not isinstance(image, np.ndarray):
        return None
        
    try:
        # 1. Resize về kích thước chuẩn của MobileNetV2
        img_resized = cv2.resize(image, (224, 224))
        
        # 2. OpenCV đọc ảnh hệ BGR, nhưng CNN cần RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Chuyển thành dạng Tensor và chuẩn hóa [-1, 1]
        x = np.expand_dims(img_rgb, axis=0)
        x = preprocess_input(x)
        
        # 4. Đưa qua CNN để trích xuất
        features = cnn_model.predict(x, verbose=0)
        
        return features[0]  # Trả về một mảng chứa 1280 con số
        
    except Exception as e:
        print(f"❌ Lỗi trích xuất CNN: {e}")
        return None