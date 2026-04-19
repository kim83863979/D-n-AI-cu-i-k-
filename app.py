from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import joblib
import os
from trich_xuat import extract_features # Gọi đôi mắt MobileNetV2

app = Flask(__name__)

# Tải bộ não AI
print("⏳ Đang tải mô hình Random Forest...")
try:
    model = joblib.load("random_forest.pkl")
    print("✅ Tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi tải mô hình: {e}")

# 1. Route hiển thị Giao diện chính (HTML)
@app.route('/')
def index():
    return render_template('index.html')

# 2. Route xử lý Upload Ảnh
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"result": "Không tìm thấy ảnh"})
    
    file = request.files['image']
    
    # Đọc ảnh từ dữ liệu web thành format của OpenCV
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"result": "Ảnh lỗi"})

    # Trích xuất 1280 đặc trưng
    features = extract_features(img)
    if features is None:
        return jsonify({"result": "Lỗi trích xuất CNN"})
        
    # Phân loại bằng Random Forest
    result = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    
    if result == 0:
        return jsonify({"result": f"✅ TƯƠI (Độ tin cậy: {proba[0]*100:.1f}%)"})
    else:
        return jsonify({"result": f"⚠️ HƯ (Độ tin cậy: {proba[1]*100:.1f}%)"})

# 3. Hàm xử lý Webcam Stream liên tục
def generate_frames():
    cap = cv2.VideoCapture(0) # Mở camera
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Vẽ khung vuông quét ảnh
        x1, y1 = w//4, h//4
        x2, y2 = 3*w//4, 3*h//4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Cắt ảnh trong khung để AI dự đoán
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            features = extract_features(roi)
            if features is not None:
                result = model.predict([features])[0]
                proba = model.predict_proba([features])[0]
                
                # In chữ lên màn hình Webcam
                if result == 0:
                    text = f"TUOI: {proba[0]*100:.1f}%"
                    color = (0, 255, 0)
                else:
                    text = f"HU: {proba[1]*100:.1f}%"
                    color = (0, 0, 255)
                    
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Chuyển đổi frame OpenCV thành dữ liệu JPEG để gửi lên Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 4. Route phát Video
@app.route('/video_feed')
def video_feed():
    # Sử dụng multipart response để nhúng luồng video liên tục vào thẻ <img>
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)