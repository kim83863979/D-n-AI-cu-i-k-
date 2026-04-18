import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from trich_xuat import extract_features  

def load_data_and_extract(thumuc_goc="dataset"):
    X = []
    y = []
    
    if not os.path.exists(thumuc_goc):
        print(f"❌ LỖI: Không tìm thấy thư mục '{thumuc_goc}'. Hãy đổi tên thư mục ảnh của bạn thành 'dataset' và để ngang hàng với file code.")
        return np.array([]), np.array([])

    print("🔍 Đang lùng sục dữ liệu trong thư mục...")
    # Lặp qua từng thư mục trái cây (Apple, Banana, Strawberry...)
    for fruit_folder in os.listdir(thumuc_goc):
        fruit_path = os.path.join(thumuc_goc, fruit_folder)
        if not os.path.isdir(fruit_path):
            continue
            
        print(f"\n🍎 Đang xử lý: {fruit_folder}...")
        
        # Lặp qua 2 thư mục Fresh và Rotten bên trong
        for condition in ["Fresh", "Rotten"]:
            condition_path = os.path.join(fruit_path, condition)
            if not os.path.exists(condition_path):
                continue
                
            # Gán nhãn: Fresh = 0 (Tươi), Rotten = 1 (Hư)
            label = 0 if condition == "Fresh" else 1
            label_name = "Tươi" if label == 0 else "Hư"
            
            print(f"  👉 Trích xuất ảnh {label_name}...")
            
            # Đọc từng ảnh và đưa qua CNN
            dem_anh = 0
            for img_name in os.listdir(condition_path):
                img_path = os.path.join(condition_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    features = extract_features(img)
                    if features is not None:
                        X.append(features)
                        y.append(label)
                        dem_anh += 1
            print(f"     Đã xong {dem_anh} ảnh.")
            
    return np.array(X), np.array(y)

# ================= KỊCH BẢN CHÍNH =================
print("--- BẮT ĐẦU QUÁ TRÌNH HỌC ---")
X, y = load_data_and_extract("dataset/Fruit Freshness Dataset")

if len(X) == 0:
    print("\n❌ Lỗi: Không có dữ liệu để học. Code đã dừng.")
else:
    print(f"\n✅ Đã trích xuất tổng cộng {len(X)} bức ảnh.")
    print("🧠 Đang tìm kiếm BỘ THÔNG SỐ TỐI ƯU cho Random Forest (Grid Search)...")
    
    # 1. Định nghĩa các "ứng cử viên" muốn máy tính chạy thử
    param_grid = {
        'n_estimators': [50, 100, 200],  # Số lượng cây: Thử ít, vừa, và nhiều
        'max_depth': [10, 20, None]      # Độ sâu của cây: Ngắn, dài, và không giới hạn
    }
    
    # 2. Tạo mô hình cơ sở
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 3. Thiết lập bộ quét GridSearch
    # cv=3: Chia dữ liệu làm 3 phần để test chéo (Cross-Validation) giúp đánh giá khách quan
    # verbose=2: In quá trình chạy ra màn hình để bạn đỡ sốt ruột
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                               cv=3, scoring='accuracy', verbose=2)
    
    # 4. Bắt đầu ép máy tính thử nghiệm tất cả các trường hợp!
    grid_search.fit(X, y)
    
    # 5. Lấy ra con AI "Thủ khoa"
    best_rf = grid_search.best_estimator_
    
    print("\n" + "="*50)
    print(f"🌟 BỘ THÔNG SỐ TỐT NHẤT TÌM ĐƯỢC: {grid_search.best_params_}")
    print("="*50)
    
    # 6. Kiểm tra độ chính xác của Thủ khoa
    pred = best_rf.predict(X)
    print(f"🎯 Độ chính xác (trên tập huấn luyện): {accuracy_score(y, pred):.2%}")
    
    # 7. Lưu bộ não xịn nhất
    joblib.dump(best_rf, "random_forest.pkl")
    print("💾 Đã lưu 'bộ não' TỐI ƯU thành công: random_forest.pkl")
    print("\n🎉 HOÀN TẤT! Bạn đã có một mô hình siêu xịn.")
