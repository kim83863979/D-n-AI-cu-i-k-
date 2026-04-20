import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from trich_xuat import extract_features  

def load_data_and_extract(thumuc_goc):
    X = []
    y = []
    
    if not os.path.exists(thumuc_goc):
        print(f"❌ LỖI: Không tìm thấy thư mục '{thumuc_goc}'.")
        return np.array([]), np.array([])

    print("🔍 Đang lùng sục dữ liệu trong thư mục...")
    
    # Lặp qua các thư mục con bên trong (VD: freshapples, rottenbanana...)
    for folder_name in os.listdir(thumuc_goc):
        folder_path = os.path.join(thumuc_goc, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Xác định nhãn (Label) dựa vào tên thư mục
        folder_name_lower = folder_name.lower()
        if "fresh" in folder_name_lower:
            label = 0
            label_name = "Tươi"
        elif "rotten" in folder_name_lower:
            label = 1
            label_name = "Hư"
        else:
            continue # Nếu có thư mục lạ không chứa chữ fresh/rotten thì bỏ qua
            
        print(f"\n🍎 Đang xử lý: {folder_name} (Nhãn: {label_name})...")
        
        # Đọc từng ảnh và đưa qua CNN
        dem_anh = 0
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                features = extract_features(img)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    dem_anh += 1
        print(f"    Đã xong {dem_anh} ảnh.")
            
    return np.array(X), np.array(y)

# ================= KỊCH BẢN CHÍNH =================
print("--- BẮT ĐẦU QUÁ TRÌNH HỌC ---")

# CHÚ Ý: Đường dẫn lúc này sẽ trỏ thẳng vào thư mục 'train'
# Theo ảnh của bạn, nó lồng 2 lần thư mục dataset
thu_muc_train = "dataset/dataset/train" 

X, y = load_data_and_extract(thu_muc_train)

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
