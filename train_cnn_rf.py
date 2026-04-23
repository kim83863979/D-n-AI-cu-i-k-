import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# from trich_xuat import extract_features

def load_data_and_extract(thumuc_goc):
    X = []
    y = []

    tong_tuoi = 0
    tong_hu = 0

    if not os.path.exists(thumuc_goc):
        print(f" LỖI: Không tìm thấy thư mục '{thumuc_goc}'.")
        return np.array([]), np.array([])

    print(" Đang đọc dữ liệu trong thư mục...")

    for folder_name in os.listdir(thumuc_goc):
        folder_path = os.path.join(thumuc_goc, folder_name)

        if not os.path.isdir(folder_path):
            continue

        folder_name_lower = folder_name.lower()

        if "fresh" in folder_name_lower:
            label = 0
            label_name = "Tươi"
        elif "rotten" in folder_name_lower:
            label = 1
            label_name = "Hư"
        else:
            continue

        print(f"\n Đang xử lý: {folder_name} (Nhãn: {label_name})...")

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

        if label == 0:
            tong_tuoi += dem_anh
        else:
            tong_hu += dem_anh

        print(f"    Đã xong {dem_anh} ảnh.")

    print("\n THỐNG KÊ:")
    print(f"    Tổng ảnh Tươi : {tong_tuoi}")
    print(f"    Tổng ảnh Hư   : {tong_hu}")
    print(f"    Tổng cộng     : {tong_tuoi + tong_hu}")

    return np.array(X), np.array(y)


# ================= KỊCH BẢN CHÍNH =================

print("--- BẮT ĐẦU QUÁ TRÌNH HỌC ---")

# TRAIN
thu_muc_train = "/content/fruits/dataset/dataset/train"
print("\n================ TRAIN DATA ================\n")
X, y = load_data_and_extract(thu_muc_train)

# TEST
thu_muc_test = "/content/fruits/dataset/dataset//test"
print("\n================ TEST DATA ================\n")
X_test, y_test = load_data_and_extract(thu_muc_test)

if len(X) == 0:
    print("\n Lỗi: Không có dữ liệu để học. Code đã dừng.")

else:
    print("\n Đang tìm kiếm BỘ THÔNG SỐ TỐI ƯU cho Random Forest...")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }

    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2
    )

    grid_search.fit(X, y)

    best_rf = grid_search.best_estimator_

    print("\n" + "=" * 55)
    print(f" BỘ THÔNG SỐ TỐT NHẤT: {grid_search.best_params_}")
    print("=" * 55)

    # TRAIN
    pred_train = best_rf.predict(X)
    acc_train = accuracy_score(y, pred_train)

    print("\n KẾT QUẢ TRAIN")
    print(f"    Số ảnh train        : {len(X)}")
    print(f"    Accuracy train     : {acc_train:.2%}")

    # TEST
    if len(X_test) > 0:
        pred_test = best_rf.predict(X_test)
        acc_test = accuracy_score(y_test, pred_test)

        print("\n KẾT QUẢ TEST")
        print(f"    Số ảnh test         : {len(X_test)}")
        print(f"    Accuracy test      : {acc_test:.2%}")

    joblib.dump(best_rf, "random_forest.pkl")

    print("\n Đã lưu model: random_forest.pkl")
    print(" HOÀN TẤT!")


 # ĐÁNH GIÁ CHI TIẾT
    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_data, y_data, ten_tap="DATA"):
    y_pred = model.predict(X_data)

    acc = accuracy_score(y_data, y_pred)
    pre = precision_score(y_data, y_pred)
    rec = recall_score(y_data, y_pred)
    f1  = f1_score(y_data, y_pred)

    cm = confusion_matrix(y_data, y_pred)
    tn, fp, fn, tp = cm.ravel()

    dung = tn + tp
    sai  = fp + fn

    print("=" * 60)
    print(f" ĐÁNH GIÁ {ten_tap.upper()}")
    print("=" * 60)

    print(f" Tổng số ảnh   : {len(y_data)}")
    print(f" Đoán đúng     : {dung}")
    print(f" Đoán sai      : {sai}")

    print(f"\n Accuracy      : {acc:.2%}")
    print(f" Precision     : {pre:.2%}")
    print(f" Recall        : {rec:.2%}")
    print(f" F1-score      : {f1:.2%}")

    print("\n Ma trận nhầm lẫn:")
    print(cm)

    print("\n Classification Report:")
    print(classification_report(
        y_data,
        y_pred,
        target_names=["Fresh", "Rotten"]
    ))


# ĐÁNH GIÁ TRAIN
evaluate_model(best_rf, X, y, "Train")

# ĐÁNH GIÁ TEST
evaluate_model(best_rf, X_test, y_test, "Test")