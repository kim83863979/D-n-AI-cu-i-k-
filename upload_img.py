import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import joblib
import numpy as np

from trich_xuat import extract_features

MODEL_PATH = "random_forest.pkl" 


def convert_prediction_to_label(pred):
    """
    Hỗ trợ cả trường hợp model trả về số (0/1) hoặc chuỗi.
    """
    # Nếu model trả về kiểu số
    if isinstance(pred, (int, np.integer, float, np.floating)):
        return "Tươi" if int(pred) == 0 else "Hư"

    # Nếu model trả về kiểu chuỗi
    pred_str = str(pred).strip().lower()
    if pred_str in ["0", "tuoi", "tươi", "fresh"]:
        return "Tươi"
    return "Hư"


class FruitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân loại trái cây tươi / hư")
        self.root.geometry("800x650")
        self.root.configure(bg="#f5f5f5")

        self.model = None
        self.tk_image = None

        self.load_model()

        title = tk.Label(
            root,
            text="PHÂN LOẠI TRÁI CÂY TƯƠI / HƯ",
            font=("Arial", 20, "bold"),
            bg="#f5f5f5",
            fg="#222222"
        )
        title.pack(pady=15)

        self.btn_upload = tk.Button(
            root,
            text="Upload Ảnh",
            font=("Arial", 14, "bold"),
            width=20,
            height=2,
            command=self.upload_image,
            bg="#4CAF50",
            fg="white"
        )
        self.btn_upload.pack(pady=10)

        self.image_label = tk.Label(root, bg="#f5f5f5")
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="Kết quả: Chưa có ảnh",
            font=("Arial", 18, "bold"),
            bg="#f5f5f5",
            fg="#333333"
        )
        self.result_label.pack(pady=10)

        self.detail_label = tk.Label(
            root,
            text="",
            font=("Arial", 12),
            bg="#f5f5f5",
            fg="#555555",
            justify="left"
        )
        self.detail_label.pack(pady=5)

    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            messagebox.showerror(
                "Lỗi model",
                f"Không thể load model từ file '{MODEL_PATH}'.\n\nChi tiết: {e}"
            )
            self.model = None

    def upload_image(self):
        if self.model is None:
            messagebox.showerror("Lỗi", "Model chưa được nạp.")
            return

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh trái cây",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # Đọc ảnh bằng OpenCV
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh. Hãy chọn ảnh hợp lệ.")
            return

        # Hiển thị ảnh lên giao diện
        self.show_image(image_bgr)

        # Trích xuất đặc trưng
        features = extract_features(image_bgr)
        if features is None:
            messagebox.showerror("Lỗi", "Không trích xuất được đặc trưng từ ảnh.")
            return

        try:
            # Dự đoán
            prediction = self.model.predict([features])[0]
            label = convert_prediction_to_label(prediction)

            # Độ tin cậy nếu model hỗ trợ predict_proba
            confidence_text = ""
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba([features])[0]
                confidence = np.max(probs) * 100
                confidence_text = f"Độ tin cậy: {confidence:.2f}%"

            # Hiển thị kết quả
            if label == "Tươi":
                self.result_label.config(text=f"Kết quả: {label}", fg="green")
            else:
                self.result_label.config(text=f"Kết quả: {label}", fg="red")

            self.detail_label.config(text=confidence_text)

        except Exception as e:
            messagebox.showerror("Lỗi dự đoán", f"Dự đoán thất bại.\n\nChi tiết: {e}")

    def show_image(self, image_bgr):
        # Chuyển BGR -> RGB để hiển thị đúng màu
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Resize ảnh để vừa cửa sổ
        h, w = image_rgb.shape[:2]
        max_w, max_h = 500, 350

        scale = min(max_w / w, max_h / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image_rgb = cv2.resize(image_rgb, (new_w, new_h))

        pil_image = Image.fromarray(image_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        self.image_label.config(image=self.tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitClassifierApp(root)
    root.mainloop()
