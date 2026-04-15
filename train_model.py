
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def ve_confusion_matrix(y_test, pred, ten_model):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = [["True Negative\n(TN)", "False Positive\n(FP)"],
              ["False Negative\n(FN)", "True Positive\n(TP)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]}",
                   ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tuoi", "Hu"])
    ax.set_yticklabels(["Tuoi", "Hu"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {ten_model}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("dac_trung_v2.csv")
X = df.drop("nhan", axis=1)
y = df["nhan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(f"Decision Tree: {accuracy_score(y_test, dt_pred):.2%}")
print(f"Random Forest: {accuracy_score(y_test, rf_pred):.2%}")

ve_confusion_matrix(y_test, dt_pred, "Decision Tree")
ve_confusion_matrix(y_test, rf_pred, "Random Forest")

joblib.dump(dt, "decision_tree.pkl")
joblib.dump(rf, "random_forest.pkl")
print("Da luu model!")
