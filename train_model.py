import os
import warnings
import logging
from datetime import datetime, timezone

import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler # <<< ИЗМЕНЕНИЕ: Импортируем StandardScaler

# Подавляем предупреждения и настраиваем логирование
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

def main():
    # 1) Загружаем данные
    df = pd.read_csv("training_dataset.csv")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].copy()
    y = df["label"].copy()

    # 2) Убираем NaN
    before = len(X)
    X["label"] = y
    X = X.dropna()
    y = X.pop("label")
    print(f"Rows: before dropna = {before}, after = {len(X)}")

    # 3) Если одного из классов <10 — делаем SMOTE
    counts = y.value_counts()
    if counts.min() < 10:
        print("⚠️ Слишком мало примеров — делаем SMOTE")
        k_neighbors = min(counts.min() - 1, 5) # k_neighbors должно быть меньше, чем примеров
        if k_neighbors > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            try:
                X, y = smote.fit_resample(X, y)
                print("After SMOTE:", y.value_counts().to_dict())
            except ValueError as e:
                print(f"❗ Ошибка SMOTE: {e}. Пропускаем oversampling.")
        else:
            print("❗ Недостаточно примеров для SMOTE. Пропускаем.")


    # 4) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        shuffle=False, # Для временных рядов
        random_state=42
    )

    # 5) Убираем константные признаки
    if X_train.shape[0] > 1:
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(X_train)
        keep_idx = sel.get_support(indices=True)
        original_feature_cols = X_train.columns.tolist() # Получаем имена колонок
        feature_cols = [original_feature_cols[i] for i in keep_idx]
        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]
        print(f"Features left after VarianceThreshold: {len(feature_cols)}")
        if not feature_cols:
            raise RuntimeError("Нет признаков для обучения — все константны")
    else:
        print("❗ Only one train sample — пропускаем VarianceThreshold")

    # <<< ИЗМЕНЕНИЕ: Добавляем масштабирование данных >>>
    # ----------------------------------------------------
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    # Обучаем scaler ТОЛЬКО на обучающих данных
    X_train_scaled = scaler.fit_transform(X_train)
    # Применяем тот же scaler к тестовым данным
    X_test_scaled = scaler.transform(X_test)
    # ----------------------------------------------------


    # 6) Инициализируем модель
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=1,
        min_data_in_bin=1,
        class_weight="balanced",
        verbosity=-1
    )

    # 7) Обучаем на МАСШТАБИРОВАННЫХ данных
    model.fit(
        X_train_scaled, # <<< ИЗМЕНЕНИЕ
        y_train,
        eval_set=[(X_test_scaled, y_test)], # <<< ИЗМЕНЕНИЕ
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=20)
        ]
    )

    # 8) Предсказания и отчет на МАСШТАБИРОВАННЫХ данных
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # <<< ИЗМЕНЕНИЕ
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 9) ROC-анализ и подбор порога Youden
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        if len(thr) > 1:
            fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:] # убираем inf
            youden = tpr - fpr
            best_idx = youden.argmax()
            best_thr = thr[best_idx]
            print(f"ROC AUC: {auc:.4f}, Optimal threshold by Youden: {best_thr:.4f}")

            plt.figure(figsize=(6,4))
            sns.lineplot(x=fpr, y=tpr, label="ROC Curve")
            plt.plot([0,1], [0,1], "--", color="gray")
            plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label=f"Best thr = {best_thr:.2f}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC AUC: {auc:.4f}")
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        print("❗ Only one class in test — skipping ROC")

    # 10) Сохраняем модель и SCALER
    save_model(model, scaler, feature_cols) # <<< ИЗМЕНЕНИЕ

def save_model(model, scaler, features): # <<< ИЗМЕНЕНИЕ
    os.makedirs("models", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    ver = f"models/trade_model_{ts}.pkl"
    
    # <<< ИЗМЕНЕНИЕ: Сохраняем в словарь и модель, и scaler
    data_to_save = {"model": model, "scaler": scaler, "features": features}
    
    joblib.dump(data_to_save, ver)
    # Для обратной совместимости можно оставить и старый формат, если он где-то нужен
    joblib.dump(data_to_save, "trade_model.pkl") 
    print(f"✅ Models and scaler saved: {ver} & trade_model.pkl")

if __name__ == "__main__":
    main()