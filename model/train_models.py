import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
accuracy_score,
roc_auc_score,
precision_score,
recall_score,
f1_score,
matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw) # 0/1
    X_raw = df.drop(columns=[target_col])
    X = pd.get_dummies(X_raw, drop_first=True)
    original_columns = X_raw.columns.tolist()
    feature_names = X.columns.tolist()
    return X, y, df, label_encoder, original_columns, feature_names

def train_test_split_scaled(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def compute_metrics(y_true, y_pred, y_proba=None, average="binary"):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    if y_proba is not None:
        if y_proba.ndim > 1:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
        try:
            metrics["auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["auc"] = np.nan
    else:
        metrics["auc"] = np.nan

    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    return metrics


def train_all_models(csv_path: str, target_col: str):
    X, y, df, label_encoder, original_columns, feature_names = load_data(
        csv_path, target_col
    )
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = train_test_split_scaled(X, y)

    models = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)
    models["Logistic Regression"] = {
        "model": lr,
        "metrics": compute_metrics(y_test, y_pred_lr, y_proba_lr),
        "needs_scaling": True,
    }

    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)
    models["Decision Tree"] = {
        "model": dt,
        "metrics": compute_metrics(y_test, y_pred_dt, y_proba_dt),
        "needs_scaling": False,
    }

    # 3. kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    y_proba_knn = knn.predict_proba(X_test_scaled)
    models["kNN"] = {
        "model": knn,
        "metrics": compute_metrics(y_test, y_pred_knn, y_proba_knn),
        "needs_scaling": True,
    }

    # 4. Naive Bayes (Gaussian)
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    y_pred_nb = nb.predict(X_test_scaled)
    y_proba_nb = nb.predict_proba(X_test_scaled)
    models["Naive Bayes"] = {
        "model": nb,
        "metrics": compute_metrics(y_test, y_pred_nb, y_proba_nb),
        "needs_scaling": True,
    }

    # 5. Random Forest
    rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    models["Random Forest"] = {
        "model": rf,
        "metrics": compute_metrics(y_test, y_pred_rf, y_proba_rf),
        "needs_scaling": False,
    }

    # 6. XGBoost (binary – Adult dataset)
    xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    )
    xgb.fit(X_train, y_train) # y is integer 0/1
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)
    models["XGBoost"] = {
        "model": xgb,
        "metrics": compute_metrics(y_test, y_pred_xgb, y_proba_xgb),
        "needs_scaling": False,
    }

    return {
        "models": models,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "df": df,
        "feature_names": feature_names,
        "original_columns": original_columns,
        "label_encoder": label_encoder,
        "target_col": target_col,
    }


if __name__ == "__main__":
    result = train_all_models("model/adult.csv", "income")
    for name, info in result["models"].items():
        print(name, info["metrics"])