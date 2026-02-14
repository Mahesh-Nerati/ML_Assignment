import streamlit as st
import pandas as pd
import numpy as np

from model.train_models import train_all_models

st.set_page_config(
page_title="Machine Learning Classification Models – Assignment 2",
layout="wide"
)


@st.cache_resource
def get_trained_models(csv_path, target_col):
    return train_all_models(csv_path, target_col)


def main():
    st.title("Machine Learning Classification Models – Assignment 2")
    st.sidebar.header("Configuration")
    csv_path = st.sidebar.text_input("Dataset CSV path", "model/adult.csv")
    target_col = st.sidebar.text_input("Target column name", "income")
    if st.sidebar.button("Train / Reload Models"):
        st.session_state["trained"] = True
    if "trained" not in st.session_state:
        st.info("Set dataset path and target column, then click 'Train / Reload Models'.")
        return
    with st.spinner("Training models..."):
        result = get_trained_models(csv_path, target_col)
        print(result)

    models = result["models"]
    df = result["df"]
    encoded_feature_names = result["feature_names"]
    original_cols = result["original_columns"]
    X_train = result["X_train"]
    scaler = result["scaler"]
    label_encoder = result["label_encoder"]

    # Metrics table
    st.subheader("Model Evaluation Metrics")
    rows = []
    for name, info in models.items():
        m = info["metrics"]
        rows.append({
            "ML Model Name": name,
            "Accuracy": round(m["accuracy"], 4),
            "AUC": None if pd.isna(m["auc"]) else round(m["auc"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "MCC": round(m["mcc"], 4),
        })
    st.dataframe(pd.DataFrame(rows))

    # Interactive prediction
    st.subheader("Interactive Prediction")
    model_name = st.selectbox(
    "Select model",
    list(models.keys())
    )
    selected = models[model_name]
    selected_model = selected["model"]
    needs_scaling = selected["needs_scaling"]
    st.markdown("Enter feature values:")
    input_raw = {}
    cols = st.columns(2)
    for i, feat in enumerate(original_cols):
        with cols[i % 2]:
            if np.issubdtype(df[feat].dtype, np.number):
                default_val = float(df[feat].median())
                input_raw[feat] = st.number_input(feat, value=default_val)
            else:
                mode_val = str(df[feat].mode().iloc[0])
                input_raw[feat] = st.text_input(feat, value=mode_val)

    if st.button("Predict"):
        input_df_raw = pd.DataFrame([input_raw])
        input_encoded = pd.get_dummies(input_df_raw, drop_first=True)
        for col in encoded_feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[encoded_feature_names]
        if needs_scaling:
            input_arr = scaler.transform(input_encoded)
        else:
            input_arr = input_encoded.values

        pred_encoded = selected_model.predict(input_arr)[0]
        try:
            proba = selected_model.predict_proba(input_arr)[0]
        except Exception:
            proba = None

        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        st.markdown(f"**Predicted class:** {pred_label}")
        if proba is not None:
            st.write("Class probabilities (per label order):")
            st.write(
                {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}
            )


if __name__ == "__main__":
    main()