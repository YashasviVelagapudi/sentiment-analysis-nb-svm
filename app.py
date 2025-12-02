import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# =========================================
# Streamlit Page Settings
# =========================================
st.set_page_config(
    page_title="Text Sentiment Analysis (NB vs SVM)",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# Load Saved Models & Metrics
# =========================================
nb = pickle.load(open("./models/nb_model.pkl", "rb"))
svm = pickle.load(open("./models/svm_model.pkl", "rb"))
tfidf = pickle.load(open("./models/tfidf.pkl", "rb"))
metrics = pickle.load(open("./models/metrics.pkl", "rb"))

nb_acc = metrics["nb_accuracy"]
svm_acc = metrics["svm_accuracy"]
nb_report = metrics["nb_report"]
svm_report = metrics["svm_report"]
cm_nb = metrics["cm_nb"]
cm_svm = metrics["cm_svm"]

# =========================================
# Sidebar Navigation
# =========================================
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Choose a section",
    ["🏠 Live Demo", "📊 Model Performance", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("**Built with ❤️ using Python & Streamlit**")
# =========================
#   SECTION: LIVE DEMO
# =========================
if section == "🏠 Live Demo":
    st.title("🔍 Live Sentiment Prediction")
    st.write("Enter a sentence below and see predictions from both **Naive Bayes** and **SVM** models.")

    # Main Input Text Box
    user_input = st.text_area(
        "Enter your sentence here:",
        height=120,
        placeholder="Type something like: I love this product!"
    )

    # Predict Button
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            # Convert text to TF-IDF vector
            vec = tfidf.transform([user_input])

            # Model Predictions
            nb_res = nb.predict(vec)[0]
            svm_res = svm.predict(vec)[0]

            # Two-column layout
            col1, col2 = st.columns(2)

            # ========= Naive Bayes Output =========
            with col1:
                st.subheader("Naive Bayes")
                if nb_res == 1:
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")

            # ========= SVM Output =========
            with col2:
                st.subheader("SVM (LinearSVC)")
                if svm_res == 1:
                    st.success("🟢 Positive")
                else:
                    st.error("🔴 Negative")

    # Optional: Clear Button
    st.button("Clear Input", on_click=lambda: st.session_state.update({"user_input": ""}))


# =========================================
# SECTION 2: MODEL PERFORMANCE
# =========================================
elif section == "📊 Model Performance":
    st.title("📈 Model Performance Dashboard")

    st.markdown("Visual comparison of **Naive Bayes** and **SVM** models.")

    # ------------------------------
    # Accuracy Progress Bars
    # ------------------------------
    st.subheader("🎯 Accuracy Overview")

    st.write(f"**Naive Bayes Accuracy:** {nb_acc:.4f}")
    st.progress(float(nb_acc))

    st.write(f"**SVM Accuracy:** {svm_acc:.4f}")
    st.progress(float(svm_acc))

    st.markdown("---")

    # ------------------------------
    # Accuracy Comparison Chart
    # ------------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Accuracy Table")

        acc_df = pd.DataFrame({
            "Model": ["Naive Bayes", "SVM"],
            "Accuracy": [nb_acc, svm_acc]
        }).set_index("Model")

        st.dataframe(acc_df.style.format("{:.3f}"))

    with col2:
        st.write("### Accuracy Bar Chart")

        fig_acc, ax = plt.subplots()
        ax.bar(["Naive Bayes", "SVM"], [nb_acc, svm_acc], color=["#4C9AFF", "#36CFC9"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Naive Bayes vs SVM")
        st.pyplot(fig_acc)

    st.markdown("---")

    # ------------------------------
    # Confusion Matrices
    # ------------------------------
    st.subheader("📌 Confusion Matrices")

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.write("### Naive Bayes")
        fig_nb, ax_nb = plt.subplots()
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        st.pyplot(fig_nb)

    with col_cm2:
        st.write("### SVM")
        fig_svm, ax_svm = plt.subplots()
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        st.pyplot(fig_svm)

    st.markdown("---")

    # ------------------------------
    # Comparison Table (Precision, Recall, F1)
    # ------------------------------
    st.subheader("📋 Detailed Metrics Comparison")

    comparison_df = pd.DataFrame([
        {
            "Model": "Naive Bayes",
            "Accuracy": nb_acc,
            "Precision": nb_report["weighted avg"]["precision"],
            "Recall": nb_report["weighted avg"]["recall"],
            "F1-Score": nb_report["weighted avg"]["f1-score"],
        },
        {
            "Model": "SVM",
            "Accuracy": svm_acc,
            "Precision": svm_report["weighted avg"]["precision"],
            "Recall": svm_report["weighted avg"]["recall"],
            "F1-Score": svm_report["weighted avg"]["f1-score"],
        }
    ]).set_index("Model")

    st.dataframe(comparison_df.style.format("{:.3f}"))

# =========================================
# SECTION 3: ABOUT PROJECT
# =========================================
elif section == "ℹ️ About Project":
    st.title("ℹ️ Project Overview")

    st.markdown("""
    ## 🎯 Objective
    Build a system that classifies text as **Positive (1)** or **Negative (0)** using  
    **Naive Bayes** and **Support Vector Machine (SVM)**.

    ## 🧠 Approach
    - Merge training + testing dataset  
    - Convert text → numerical using **TF-IDF**  
    - Train Naive Bayes  
    - Train SVM  
    - Compare their performance  

    ## 📊 Evaluation Metrics Used
    - Accuracy  
    - Precision  
    - Recall  
    - F1-score  
    - Confusion Matrix  

    ## 🛠️ Technologies Used
    - Python  
    - Scikit-learn  
    - Pandas  
    - Matplotlib + Seaborn  
    - Streamlit  
    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:13px;'>"
    "💻 Sentiment Analysis Dashboard • Built with ❤️ using Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True
)
