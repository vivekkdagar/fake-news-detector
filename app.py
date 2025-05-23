import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from fpdf import FPDF

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detection App")
st.markdown("Welcome to the **Fake News Detection App**! Paste any news article below, choose a machine learning model, and find out if it's real or fake. 🔍")

# === PDF generation function ===
def generate_pdf(news_text, model_name, accuracy, precision, recall, f1, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Fake News Detection Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Model used: {model_name}", ln=True)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, f"Accuracy: {accuracy * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Precision: {precision * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Recall: {recall * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"F1-score: {f1 * 100:.2f}%", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "News Article:", ln=True)
    pdf.set_font("Arial", "", 12)
    for line in news_text.split('\n'):
        pdf.multi_cell(0, 8, line)

    return pdf.output(dest='S').encode('latin-1')

# === Load dataset ===
if os.path.exists("fake.csv") and os.path.exists("true.csv"):
    df_fake = pd.read_csv("fake.csv")
    df_true = pd.read_csv("true.csv")

    df_fake["label"] = 0
    df_true["label"] = 1
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.drop(["title", "subject", "date"], axis=1, errors='ignore')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sidebar: Label distribution plot
    with st.sidebar:
        st.subheader("📊 Label Distribution")
        fig, ax = plt.subplots()
        sb.countplot(x='label', data=df, hue='label', palette='hls', ax=ax, legend=False)
        ax.set_xlabel("Label (0 = Fake, 1 = Real)")
        ax.set_ylabel("Count")
        ax.set_title("Label Distribution")
        st.pyplot(fig)

    # Vectorize and split
    vectorizer_path = "tfidf_vectorizer.joblib"
    model_path = "trained_model.joblib"
    saved_model_name_path = "saved_model_name.txt"

    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(df["text"])
        joblib.dump(vectorizer, vectorizer_path)

    X = vectorizer.transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    model_options = {
        "Naive Bayes": MultinomialNB,
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
        "Random Forest": lambda: RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": lambda: SVC(probability=True, random_state=42),
    }

    st.markdown("---")
    st.subheader("🧪 Choose Machine Learning Model")
    selected_model_name = st.selectbox("Model", list(model_options.keys()), index=1)

    st.markdown("---")
    st.subheader("📝 Paste the News Article")
    user_input = st.text_area("Enter the news article text here:", height=200)

    def predict_news(text, model):
        vectorized_text = vectorizer.transform([text])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)[0]
            max_prob = np.max(probabilities)
            predicted_label = np.argmax(probabilities)
        else:
            predicted_label = model.predict(vectorized_text)[0]
            max_prob = 0.5

        if max_prob < 0.6:
            return "Fake News (Uncertain)"
        elif predicted_label == 0:
            return "Fake News"
        else:
            return "Real News"

    model = None
    if os.path.exists(model_path) and os.path.exists(saved_model_name_path):
        with open(saved_model_name_path, "r") as f:
            saved_model_name = f.read().strip()
        if saved_model_name == selected_model_name:
            model = joblib.load(model_path)

    if st.button("🔍 Predict"):
        if user_input.strip():
            with st.spinner("Analyzing the article..."):
                if model is None:
                    ModelClass = model_options[selected_model_name]
                    model = ModelClass() if callable(ModelClass) else ModelClass
                    model.fit(X_train, y_train)
                    joblib.dump(model, model_path)
                    with open(saved_model_name_path, "w") as f:
                        f.write(selected_model_name)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.markdown("---")
                st.subheader(f"📈 Model Performance: {selected_model_name}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc * 100:.2f}%")
                col2.metric("Precision", f"{prec * 100:.2f}%")
                col3.metric("Recall", f"{rec * 100:.2f}%")
                col4.metric("F1 Score", f"{f1 * 100:.2f}%")

                st.markdown("#### Confusion Matrix")
                fig_cm, ax = plt.subplots()
                im = ax.imshow(cm, cmap="viridis")
                plt.xticks([0, 1], ["Fake", "Real"])
                plt.yticks([0, 1], ["Fake", "Real"])
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{selected_model_name} Confusion Matrix")
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, str(z), ha='center', va='center',
                            color='white' if cm[i][j] > cm.max()/2 else 'black')
                st.pyplot(fig_cm)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    st.markdown("#### ROC Curve")
                    fig_roc, ax = plt.subplots()
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'{selected_model_name} ROC Curve')
                    ax.legend(loc="lower right")
                    st.pyplot(fig_roc)

                if selected_model_name in ["Logistic Regression", "Support Vector Machine"]:
                    st.markdown("#### 🔤 Top Informative TF-IDF Terms")
                    coef = model.coef_[0]
                    feature_names = vectorizer.get_feature_names_out()
                    top_pos_indices = np.argsort(coef)[-15:][::-1]
                    top_neg_indices = np.argsort(coef)[:15]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top Real News Words**")
                        for idx in top_pos_indices:
                            st.write(f"{feature_names[idx]} ({coef[idx]:.4f})")
                    with col2:
                        st.markdown("**Top Fake News Words**")
                        for idx in top_neg_indices:
                            st.write(f"{feature_names[idx]} ({coef[idx]:.4f})")

                prediction = predict_news(user_input, model)
                st.markdown("#### 🧾 Final Prediction")
                st.success(f"**Prediction:** {prediction}")

                pdf_bytes = generate_pdf(user_input, selected_model_name, acc, prec, rec, f1, prediction)
                st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="fake_news_report.pdf", mime="application/pdf")

        else:
            st.warning("⚠️ Please enter a news article to predict.")

else:
    st.error("❌ Dataset files 'fake.csv' and 'true.csv' not found. Please upload them to proceed.")