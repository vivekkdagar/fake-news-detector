# fake-news-detector
This is a Streamlit-based web application that uses machine learning to classify news articles as **Real** or **Fake**. Users can paste any news article, select a machine learning model, and get predictions along with performance metrics. A downloadable PDF report is also available.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dailybugle-news-detector.streamlit.app/)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Streamlit-lightgrey)
![Made with scikit-learn](https://img.shields.io/badge/Made%20with-scikit--learn-orange)

<hr/>

## 🚀 Live Demo

Check out the live app here:  
[https://dailybugle-news-detector.streamlit.app/](https://dailybugle-news-detector.streamlit.app/)

![Demo Screenshot](https://github.com/vivekkdagar/fake-news-detector/blob/main/Screenshot_23-5-2025_224244_dailybugle-news-detector.streamlit.app.jpeg)

---

## Features

- Paste custom news articles for evaluation.
- Choose from six ML models:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors
  - Decision Tree
  - Support Vector Machine (SVM)
- Visualizations:
  - Confusion Matrix
  - ROC Curve
  - TF-IDF Top Features (for Logistic Regression and SVM)
  - Label distribution (Sidebar)
- PDF report generation with prediction and metrics.
- Upload custom `fake.csv` and `true.csv` files if not found.

---

## Local Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/vivekkdagar/fake-news-detector.git
   cd fake-news-detector
   ```
2. **Install dependencies**

  ```bash
  pip install -r requirements.txt
  ```
3. **Add the datasets** : Place fake.csv and true.csv in the root directory. If unavailable, upload them via the app interface.
**Dataset Format** : Each CSV should include a text column with the news content. Optional columns like title, subject, or date will be ignored.

4. **Run the App**
```bash
python3 -m streamlit run app.py
```
The app will open in your browser where you can paste your article, select a model, and get predictions.

## 📄 PDF Report
Download a detailed PDF report including:

- ✅ Model used  
- 📊 Prediction result  
- 📈 Accuracy, Precision, Recall, F1-score  
- 📰 Full article text

## 📦 Dependencies

- `streamlit`  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `fpdf`  
- `joblib`  

### 📥 Install all dependencies

```bash
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
