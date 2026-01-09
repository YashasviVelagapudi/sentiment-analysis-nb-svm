# 📊 Sentiment Analysis Dashboard (Naive Bayes vs SVM)

A complete end-to-end **Text Sentiment Analysis** project that classifies text as **Positive (1)** or **Negative (0)** using **Naive Bayes** and **Support Vector Machine (SVM)**.  
The project includes model training, evaluation, and an interactive **Streamlit dashboard** for live prediction and performance comparison.

---

## 🚀 Features

- Text classification using **TF-IDF**
- Two ML models:
  - Naive Bayes
  - Support Vector Machine (LinearSVC)
- Interactive **Streamlit UI**
- Live sentiment prediction
- Accuracy comparison
- Confusion matrix visualization
- Model persistence using Pickle

---

## 🗂️ Project Structure

```
SentimentAnalysisProject/
│
├── src/
│   └── main.py          # Model training & evaluation
│
├── models/
│   ├── nb_model.pkl     # Trained Naive Bayes model
│   ├── svm_model.pkl    # Trained SVM model
│   ├── tfidf.pkl        # TF-IDF vectorizer
│   └── metrics.pkl      # Accuracy & confusion matrices
│
├── app.py               # Streamlit dashboard
├── movie_data.csv       # Dataset (movie reviews)
├── README.md            # Project documentation
└── .gitignore
```

---

## 📁 Dataset

- **Dataset:** Movie Reviews Sentiment Dataset  
- **Columns:**
  - `review` – Text review
  - `sentiment` – Label (`1 = Positive`, `0 = Negative`)
- The dataset is split into training and testing internally using `train_test_split`.

---

## ⚙️ Technologies Used

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 🧠 Model Pipeline

1. Load dataset
2. Preprocess text
3. Convert text to numerical form using **TF-IDF**
4. Train:
   - Multinomial Naive Bayes
   - Linear Support Vector Machine
5. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
6. Save trained models and metrics
7. Visualize results in Streamlit

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```

### 2️⃣ Train the Models

```bash
python src/main.py
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 📊 Streamlit Dashboard Sections

- **Live Demo** – Enter text and get predictions from NB & SVM
- **Model Performance** – Accuracy, bar chart, confusion matrices
- **About Project** – Objective, approach, and technologies used

---

## 🧪 Sample Test Sentences

**Positive**
- I absolutely loved this movie.
- The acting and direction were fantastic.

**Negative**
- This was a complete waste of time.
- The plot was boring and predictable.

**Mixed**
- The story was good, but the execution was poor.

---

## 📌 Future Enhancements

- Add neutral sentiment class
- Improve preprocessing (lemmatization, bigrams)
- Deploy on Streamlit Cloud
- Add deep learning models (LSTM / BERT)

---

## 👤 Author

**Yashasvi Velagapudi**  
Computer Science Student  

---

## 📜 License

This project is for academic and learning purposes.
