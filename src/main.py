import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# -----------------------------------
# 1. Load the CSVs
# -----------------------------------
train = pd.read_csv("./dataset/train_data.csv")
test = pd.read_csv("./dataset/test_data.csv")

# Combine both into one dataset
data = pd.concat([train, test], ignore_index=True)

# Features & labels
X = data['sentence']
y = data['sentiment']   # 0 = negative, 1 = positive

# -----------------------------------
# 2. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 3. TF-IDF Vectorization
# -----------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=50000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------------
# 4. Train Naive Bayes
# -----------------------------------
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)

# -----------------------------------
# 5. Train SVM
# -----------------------------------
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

# Calculate accuracy
nb_accuracy = accuracy_score(y_test, nb_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n================ NB RESULTS ================")
print("Naive Bayes Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_pred))

print("\n================ SVM RESULTS ================")
print("SVM Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_pred))


# Generate reports as dicts (for table later)
nb_report = classification_report(y_test, nb_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)

# -------- Confusion Matrices --------
cm_nb = confusion_matrix(y_test, nb_pred)
cm_svm = confusion_matrix(y_test, svm_pred)

# Plot NB confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, cmap="Blues",
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Plot SVM confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, cmap="Greens",
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------- Save models + metrics --------
if not os.path.exists("./models"):
    os.makedirs("./models")

pickle.dump(nb, open("./models/nb_model.pkl", "wb"))
pickle.dump(svm, open("./models/svm_model.pkl", "wb"))
pickle.dump(tfidf, open("./models/tfidf.pkl", "wb"))

metrics = {
    "nb_accuracy": nb_accuracy,
    "svm_accuracy": svm_accuracy,
    "nb_report": nb_report,
    "svm_report": svm_report,
    "cm_nb": cm_nb,
    "cm_svm": cm_svm,
}

pickle.dump(metrics, open("./models/metrics.pkl", "wb"))

print("✅ Models and metrics saved successfully!")