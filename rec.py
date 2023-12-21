# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Membaca dataset
data = pd.read_csv('data_recom.csv')

df = pd.DataFrame(data)

# Memisahkan data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(df['prompt'], df['jenis_kegiatan'], test_size=0.2, random_state=42)

# Membuat TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Mengubah teks menjadi matriks TF-IDF
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Membuat model Naive Bayes
model = MultinomialNB()

# Melatih model
model.fit(X_train_tfidf, y_train)

# Memprediksi jenis_kegiatan untuk testing set
y_pred = model.predict(X_test_tfidf)

# Evaluasi performa model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Contoh penggunaan model untuk merekomendasikan prompt
def recommend_prompt(input_prompt):
    input_tfidf = tfidf_vectorizer.transform([input_prompt])
    predicted_category = model.predict(input_tfidf)[0]
    recommended_prompts = df[df['theme'] == predicted_category]['prompt']
    return recommended_prompts

# Contoh penggunaan model untuk prompt tertentu
input_prompt = "cara menjadi data analyst"
recommended_prompts = recommend_prompt(input_prompt)
print(f"Rekomendasi prompt untuk kategori '{model.predict(tfidf_vectorizer.transform([input_prompt]))[0]}':\n", recommended_prompts)