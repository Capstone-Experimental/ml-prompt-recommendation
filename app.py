from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data_recom = pd.read_csv('dataset/data_recom.csv')
corpus = data_recom['prompt'].tolist()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

@app.route('/api/trigger', methods=['GET'])
def trigger():
    return jsonify({"message" : "trigger"})

@app.route('/api/recommendation', methods=['POST'])
def recommendation_route():
    if request.method == 'POST':
        try:
            data = request.json
            input_prompt = data.get('prompt', '')

            input_tfidf = tfidf_vectorizer.transform([input_prompt])

            similarities = cosine_similarity(input_tfidf, tfidf_matrix)

            top_indices = similarities.argsort(axis=1)[:, -5:][0][::-1]

            top_prompts = list(data_recom.loc[top_indices, 'prompt'])
            # top_similarities = list(similarities[0, top_indices].astype(float))

            return jsonify({
                "recommendations": top_prompts,
            })

        except Exception as e:
            return jsonify({
                "status": {
                    "code": 500,
                    "message": "Internal Server Error",
                },
                "data": str(e),
            }), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
