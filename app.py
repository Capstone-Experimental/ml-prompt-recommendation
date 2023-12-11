from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

app = Flask(__name__)

# Load the dataset
data_recom = pd.read_csv('data_recom.csv')

# Load TinyBERT model and tokenizer only once
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode all prompts in the dataset during initialization
all_prompt_embeddings = np.vstack([model(torch.tensor([tokenizer.encode(prompt, max_length=128, truncation=True, padding='max_length')]))[0].squeeze().detach().numpy().reshape(1, -1) for prompt in data_recom['prompt']])

def encode_text(text):
    """Encode text using TinyBERT tokenizer and model."""
    encoding = tokenizer.encode(text, max_length=128, truncation=True, padding='max_length')
    # Use squeeze to remove the singleton dimension and reshape to (1, -1)
    return model(torch.tensor([encoding]))[0].squeeze().detach().numpy().reshape(1, -1)

@app.route('/recommend', methods=['POST'])
def recommendation_route():
    if request.method == 'POST':
        try:
            data = request.json
            input_prompt = data.get('prompt', '')  # Use get to handle missing key

            # Encode the input prompt
            prompt_embedding = encode_text(input_prompt)

            # Calculate cosine similarity in parallel
            similarities = cosine_similarity(prompt_embedding, all_prompt_embeddings)

            # Get the indices of the top 5 most similar prompts
            top_indices = np.argsort(similarities[0])[::-1][:5]  # Descending order, limit to 5 prompts

            # Get the top 5 most similar prompts and their corresponding similarities
            top_prompts = list(data_recom.loc[top_indices, 'prompt'])
            top_similarities = list(similarities[0, top_indices].astype(float))

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success",
                },
                "data": {
                    "recommended_prompts": top_prompts,
                    "similarities": top_similarities,
                }
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
    app.run(debug=True)