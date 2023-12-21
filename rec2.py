import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("data_recom.csv")

# Load Universal Sentence Encoder from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Create embeddings for all prompts
prompt_embeddings = model(data['shortdesc']).numpy()

def recommend_prompt(input_prompt, num_recommendations=5):
    # Embed the input prompt
    input_embedding = model([input_prompt]).numpy()

    # Calculate cosine similarity between input prompt and all prompts
    similarities = cosine_similarity(prompt_embeddings, input_embedding)

    # Get indices of top recommendations
    recommended_indices = similarities.flatten().argsort()[-num_recommendations-1:-1][::-1]

    # Get recommended prompts
    recommended_prompts = data.loc[recommended_indices, 'prompt'].tolist()

    return recommended_prompts

# Example usage
user_input_prompt = "memulai proses belajar bermain berbagai alat musik dengan pemahaman mendalam"
recommended_prompts = recommend_prompt(user_input_prompt)

print(f"Recommended prompts based on '{user_input_prompt}':")
for i, prompt in enumerate(recommended_prompts, 1):
    print(f"{i}. {prompt}")