import pandas as pd 
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load text data
df = pd.read_csv('data_recom.csv')
text_data = df['shortdesc'].tolist()
text_data.append('example query')

# Create tokenizer  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
vocab_size = len(tokenizer.word_index) + 1

# Integer encode text
encoded_data = tokenizer.texts_to_sequences(text_data) 
padded_data = pad_sequences(encoded_data, padding='post')

# Create model
model = Sequential()
model.add(Embedding(vocab_size, 100)) 
model.add(LSTM(128))
model.add(Dense(100, activation='tanh'))

model.compile(loss='mse', optimizer='adam')
encoded_labels = np.zeros((len(padded_data), 100))
model.fit(padded_data, encoded_labels, epochs=10)

# Generate embeddings  
embeddings = model.predict(padded_data)
query_embedding = embeddings[-1]
similarity = np.inner(query_embedding, embeddings[:-1])
most_similar = similarity.argsort()[-5:][::-1]

most_similar_descs = df.iloc[most_similar]['prompt'].tolist()
print(most_similar_descs)