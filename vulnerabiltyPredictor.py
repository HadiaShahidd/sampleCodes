import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gensim
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load the dataset
data = pd.read_json('train.jsonl', lines=True) 
texts = data['code'].values  # Extract source code snippets (features)
labels = data['target'].values  # Extract corresponding labels (target)

# Step 2: Tokenize the source code
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Initialize tokenizer with max vocab size of 5000 and OOV token
tokenizer.fit_on_texts(texts)  # Fit tokenizer on the source code texts
sequences = tokenizer.texts_to_sequences(texts)  # Convert texts to sequences of integers
padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)  # Pad sequences for uniform length (100)

# Step 3: Word2Vec Embedding
# Train a Word2Vec model using the source code texts (split by whitespace)
word2vec_model = Word2Vec([text.split() for text in texts], vector_size=100, window=5, min_count=1)

# Set up embedding layer input size
vocab_size = len(tokenizer.word_index) + 1  # Get vocabulary size from the tokenizer (plus 1 for padding/OOV tokens)
embedding_dim = 100  

# Step 4: Create an embedding matrix that matches the tokenizer's vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim)) 

# Populate the embedding matrix with Word2Vec embeddings for each word in the tokenizer's vocabulary
for word, index in tokenizer.word_index.items():
    if word in word2vec_model.wv.key_to_index:  # Check if the word exists in Word2Vec model
        embedding_matrix[index] = word2vec_model.wv[word]  # Add Word2Vec embedding for the word
    else:
        embedding_matrix[index] = np.zeros(embedding_dim)  # Optionally assign a zero vector for unknown words

# Step 5: Build the Deep Neural Network
model = Sequential()  # Initialize the sequential model
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix], 
                    input_length=100,
                    trainable=False))  # Freeze the embedding layer
model.add(LSTM(128, return_sequences=False))  # Add an LSTM layer with 128 units
model.add(Dense(64, activation='relu'))  # Add a fully connected (dense) layer with 64 units and ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Add output layer with sigmoid activation for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model with Adam optimizer and binary cross-entropy loss

# Step 6: Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)  # Split data into training and test sets
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)  # Train the model for 10 epochs

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test) 
print(f"Test Accuracy: {accuracy * 100:.2f}%")  

# Example of predicting vulnerability in new code
new_code_sample = "int main() { int a = 0; while(a != 10) { a++; } return 0; }"  # Define a new code sample
new_sequence = tokenizer.texts_to_sequences([new_code_sample])  # Tokenize 
new_padded_sequence = pad_sequences(new_sequence, maxlen=100)  
prediction = model.predict(new_padded_sequence)  # Make a prediction
vulnerable = prediction[0][0] > 0.5  # Check if the model predicts the code as vulnerable
print(f"Vulnerability Prediction: {'Yes' if vulnerable else 'No'}")  # Print the prediction result
