import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

# Sample sentences
sentences = [
    "I love this movie!",
    "This movie is fantastic.",
    "This movie is amazing",
    "I adore this movie.",
    "This movie is excellent.",
    "This movie is top-notch.",
    "This movie is brilliant.",
    "This movie is superb.",
    "This movie is a masterpiece.",
    "This movie is fantastic.",
    "This movie is amazing.",
    "This movie is great.",
    "This movie is wonderful.",
    "This movie is incredible.",
    "This movie is perfect.",
    "This movie is awesome.",
    "This movie is outstanding.",
    "This movie is terrific.",
    "This movie is phenomenal.",
    "This movie is first-rate.",
    "This movie is terrible.",
    "This movie is awful.",
    "I hate this movie.",
    "This movie is dreadful.",
    "This movie is garbage.",
    "This movie is lousy.",
    "This movie is horrendous.",
    "This movie is atrocious.",
    "This movie is rubbish.",
    "This movie is mediocre.",
    "This movie is bad.",
    "This movie is subpar.",
    "This movie is poor.",
    "This movie is dreadful.",
    "This movie is trash.",
    "This movie is horrible.",
    "This movie is appalling.",
    "This movie is pathetic.",
    "This movie is disappointing.",
    "This movie is unsatisfactory."
]



# Labels for the sentences (1 for positive sentiment, 0 for negative sentiment)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# Parameters for tokenization and padding
vocab_size = 1000
max_length = 20
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert sentences to sequences and pad them
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=100, batch_size=4)

# Evaluate model performance on training data
loss, accuracy = model.evaluate(padded_sequences, labels)
print("Training Accuracy:", accuracy)

# Test the model with new sentences
test_sentences = [
    "I bad this film!",
    " amazing.",
    "great",
    "nasty"
    
]

# Convert test sentences to sequences and pad them
test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Predict sentiment for test sentences
predictions = model.predict(padded_test_sequences)

for i in range(len(test_sentences)):
    print(f"Sentence: {test_sentences[i]} - Sentiment Probability: {predictions[i]} - Sentiment: {'Positive' if predictions[i] > 0.5 else 'Negative'}")
