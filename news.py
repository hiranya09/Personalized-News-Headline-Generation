import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Dot, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Parameters
embedding_dim = 256
gru_units = 512
vocab_size = 10000  # Adjust based on your dataset
max_sequence_length = 500  # Adjust according to article length

# Sample Data (Replace with real dataset)
articles = ["Sample text for the encoder input.", "Another example of encoder input."]
headlines = ["Generated headline for decoder input.", "Another headline example."]

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(articles + headlines)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
encoder_input_sequences = tokenizer.texts_to_sequences(articles)
decoder_input_sequences = tokenizer.texts_to_sequences(headlines)

# Pad sequences
X_train_encoder = pad_sequences(encoder_input_sequences, maxlen=max_sequence_length, padding='post')
X_train_decoder = pad_sequences(decoder_input_sequences, maxlen=max_sequence_length, padding='post')

# Target sequences (shifted decoder input)
y_train = X_train_decoder[:, 1:]
y_train = pad_sequences(y_train, maxlen=max_sequence_length, padding='post')

# Encoder Model
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
encoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='encoder_gru')
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)

# Decoder Model
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
decoder_gru = GRU(gru_units, return_sequences=True, return_state=True, name='decoder_gru')
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)

# Attention Mechanism
attention_scores = Dot(axes=[2, 2], name='attention_scores')([decoder_outputs, encoder_outputs])
attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
context_vector = Dot(axes=[2, 1], name='context_vector')([attention_weights, encoder_outputs])

# Concatenate context vector with decoder outputs
concat_output = Concatenate(name='concat_layer')([decoder_outputs, context_vector])

# Dense layer for word prediction
decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(concat_output)

# Final Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Summary
print(model.summary())

# Train Model (Dummy Training)
model.fit([X_train_encoder, X_train_decoder], y_train, epochs=1, batch_size=2)

# Save model in Keras format
model.save("news_model.keras")

print("Model and tokenizer saved successfully.")
