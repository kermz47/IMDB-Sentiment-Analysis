import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(vocab_size, embedding_dim, rnn_units, max_length):
    model = models.Sequential([
        # Input layer to define shape explicitly
        layers.Input(shape=(max_length,)),
        
        # Embedding layer
        # mask_zero=True tells the LSTM to ignore padding (0s)
        layers.Embedding(input_dim=vocab_size + 2, 
                         output_dim=embedding_dim, 
                         mask_zero=True),
        
        # LSTM Layer
        layers.LSTM(rnn_units),
        
        # Output Layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def build_gru_model(vocab_size, embedding_dim, rnn_units, max_length):
    model = models.Sequential([
        # Input layer to define shape explicitly
        layers.Input(shape=(max_length,)),
        
        # Embedding layer
        layers.Embedding(input_dim=vocab_size + 2, 
                         output_dim=embedding_dim, 
                         mask_zero=True),
        
        # GRU Layer
        layers.GRU(rnn_units),
        
        # Output Layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model