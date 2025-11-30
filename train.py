import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_dataset, get_vectorization_layer
from models import build_lstm_model, build_gru_model

# --- OPTIMIZED HYPERPARAMETERS FOR SPEED ---
VOCAB_SIZE = 10000    
MAX_LENGTH = 150      
EMBEDDING_DIM = 64    
RNN_UNITS = 32        
BATCH_SIZE = 64       
EPOCHS = 3            
# -------------------------------------------

def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss')
    plt.savefig(f'{model_name}_history.png')
    plt.close()

def main():
    # 1. Load Data
    print("Loading data...")
    train_dir = os.path.join('aclImdb', 'train')
    test_dir = os.path.join('aclImdb', 'test')
    
    train_texts, train_labels = load_dataset(train_dir)
    test_texts, test_labels = load_dataset(test_dir)

    # 2. Use Full Dataset
    print(f"Training on full dataset: {len(train_texts)} samples.")
    
    # --- FIX: SHUFFLE DATA ---
    print("Shuffling data...")
    # Shuffle training data
    train_indices = np.arange(len(train_texts))
    np.random.shuffle(train_indices)
    train_texts = np.array(train_texts)[train_indices]
    train_labels = np.array(train_labels)[train_indices]
    
    # Shuffle test data
    test_indices = np.arange(len(test_texts))
    np.random.shuffle(test_indices)
    test_texts = np.array(test_texts)[test_indices]
    test_labels = np.array(test_labels)[test_indices]
    # -----------------------------------

    # 3. Prepare Vectorization
    print("Preparing vectorization layer...")
    vocab_file = os.path.join('aclImdb', 'imdb.vocab')
    vectorize_layer = get_vectorization_layer(vocab_file, VOCAB_SIZE, MAX_LENGTH)
    
    # 4. Vectorize datasets (CORRECCIÓN AQUÍ)
    print("Vectorizing data (converting text to numbers)...")
    # Convertimos el texto a números ANTES de crear el dataset
    with tf.device('/CPU:0'):
        train_texts_vec = vectorize_layer(np.array(train_texts))
        test_texts_vec = vectorize_layer(np.array(test_texts))

    # Create datasets with the VECTORIZED text
    train_ds = tf.data.Dataset.from_tensor_slices((train_texts_vec, train_labels))
    train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_texts_vec, test_labels))
    test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    # 5. Train LSTM
    print("\n--- Training LSTM ---")
    lstm_model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, MAX_LENGTH)
    lstm_history = lstm_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )
    plot_history(lstm_history, 'LSTM')
    
    # 6. Train GRU
    print("\n--- Training GRU ---")
    gru_model = build_gru_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, MAX_LENGTH)
    gru_history = gru_model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )
    plot_history(gru_history, 'GRU')

    # 7. Compare and Save
    lstm_acc = lstm_history.history['val_accuracy'][-1]
    gru_acc = gru_history.history['val_accuracy'][-1]
    
    print(f"\nResults:\nLSTM Accuracy: {lstm_acc:.4f}\nGRU Accuracy: {gru_acc:.4f}")
    
    results_df = pd.DataFrame({
        'Model': ['LSTM', 'GRU'],
        'Accuracy': [lstm_acc, gru_acc]
    })
    results_df.to_csv('model_comparison.csv', index=False)
    print("Comparison saved to model_comparison.csv")

    if lstm_acc > gru_acc:
        print("Saving LSTM model as best_model.keras")
        lstm_model.save('best_model.keras')
    else:
        print("Saving GRU model as best_model.keras")
        gru_model.save('best_model.keras')

if __name__ == "__main__":
    main()