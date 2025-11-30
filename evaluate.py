import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_dataset, get_vectorization_layer

# Configuration
VOCAB_FILE = os.path.join('aclImdb', 'imdb.vocab')
TEST_DIR = os.path.join('aclImdb', 'test')
MAX_TOKENS = 10000
MAX_LENGTH = 150
BATCH_SIZE = 32

def main():
    # 1. Load Data
    print("Loading test data...")
    test_texts, test_labels = load_dataset(TEST_DIR)
    test_labels = np.array(test_labels)

    # 2. Load Model
    print("Loading best model...")
    try:
        model = tf.keras.models.load_model('best_model.keras')
    except:
        print("Error: 'best_model.keras' not found. Run train.py first.")
        return

    # 3. Prepare Vectorization
    print("Preparing vectorization layer...")
    vectorize_layer = get_vectorization_layer(VOCAB_FILE, MAX_TOKENS, MAX_LENGTH)
    
    # 4. Vectorize Data
    print("Vectorizing test data...")
    with tf.device('/CPU:0'):
        test_texts_vec = vectorize_layer(np.array(test_texts))

    # 5. Predict
    print("Predicting...")
    # Predict returns probabilities, we convert to 0 or 1
    y_pred_probs = model.predict(test_texts_vec)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # 6. Calculate Metrics
    acc = accuracy_score(test_labels, y_pred)
    prec = precision_score(test_labels, y_pred)
    rec = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 7. Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 8. Show Examples (Best and Worst)
    # Correct predictions
    correct_indices = np.where(y_pred == test_labels)[0]
    incorrect_indices = np.where(y_pred != test_labels)[0]
    
    print("\n--- 5 Correctly Classified Examples ---")
    for i in correct_indices[:5]:
        sentiment = "Positive" if test_labels[i] == 1 else "Negative"
        print(f"[{sentiment}] {test_texts[i][:100]}...")

    print("\n--- 5 Incorrectly Classified Examples ---")
    for i in incorrect_indices[:5]:
        actual = "Positive" if test_labels[i] == 1 else "Negative"
        pred = "Positive" if y_pred[i] == 1 else "Negative"
        print(f"[Actual: {actual}, Pred: {pred}] {test_texts[i][:100]}...")

if __name__ == "__main__":
    main()
