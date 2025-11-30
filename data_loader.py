import os
import tensorflow as tf

def load_dataset(directory):
    texts = []
    labels = []
    
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        if not os.path.exists(dir_name):
            print(f"Warning: Directory {dir_name} not found.")
            continue
            
        print(f"Loading {label_type} reviews from {dir_name}...")
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                fpath = os.path.join(dir_name, fname)
                with open(fpath, encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
                
    return texts, labels

def get_vectorization_layer(vocab_file, max_tokens=20000, output_sequence_length=200):
    """
    Creates and adapts a TextVectorization layer.
    If vocab_file is provided, it initializes with that vocabulary.
    """
    # Read vocabulary
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f.readlines()]
    
    # Limit vocab size if necessary (though imdb.vocab is usually full)
    if len(vocab) > max_tokens:
        vocab = vocab[:max_tokens]
        
    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode='int',
        output_sequence_length=output_sequence_length,
        vocabulary=vocab
    )
    
    return vectorize_layer
