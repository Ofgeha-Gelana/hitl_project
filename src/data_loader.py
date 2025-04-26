# src/data_loader.py

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def load_and_prepare_data():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    # Get train and test sets
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # Shuffle train data to mix labels properly
    print("Shuffling training data...")
    df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Subset smaller train and test for faster training
    train_texts = df_train['text'][:2000]
    train_labels = df_train['label'][:2000]
    test_texts = test_texts[:1000]
    test_labels = test_labels[:1000]

    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    y_train = train_labels
    y_test = test_labels

    # Split into small labeled set + pool
    print("Splitting small training set and pool...")
    X_train_small, X_pool, y_train_small, y_pool = train_test_split(
        X_train, y_train, train_size=100, stratify=y_train, random_state=42
    )

    return X_train_small, y_train_small, X_pool, y_pool, X_test, y_test, vectorizer
