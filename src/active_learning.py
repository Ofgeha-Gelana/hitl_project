# src/active_learning.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

def select_uncertain_samples(model, X_pool, n_instances=10):
    """
    Select the most uncertain samples (lowest confidence).
    """
    probs = model.predict_proba(X_pool)
    uncertainties = 1 - np.max(probs, axis=1)  # least confident
    query_idx = np.argsort(uncertainties)[-n_instances:]  # top uncertain samples
    return query_idx

def query_human(X_pool, query_idx, vectorizer):
    """
    Simulate a human annotator via CLI (Command Line).
    Shows the uncertain texts and asks for correct label.
    """
    X_selected = X_pool[query_idx]
    new_texts = vectorizer.inverse_transform(X_selected)
    
    new_labels = []
    
    for idx, text_tokens in enumerate(new_texts):
        text = ' '.join(text_tokens)
        print("\n=== Review Text ===")
        print(text)
        print("===================")
        label = input("Label (0 = negative, 1 = positive, s = skip): ")
        
        if label == 's':
            new_labels.append(None)  # if skipped
        else:
            new_labels.append(int(label))
    
    return new_labels

def retrain_model(model, X_train, y_train):
    """
    Retrains the model on the updated labeled set.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
