# main.py

from src.data_loader import load_and_prepare_data
from src.model import train_model, evaluate_model
from src.active_learning import select_uncertain_samples, query_human, retrain_model
import numpy as np
from scipy import sparse

if __name__ == "__main__":
    # Load and prepare data
    X_train_small, y_train_small, X_pool, y_pool, X_test, y_test, vectorizer = load_and_prepare_data()

    # Train baseline model
    model = train_model(X_train_small, y_train_small)

    # Evaluate baseline model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Baseline Model Accuracy: {accuracy:.4f}")

    # Start Active Learning Loop
    for iteration in range(5):  # do 5 rounds
        print(f"\n=== Active Learning Iteration {iteration+1} ===")
        
        # Select uncertain samples
        query_idx = select_uncertain_samples(model, X_pool, n_instances=5)
        
        # Human annotates
        new_labels = query_human(X_pool, query_idx, vectorizer)
        
        # Filter out skipped samples
        labeled_idx = [i for i, label in enumerate(new_labels) if label is not None]
        new_labeled_samples = X_pool[query_idx[labeled_idx]]
        new_labels_filtered = np.array([new_labels[i] for i in labeled_idx])
        
        if len(new_labels_filtered) == 0:
            print("No new labels provided. Skipping iteration.")
            continue
        
        # Add new labeled data to train set
        X_train_small = sparse.vstack([X_train_small, new_labeled_samples])
        y_train_small = np.concatenate([y_train_small, new_labels_filtered])
        
        # Remove labeled samples from pool
        mask = np.ones(X_pool.shape[0], dtype=bool)
        mask[query_idx[labeled_idx]] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        # Retrain model
        model = retrain_model(model, X_train_small, y_train_small)
        
        # Re-evaluate model
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"New Model Accuracy: {accuracy:.4f}")
