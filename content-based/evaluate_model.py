import joblib
import json
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor

def quick_test(test_file, users_file, ads_file):
    """Test nhanh - chỉ tính accuracy"""
    print("Loading model...")
    model = joblib.load('../src/models/recommendation_model.joblib')
    
    print("Loading test data...")
    preprocessor = DataPreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Load data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    users_df = preprocessor.process_user_data(users_file)
    ads_df = preprocessor.process_ad_data(ads_file)
    test_df = pd.DataFrame(test_data)
    
    # Extract features
    user_features, ad_features = feature_extractor.fit_transform_features(users_df, ads_df)
    X_test, y_test = feature_extractor.compute_similarity_features(users_df, ads_df, test_df)
    
    # Predict
    print("Predicting...")
    y_pred = model.predict(X_test)
    
        # Predict
    print("Predicting...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy pre 71,24 rec = 24,51
    correct = np.sum(y_pred == y_test)
    total = len(y_test)
    accuracy = correct / total

        # Confusion matrix components
    TN = np.sum((y_test == 1) & (y_pred == 1))
    TP = np.sum((y_test == 0) & (y_pred == 0))
    FN = np.sum((y_test == 0) & (y_pred == 1))
    FP = np.sum((y_test == 1) & (y_pred == 0))

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== RESULTS ===")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Wrong predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n=== CONFUSION MATRIX ===")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    print("\n=== METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


    
    return accuracy

if __name__ == "__main__":
    accuracy = quick_test(
        '../../data/user_ad_interactions_test.json',
        '../../data/reddit_dataset_sentiment.json', 
        '../../data/ads_dataset_14_unique.json'
    )