from data_preprocessing import DataPreprocessor
from feature_extraction import FeatureExtractor
from model import RecommendationModel
from recommendation import RecommendationEngine
import os

def main():
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # 1. Data Preprocessing
    print("=== Data Preprocessing ===")
    preprocessor = DataPreprocessor()
    
    users_df = preprocessor.process_user_data('../../data/reddit_dataset_sentiment.json')
    ads_df = preprocessor.process_ad_data('../../data/ads_dataset_14_unique.json')
    interactions_df = preprocessor.process_interaction_data('../../data/user_ad_interactions_train.json')

    print(f"Số lượng users: {len(users_df)}")
    print(f"Số lượng ads: {len(ads_df)}")
    print(f"Số lượng interactions: {len(interactions_df)}")
    
    # 2. Feature Extraction
    print("\n=== Feature Extraction ===")
    feature_extractor = FeatureExtractor()
    
    user_features, ad_features = feature_extractor.fit_transform_features(users_df, ads_df)
    X, y = feature_extractor.compute_similarity_features(users_df, ads_df, interactions_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Positive samples: {sum(y)} / {len(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    # 3. Model Training
    print("\n=== Model Training ===")
    model = RecommendationModel(model_type='random_forest')
    
    # Train và evaluate
    results = model.train(X, y)
    
    # Cross validation
    cv_scores = model.cross_validate(X, y)
    
    # 4. Recommendation Engine
    print("\n=== Recommendation Engine ===")
    engine = RecommendationEngine(model, feature_extractor)
    
    # Test với user đầu tiên
    if len(users_df) > 0:
        user_idx = 0
        username = users_df.loc[user_idx, 'username']
        print(f"\nGợi ý quảng cáo cho user: {username}")
        
        recommendations = engine.recommend_ads_for_user(user_idx, ads_df, users_df, top_k=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Ad ID: {rec['ad_id']}")
            print(f"   Topic: {rec['topic']}")
            print(f"   Similarity: {rec['similarity']:.4f}")
            print(f"   Probability: {rec['probability']:.4f}")
            print()
    
    # 5. Save model
    model.save_model('models/recommendation_model.joblib')
    print("Model đã được lưu tại: models/recommendation_model.joblib")

if __name__ == "__main__":
    main()
