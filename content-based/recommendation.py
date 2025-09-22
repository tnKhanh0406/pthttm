import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # Thêm import này

class RecommendationEngine:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
    
    def recommend_ads_for_user(self, user_idx, ads_df, users_df, top_k=5):
        """Gợi ý top-k quảng cáo cho 1 user"""
        
        recommendations = []
        
        for ad_idx in range(len(ads_df)):
            # Tính similarity
            similarity = self.feature_extractor.get_user_ad_similarity(user_idx, ad_idx)
            
            # Tạo feature vector
            feature_vector = np.array([[
                similarity,
                len(users_df.loc[user_idx, 'content'].split()),
                len(ads_df.loc[ad_idx, 'content'].split())
            ]])
            
            # Predict probability
            prob = self.model.predict_proba(feature_vector)[0][1]
            
            recommendations.append({
                'ad_id': ads_df.loc[ad_idx, 'ad_id'],
                'similarity': similarity,
                'probability': prob,
                'topic': ads_df.loc[ad_idx, 'topic'],
                'target_audience': ads_df.loc[ad_idx, 'target_audience']
            })
        
        # Sắp xếp theo probability
        recommendations.sort(key=lambda x: x['probability'], reverse=True)
        
        return recommendations[:top_k]
    
    def recommend_for_new_user(self, user_content, ads_df, top_k=5):
        """Gợi ý cho user mới dựa trên content"""
        
        # Transform user content
        user_vector = self.feature_extractor.tfidf_vectorizer.transform([user_content])
        
        recommendations = []
        
        for ad_idx in range(len(ads_df)):
            # Tính similarity
            ad_vector = self.feature_extractor.ad_features[ad_idx]
            similarity = cosine_similarity(user_vector, ad_vector)[0][0]  # Bây giờ đã có import
            
            # Tạo feature vector
            feature_vector = np.array([[
                similarity,
                len(user_content.split()),
                len(ads_df.loc[ad_idx, 'content'].split())
            ]])
            
            # Predict probability
            prob = self.model.predict_proba(feature_vector)[0][1]
            
            recommendations.append({
                'ad_id': ads_df.loc[ad_idx, 'ad_id'],
                'similarity': similarity,
                'probability': prob,
                'topic': ads_df.loc[ad_idx, 'topic']
            })
        
        recommendations.sort(key=lambda x: x['probability'], reverse=True)
        return recommendations[:top_k]
    
    def evaluate_recommendations(self, test_interactions, users_df, ads_df):
        """Đánh giá chất lượng recommendation"""
        
        correct_predictions = 0
        total_predictions = 0
        
        for _, interaction in test_interactions.iterrows():
            username = interaction['username']
            ad_id = interaction['ad_id']
            true_label = interaction['label']
            
            # Tìm user index
            user_idx = users_df[users_df['username'] == username].index
            if len(user_idx) == 0:
                continue
            user_idx = user_idx[0]
            
            # Gợi ý top-5 ads cho user này
            recommendations = self.recommend_ads_for_user(user_idx, ads_df, users_df, top_k=5)
            
            # Kiểm tra xem ad có trong top-5 không
            recommended_ads = [rec['ad_id'] for rec in recommendations]
            predicted_label = 1 if ad_id in recommended_ads else 0
            
            if predicted_label == true_label:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Recommendation Accuracy: {accuracy:.4f}")
        
        return accuracy