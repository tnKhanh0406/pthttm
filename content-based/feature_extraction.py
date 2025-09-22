import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams và bigrams
            min_df=2,  # Bỏ qua từ xuất hiện ít hơn 2 lần
            max_df=0.95  # Bỏ qua từ xuất hiện quá nhiều
        )
        self.user_features = None
        self.ad_features = None
    
    def fit_transform_features(self, users_df, ads_df):
        """Tính toán TF-IDF features cho users và ads"""
        
        # Kết hợp tất cả text để fit vectorizer
        all_text = list(users_df['content']) + list(ads_df['content'])
        self.tfidf_vectorizer.fit(all_text)
        
        # Transform user content
        self.user_features = self.tfidf_vectorizer.transform(users_df['content'])
        
        # Transform ad content  
        self.ad_features = self.tfidf_vectorizer.transform(ads_df['content'])
        
        return self.user_features, self.ad_features
    
    def compute_similarity_features(self, users_df, ads_df, interactions_df):
        """Tính toán features dựa trên độ tương đồng"""
        
        features = []
        labels = []
        
        for _, interaction in interactions_df.iterrows():
            username = interaction['username']
            ad_id = interaction['ad_id']
            label = interaction['label']
            
            # Tìm user index
            user_idx = users_df[users_df['username'] == username].index
            if len(user_idx) == 0:
                continue
            user_idx = user_idx[0]
            
            # Tìm ad index
            ad_idx = ads_df[ads_df['ad_id'] == ad_id].index
            if len(ad_idx) == 0:
                continue
            ad_idx = ad_idx[0]
            
            # Tính cosine similarity
            user_vector = self.user_features[user_idx].toarray()
            ad_vector = self.ad_features[ad_idx].toarray()
            
            similarity = cosine_similarity(user_vector, ad_vector)[0][0]
            
            # Thêm features khác
            feature_vector = [
                similarity,  # Cosine similarity
                len(users_df.loc[user_idx, 'content'].split()),  # Độ dài content user
                len(ads_df.loc[ad_idx, 'content'].split()),  # Độ dài content ad
            ]
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def get_user_ad_similarity(self, user_idx, ad_idx):
        """Tính similarity giữa 1 user và 1 ad cụ thể"""
        user_vector = self.user_features[user_idx].toarray()
        ad_vector = self.ad_features[ad_idx].toarray()
        
        return cosine_similarity(user_vector, ad_vector)[0][0]