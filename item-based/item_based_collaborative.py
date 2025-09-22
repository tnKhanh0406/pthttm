import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ItemBasedCollaborativeFiltering:
    def __init__(self, model_name="social_media_ad_cf"):
        self.model_name = model_name
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            lowercase=True
        )
        
        # Model components
        self.user_profiles = {}
        self.ad_profiles = {}
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.ad_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_ad = {}
        
        # Model state
        self.is_trained = False
        self.training_time = None
        self.model_stats = {}
        
    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Loại bỏ URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Loại bỏ email
        text = re.sub(r'\S+@\S+', '', text)
        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu cách
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Loại bỏ nhiều dấu cách
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_user_content(self, user_data):
        """Trích xuất nội dung từ posts và comments của user (KHÔNG dùng topic có sẵn)"""
        all_content = []
        
        # Xử lý posts
        posts = user_data.get('posts', [])
        for post in posts:
            title = post.get('title', '')
            text = post.get('text', '')
            
            # Kết hợp title và text, title có trọng số cao hơn
            if title:
                all_content.append(self.preprocess_text(title))
                all_content.append(self.preprocess_text(title))  # Lặp lại để tăng trọng số
            if text:
                all_content.append(self.preprocess_text(text))
        
        # Xử lý comments
        comments = user_data.get('comments', [])
        for comment in comments:
            comment_text = comment.get('text', '')
            if comment_text:
                all_content.append(self.preprocess_text(comment_text))
        
        # Kết hợp tất cả content
        user_content = ' '.join(filter(None, all_content))
        return user_content
    
    def extract_ad_content(self, ad_data):
        """Trích xuất nội dung từ quảng cáo"""
        description = ad_data.get('ad_description', '')
        topic = ad_data.get('topic', '')
        target_audience = ad_data.get('target_audience', '')
        
        # Kết hợp tất cả thông tin quảng cáo
        ad_content_parts = []
        if description:
            ad_content_parts.append(self.preprocess_text(description))
            ad_content_parts.append(self.preprocess_text(description))  # Tăng trọng số cho description
        if topic:
            ad_content_parts.append(self.preprocess_text(topic))
        if target_audience:
            ad_content_parts.append(self.preprocess_text(target_audience))
        
        ad_content = ' '.join(ad_content_parts)
        return ad_content
    
    def load_and_process_data(self, users_file, ads_file, train_interactions_file):
        """Load và xử lý dữ liệu từ file JSON"""
        print("Loading data files...")
        
        try:
            # Load users
            with open(users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            print(f"✓ Loaded {len(users_data)} users")
            
            # Load ads
            with open(ads_file, 'r', encoding='utf-8') as f:
                ads_data = json.load(f)
            print(f"✓ Loaded {len(ads_data)} ads")
            
            # Load interactions
            with open(train_interactions_file, 'r', encoding='utf-8') as f:
                train_interactions = json.load(f)
            print(f"✓ Loaded {len(train_interactions)} training interactions")
            
        except FileNotFoundError as e:
            print(f"✗ Error: File not found - {e}")
            return None, None, None
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON format - {e}")
            return None, None, None
        
        # Xử lý user content
        print("Processing user content...")
        user_contents = []
        user_ids = []
        valid_users = 0
        
        for user in users_data:
            username = user.get('username', '')
            if not username:
                continue
                
            user_content = self.extract_user_content(user)
            if user_content.strip():  # Chỉ thêm user có content
                user_contents.append(user_content)
                user_ids.append(username)
                valid_users += 1
        
        print(f"✓ Processed {valid_users} users with valid content")
        
        # Xử lý ad content
        print("Processing ad content...")
        ad_contents = []
        ad_ids = []
        
        for ad in ads_data:
            ad_id = ad.get('id', '')
            if not ad_id:
                continue
                
            ad_content = self.extract_ad_content(ad)
            ad_contents.append(ad_content)
            ad_ids.append(ad_id)
        
        print(f"✓ Processed {len(ad_ids)} ads")
        
        if not user_contents or not ad_contents:
            print("✗ Error: No valid content found")
            return None, None, None
        
        # Tạo TF-IDF features
        print("Creating TF-IDF features...")
        all_contents = user_contents + ad_contents
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_contents)
            print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
            
            # Tách features cho users và ads
            user_features = tfidf_matrix[:len(user_contents)]
            ad_features = tfidf_matrix[len(user_contents):]
            
            # Lưu profiles
            self.user_profiles = {user_ids[i]: user_features[i] for i in range(len(user_ids))}
            self.ad_profiles = {ad_ids[i]: ad_features[i] for i in range(len(ad_ids))}
            
        except Exception as e:
            print(f"✗ Error creating TF-IDF features: {e}")
            return None, None, None
        
        # Tạo user-item matrix
        print("Creating user-item interaction matrix...")
        self.create_user_item_matrix(train_interactions, user_ids, ad_ids)
        
        return users_data, ads_data, train_interactions
    
    def create_user_item_matrix(self, train_interactions, user_ids, ad_ids):
        """Tạo ma trận user-item từ training interactions"""
        # Tạo mapping indices
        self.user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
        self.ad_to_idx = {ad: idx for idx, ad in enumerate(ad_ids)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_ad = {idx: ad for ad, idx in self.ad_to_idx.items()}
        
        # Khởi tạo ma trận
        n_users = len(user_ids)
        n_ads = len(ad_ids)
        self.user_item_matrix = np.zeros((n_users, n_ads))
        
        # Điền dữ liệu interaction
        valid_interactions = 0
        for interaction in train_interactions:
            username = interaction.get('username', '')
            ad_id = interaction.get('ad_id', '')
            label = interaction.get('label', 0)
            
            if username in self.user_to_idx and ad_id in self.ad_to_idx:
                user_idx = self.user_to_idx[username]
                ad_idx = self.ad_to_idx[ad_id]
                self.user_item_matrix[user_idx, ad_idx] = label
                valid_interactions += 1
        
        # Tính thống kê
        total_possible = n_users * n_ads
        sparsity = 1.0 - (valid_interactions / total_possible)
        
        print(f"✓ User-item matrix: {self.user_item_matrix.shape}")
        print(f"✓ Valid interactions: {valid_interactions}/{len(train_interactions)}")
        print(f"✓ Matrix sparsity: {sparsity:.4f}")
        
        # Lưu stats
        self.model_stats.update({
            'n_users': n_users,
            'n_ads': n_ads,
            'n_interactions': valid_interactions,
            'matrix_sparsity': sparsity,
            'tfidf_features': self.user_item_matrix.shape[1] if hasattr(self, 'tfidf_vectorizer') else 0
        })
    
    def calculate_item_similarity(self):
        """Tính ma trận độ tương đồng giữa các items (ads)"""
        print("Calculating item-item similarity matrix...")
        
        # Sử dụng cosine similarity trên ma trận user-item transpose
        # Mỗi row là một ad, mỗi column là một user
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # Đặt diagonal = 0 để tránh self-similarity
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        print(f"✓ Item similarity matrix shape: {self.item_similarity_matrix.shape}")
        print(f"✓ Average similarity: {np.mean(self.item_similarity_matrix):.4f}")
    
    def predict_rating(self, username, ad_id, k=10):
        """Dự đoán rating cho user-ad pair sử dụng item-based CF"""
        if username not in self.user_to_idx or ad_id not in self.ad_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[username]
        target_ad_idx = self.ad_to_idx[ad_id]
        
        # Lấy similarity scores của target ad với tất cả ads khác
        similarities = self.item_similarity_matrix[target_ad_idx]
        
        # Lấy ratings của user cho tất cả ads
        user_ratings = self.user_item_matrix[user_idx]
        
        # Tìm top-k similar ads mà user đã có interaction
        rated_ad_indices = np.where(user_ratings > 0)[0]
        
        if len(rated_ad_indices) == 0:
            return 0.0
        
        # Lấy similarities và ratings cho các ads đã rated
        similar_similarities = similarities[rated_ad_indices]
        similar_ratings = user_ratings[rated_ad_indices]
        
        # Sắp xếp theo similarity và lấy top-k
        top_k_indices = np.argsort(similar_similarities)[::-1][:k]
        
        if len(top_k_indices) == 0:
            return 0.0
        
        # Tính weighted average
        numerator = 0.0
        denominator = 0.0
        
        for idx in top_k_indices:
            similarity = similar_similarities[idx]
            rating = similar_ratings[idx]
            
            if similarity > 0:
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            # Fallback: trả về average rating của user
            user_avg = np.mean(user_ratings[user_ratings > 0])
            return user_avg if not np.isnan(user_avg) else 0.0
        
        return numerator / denominator
    
    def train(self, users_file, ads_file, train_interactions_file):
        """Train model với dữ liệu có sẵn"""
        print("="*60)
        print("TRAINING ITEM-BASED COLLABORATIVE FILTERING MODEL")
        print("="*60)
        
        start_time = datetime.now()
        
        # Load và xử lý dữ liệu
        users_data, ads_data, train_interactions = self.load_and_process_data(
            users_file, ads_file, train_interactions_file
        )
        
        if users_data is None:
            print("✗ Training failed: Could not load data")
            return False
        
        # Tính item similarity
        self.calculate_item_similarity()
        
        # Hoàn thành training
        self.is_trained = True
        self.training_time = datetime.now()
        
        total_time = (self.training_time - start_time).total_seconds()
        self.model_stats['training_time'] = total_time
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Total training time: {total_time:.2f} seconds")
        
        # Tự động lưu model
        self.save_model()
        
        return True
    
    def evaluate(self, test_interactions_file, threshold=0.5, k=10):
        """Đánh giá model trên test set"""
        if not self.is_trained:
            print("✗ Error: Model must be trained first!")
            return None
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Load test data
        try:
            with open(test_interactions_file, 'r', encoding='utf-8') as f:
                test_interactions = json.load(f)
            print(f"✓ Loaded {len(test_interactions)} test interactions")
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
            return None
        
        # Dự đoán
        predictions = []
        actual_labels = []
        prediction_scores = []
        
        print("Making predictions...")
        eval_start = datetime.now()
        
        for i, interaction in enumerate(test_interactions):
            username = interaction.get('username', '')
            ad_id = interaction.get('ad_id', '')
            actual_label = interaction.get('label', 0)
            
            # Dự đoán score
            predicted_score = self.predict_rating(username, ad_id, k=k)
            predicted_label = 1 if predicted_score > threshold else 0
            
            predictions.append(predicted_label)
            actual_labels.append(actual_label)
            prediction_scores.append(predicted_score)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(test_interactions)} samples")
        
        eval_time = (datetime.now() - eval_start).total_seconds()
        
        # Tính metrics
        predictions = np.array(predictions)
        actual_labels = np.array(actual_labels)
        
        # Confusion Matrix
        tp = np.sum((predictions == 1) & (actual_labels == 1))
        tn = np.sum((predictions == 0) & (actual_labels == 0))
        fp = np.sum((predictions == 1) & (actual_labels == 0))
        fn = np.sum((predictions == 0) & (actual_labels == 1))
        
        total_samples = len(predictions)
        correct_predictions = tp + tn
        incorrect_predictions = fp + fn
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # In kết quả
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"Total test samples: {total_samples}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {incorrect_predictions}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\nConfusion Matrix:")
        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        
        print(f"\nPerformance Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        # Thống kê prediction scores
        # if prediction_scores:
        #     scores_array = np.array(prediction_scores)
        #     print(f"\nPrediction Score Statistics:")
        #     print(f"Mean: {np.mean(scores_array):.4f}")
        #     print(f"Std: {np.std(scores_array):.4f}")
        #     print(f"Min: {np.min(scores_array):.4f}")
        #     print(f"Max: {np.max(scores_array):.4f}")
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'eval_time': eval_time,
            'prediction_scores': prediction_scores
        }
    
    def save_model(self, filepath=None):
        """Lưu trained model"""
        if not self.is_trained:
            print("Warning: Model has not been trained yet!")
            return False
        
        if filepath is None:
            # Tạo models directory
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/{self.model_name}_{timestamp}.pkl"
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'user_profiles': self.user_profiles,
            'ad_profiles': self.ad_profiles,
            'item_similarity_matrix': self.item_similarity_matrix,
            'user_item_matrix': self.user_item_matrix,
            'user_to_idx': self.user_to_idx,
            'ad_to_idx': self.ad_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_ad': self.idx_to_ad,
            'model_stats': self.model_stats,
            'training_time': self.training_time,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_kb = os.path.getsize(filepath) / 1024
            print(f"✓ Model saved: {filepath} ({file_size_kb:.2f} KB)")
            return True
            
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load saved model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            for key, value in model_data.items():
                setattr(self, key, value)
            
            print(f"✓ Model loaded successfully from: {filepath}")
            print(f"  Model trained on: {self.training_time}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def get_model_info(self):
        """Thông tin về model"""
        if not self.is_trained:
            return "Model has not been trained yet."
        
        info = f"""
Model Information:
- Name: {self.model_name}
- Training Time: {self.training_time}
- Users: {self.model_stats.get('n_users', 'N/A')}
- Ads: {self.model_stats.get('n_ads', 'N/A')}
- Interactions: {self.model_stats.get('n_interactions', 'N/A')}
- Matrix Sparsity: {self.model_stats.get('matrix_sparsity', 'N/A'):.4f}
- Training Duration: {self.model_stats.get('training_time', 'N/A'):.2f}s
"""
        return info