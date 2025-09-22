import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class DataPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """Làm sạch và chuẩn hóa text"""
        if not text:
            return ""
        
        # Loại bỏ HTML tags, URLs, special characters
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Chuyển về lowercase
        text = text.lower()
        
        # Tokenize và loại bỏ stopwords
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def process_user_data(self, users_file):
        """Xử lý dữ liệu user"""
        with open(users_file, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        processed_users = []
        for user in users_data:
            # Kết hợp tất cả text từ posts và comments
            all_text = ""
            
            # Xử lý posts
            for post in user.get('posts', []):
                title = post.get('title', '')
                text = post.get('text', '')
                all_text += f" {title} {text}"
            
            # Xử lý comments
            for comment in user.get('comments', []):
                comment_text = comment.get('text', '')
                all_text += f" {comment_text}"
            
            # Làm sạch text
            cleaned_text = self.clean_text(all_text)
            
            processed_users.append({
                'user_id': user['user_id'],
                'username': user['username'],
                'content': cleaned_text
            })
        
        return pd.DataFrame(processed_users)
    
    def process_ad_data(self, ads_file):
        """Xử lý dữ liệu quảng cáo"""
        with open(ads_file, 'r', encoding='utf-8') as f:
            ads_data = json.load(f)
        
        processed_ads = []
        for ad in ads_data:
            # Kết hợp description và target audience
            ad_text = f"{ad.get('ad_description', '')} {ad.get('target_audience', '')}"
            cleaned_text = self.clean_text(ad_text)
            
            processed_ads.append({
                'ad_id': ad['id'],
                'topic': ad.get('topic', ''),
                'content': cleaned_text,
                'target_audience': ad.get('target_audience', '')
            })
        
        return pd.DataFrame(processed_ads)
    
    def process_interaction_data(self, interactions_file):
        """Xử lý dữ liệu tương tác"""
        with open(interactions_file, 'r', encoding='utf-8') as f:
            interactions_data = json.load(f)
        
        return pd.DataFrame(interactions_data)