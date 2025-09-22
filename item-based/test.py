#!/usr/bin/env python3
"""
Script đơn giản để test model với 5 input content và xem top 3 quảng cáo được đề xuất
"""

import os
import json
import glob
import numpy as np
from item_based_collaborative import ItemBasedCollaborativeFiltering

def load_latest_model():
    """Load model mới nhất từ thư mục models/"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("❌ Thư mục models/ không tồn tại")
        return None
    
    # Tìm tất cả file .pkl trong models/
    model_files = glob.glob(os.path.join(models_dir, "*.pkl"))
    
    if not model_files:
        print("❌ Không tìm thấy model nào trong thư mục models/")
        print("Hãy train model trước: python main.py --train")
        return None
    
    # Lấy file mới nhất
    latest_file = max(model_files, key=os.path.getmtime)
    print(f"📂 Loading model: {os.path.basename(latest_file)}")
    
    # Load model
    model = ItemBasedCollaborativeFiltering()
    if model.load_model(latest_file):
        print("✅ Model loaded successfully!")
        return model
    else:
        print("❌ Failed to load model")
        return None

def get_top_ads_for_content(model, content, top_k=3):
    """Dự đoán top K quảng cáo cho một nội dung user"""
    if not model.is_trained:
        return []
    
    # Preprocess content
    processed_content = model.preprocess_text(content)
    if not processed_content:
        return []
    
    try:
        # Transform content thành TF-IDF vector
        content_vector = model.tfidf_vectorizer.transform([processed_content])
        
        # Tính similarity với tất cả ads
        ad_similarities = []
        
        for ad_id, ad_vector in model.ad_profiles.items():
            # Tính cosine similarity
            similarity = np.dot(content_vector.toarray()[0], ad_vector.toarray()[0])
            norm_content = np.linalg.norm(content_vector.toarray()[0])
            norm_ad = np.linalg.norm(ad_vector.toarray()[0])
            
            if norm_content > 0 and norm_ad > 0:
                similarity = similarity / (norm_content * norm_ad)
            else:
                similarity = 0
            
            ad_similarities.append((ad_id, similarity))
        
        # Sort theo similarity và lấy top K
        ad_similarities.sort(key=lambda x: x[1], reverse=True)
        top_ads = ad_similarities[:top_k]
        
        return top_ads
        
    except Exception as e:
        print(f"Error processing content: {e}")
        return []

def load_ads_info(ads_file="data/ads.json"):
    """Load thông tin chi tiết về ads"""
    try:
        with open(ads_file, 'r', encoding='utf-8') as f:
            ads_data = json.load(f)
        
        ads_info = {}
        for ad in ads_data:
            ad_id = ad.get('id', '')
            ads_info[ad_id] = {
                'description': ad.get('ad_description', ''),
                'topic': ad.get('topic', ''),
                'target_audience': ad.get('target_audience', '')
            }
        
        return ads_info
    except:
        print("⚠️  Không thể load thông tin ads từ data/ads.json")
        return {}

def test_recommendations():
    """Test model với 5 input content"""
    
    print("="*80)
    print("🎯 TEST QUẢNG CÁO ĐỀ XUẤT")
    print("="*80)
    
    # Load model
    model = load_latest_model()
    if not model:
        return
    
    # Load ads info
    ads_info = load_ads_info()
    
    # 5 test contents
    test_contents = [
        "I love new technology trends, especially artificial intelligence and machine learning. The latest smartphone features are amazing and I'm always looking for innovative tech gadgets.",
        
        "Maintaining a healthy lifestyle is very important to me. I enjoy working out at the gym, doing yoga, and eating nutritious food. Fitness equipment and health supplements interest me a lot.",
        
        "Gaming is my passion! I spend hours playing video games, especially RPGs and strategy games. Always looking for the best gaming laptops and peripherals to improve my gaming experience.",
        
        "I love traveling to new places and trying different cuisines. Food photography and exploring local restaurants are my hobbies. Travel gear and cooking equipment catch my attention.",
        
        "Entrepreneurship and business development fascinate me. I'm interested in investment strategies, financial planning, and startup tools. Always reading business books and market analysis."
    ]
    
    print(f"🔍 Testing với {len(test_contents)} contents...\n")
    
    # Test từng content
    for i, content in enumerate(test_contents, 1):
        print(f"🧪 TEST {i}:")
        print(f"📝 Content: {content}")
        print()
        
        # Dự đoán top 3 ads
        top_ads = get_top_ads_for_content(model, content, top_k=3)
        
        if not top_ads:
            print("❌ Không thể dự đoán quảng cáo cho content này")
        else:
            print("🎯 TOP 3 QUẢNG CÁO ĐỀ XUẤT:")
            
            for rank, (ad_id, similarity) in enumerate(top_ads, 1):
                print(f"   {rank}. AD ID: {ad_id} (Score: {similarity:.4f})")
                
                # Hiển thị thông tin ad nếu có
                if ad_id in ads_info:
                    ad_detail = ads_info[ad_id]
                    print(f"      {ad_detail['description']}")
                    print(f"      Topic: {ad_detail['topic']} | Target: {ad_detail['target_audience']}")
                
                print()
        
        print("-" * 60)
        print()
    
    print("✅ Test hoàn thành!")

if __name__ == "__main__":
    test_recommendations()