import joblib
import json
from data_preprocessing import DataPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load model 1 lần
model = joblib.load('../src/models/recommendation_model.joblib')

# Load data 1 lần
with open('../../data/ads_dataset_14_unique.json', 'r', encoding='utf-8') as f:
    ads_data = json.load(f)

# Quick preprocessing
preprocessor = DataPreprocessor()
ads_df = pd.DataFrame([{
    'ad_id': ad['id'],
    'topic': ad.get('topic', ''),
    'content': preprocessor.clean_text(f"{ad.get('ad_description', '')} {ad.get('target_audience', '')}")
} for ad in ads_data])

# Fit vectorizer với ad content
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,1))
ad_vectors = vectorizer.fit_transform(ads_df['content'])

def recommend(user_text, top_k=5):
    """Gợi ý quảng cáo cho user text"""
    # Clean user text
    clean_text = preprocessor.clean_text(user_text)
    
    # Vectorize user text
    user_vector = vectorizer.transform([clean_text])
    
    # Tính similarity với tất cả ads
    similarities = cosine_similarity(user_vector, ad_vectors)[0]
    
    # Tạo features và predict
    results = []
    for i, sim in enumerate(similarities):
        feature = [[sim, len(clean_text.split()), len(ads_df.loc[i, 'content'].split())]]
        prob = model.predict_proba(feature)[0][1]
        results.append((i, sim, prob))
    
    # Sort by probability
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Return top-k
    recommendations = []
    for i, (idx, sim, prob) in enumerate(results[:top_k]):
        recommendations.append({
            'rank': i+1,
            'ad_id': ads_df.loc[idx, 'ad_id'],
            'topic': ads_df.loc[idx, 'topic'],
            'similarity': sim,
            'probability': prob
        })
    
    return recommendations

# Test examples
if __name__ == "__main__":
    test_cases = [
        "I love programming and machine learning AI development",
        "Fitness gym workout healthy lifestyle nutrition",
        "Travel vacation beach holiday adventure",
        "Gaming computer video games streaming"
    ]
    
    for text in test_cases:
        print(f"\nUser: {text}")
        recs = recommend(text, 3)
        for r in recs:
            print(f"  {r['rank']}. {r['ad_id']} ({r['topic']}) - {r['probability']:.3f}")