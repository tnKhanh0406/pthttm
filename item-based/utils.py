"""
Utility functions cho data preprocessing và analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

def analyze_data_statistics(users_file, ads_file, interactions_file):
    """Phân tích thống kê dữ liệu"""
    print("="*60)
    print("DATA STATISTICS ANALYSIS")
    print("="*60)
    
    # Load data
    with open(users_file, 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)
    
    with open(interactions_file, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    
    # User statistics
    print(f"\nUSER STATISTICS:")
    print(f"Total users: {len(users)}")
    
    posts_per_user = [len(user.get('posts', [])) for user in users]
    comments_per_user = [len(user.get('comments', [])) for user in users]
    
    print(f"Posts per user - Mean: {np.mean(posts_per_user):.2f}, Median: {np.median(posts_per_user):.2f}")
    print(f"Comments per user - Mean: {np.mean(comments_per_user):.2f}, Median: {np.median(comments_per_user):.2f}")
    
    # Content length analysis
    total_content_lengths = []
    for user in users:
        content_length = 0
        for post in user.get('posts', []):
            content_length += len(post.get('title', '') + ' ' + post.get('text', ''))
        for comment in user.get('comments', []):
            content_length += len(comment.get('text', ''))
        total_content_lengths.append(content_length)
    
    print(f"Content length per user - Mean: {np.mean(total_content_lengths):.2f} chars")
    
    # Ad statistics
    print(f"\nAD STATISTICS:")
    print(f"Total ads: {len(ads)}")
    
    ad_topics = [ad.get('topic', 'Unknown') for ad in ads]
    topic_counts = Counter(ad_topics)
    print(f"Top ad topics: {dict(topic_counts.most_common(5))}")
    
    # Interaction statistics
    print(f"\nINTERACTION STATISTICS:")
    print(f"Total interactions: {len(interactions)}")
    
    labels = [interaction.get('label', 0) for interaction in interactions]
    positive_rate = sum(labels) / len(labels) if labels else 0
    print(f"Positive interaction rate: {positive_rate:.4f} ({positive_rate*100:.2f}%)")
    
    # User interaction distribution
    user_interaction_counts = Counter([interaction.get('username') for interaction in interactions])
    interactions_per_user = list(user_interaction_counts.values())
    print(f"Interactions per user - Mean: {np.mean(interactions_per_user):.2f}, Median: {np.median(interactions_per_user):.2f}")
    
    # Ad interaction distribution  
    ad_interaction_counts = Counter([interaction.get('ad_id') for interaction in interactions])
    interactions_per_ad = list(ad_interaction_counts.values())
    print(f"Interactions per ad - Mean: {np.mean(interactions_per_ad):.2f}, Median: {np.median(interactions_per_ad):.2f}")
    
    return {
        'users': len(users),
        'ads': len(ads), 
        'interactions': len(interactions),
        'positive_rate': positive_rate,
        'avg_posts_per_user': np.mean(posts_per_user),
        'avg_comments_per_user': np.mean(comments_per_user),
        'avg_content_length': np.mean(total_content_lengths)
    }

def validate_data_format(users_file, ads_file, interactions_file):
    """Kiểm tra format dữ liệu"""
    print("="*60)
    print("DATA FORMAT VALIDATION")
    print("="*60)
    
    errors = []
    
    try:
        # Validate users
        with open(users_file, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        for i, user in enumerate(users[:5]):  # Check first 5 users
            if 'username' not in user:
                errors.append(f"User {i}: Missing 'username' field")
            if 'posts' not in user:
                errors.append(f"User {i}: Missing 'posts' field")
            elif not isinstance(user['posts'], list):
                errors.append(f"User {i}: 'posts' should be a list")
            if 'comments' not in user:
                errors.append(f"User {i}: Missing 'comments' field")
            elif not isinstance(user['comments'], list):
                errors.append(f"User {i}: 'comments' should be a list")
        
        print(f"✓ Users file format OK ({len(users)} users)")
        
    except Exception as e:
        errors.append(f"Users file error: {e}")
    
    try:
        # Validate ads
        with open(ads_file, 'r', encoding='utf-8') as f:
            ads = json.load(f)
        
        for i, ad in enumerate(ads[:5]):  # Check first 5 ads
            if 'id' not in ad:
                errors.append(f"Ad {i}: Missing 'id' field")
            if 'ad_description' not in ad:
                errors.append(f"Ad {i}: Missing 'ad_description' field")
        
        print(f"✓ Ads file format OK ({len(ads)} ads)")
        
    except Exception as e:
        errors.append(f"Ads file error: {e}")
    
    try:
        # Validate interactions
        with open(interactions_file, 'r', encoding='utf-8') as f:
            interactions = json.load(f)
        
        for i, interaction in enumerate(interactions[:5]):  # Check first 5 interactions
            if 'username' not in interaction:
                errors.append(f"Interaction {i}: Missing 'username' field")
            if 'ad_id' not in interaction:
                errors.append(f"Interaction {i}: Missing 'ad_id' field")
            if 'label' not in interaction:
                errors.append(f"Interaction {i}: Missing 'label' field")
            elif interaction['label'] not in [0, 1]:
                errors.append(f"Interaction {i}: Label should be 0 or 1")
        
        print(f"✓ Interactions file format OK ({len(interactions)} interactions)")
        
    except Exception as e:
        errors.append(f"Interactions file error: {e}")
    
    if errors:
        print(f"\n✗ Found {len(errors)} validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"\n✓ All data files passed validation!")
        return True

def check_data_consistency(users_file, ads_file, train_file, test_file):
    """Kiểm tra tính nhất quán giữa các file dữ liệu"""
    print("="*60)
    print("DATA CONSISTENCY CHECK")
    print("="*60)
    
    # Load all data
    with open(users_file, 'r', encoding='utf-8') as f:
        users = json.load(f)
    with open(ads_file, 'r', encoding='utf-8') as f:
        ads = json.load(f)
    with open(train_file, 'r', encoding='utf-8') as f:
        train_interactions = json.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_interactions = json.load(f)
    
    # Get all usernames and ad_ids
    all_usernames = set(user['username'] for user in users)
    all_ad_ids = set(ad['id'] for ad in ads)
    
    # Check train interactions
    train_usernames = set(interaction['username'] for interaction in train_interactions)
    train_ad_ids = set(interaction['ad_id'] for interaction in train_interactions)
    
    # Check test interactions  
    test_usernames = set(interaction['username'] for interaction in test_interactions)
    test_ad_ids = set(interaction['ad_id'] for interaction in test_interactions)
    
    # Find missing references
    missing_train_users = train_usernames - all_usernames
    missing_train_ads = train_ad_ids - all_ad_ids
    missing_test_users = test_usernames - all_usernames
    missing_test_ads = test_ad_ids - all_ad_ids
    
    print(f"Users in data: {len(all_usernames)}")
    print(f"Ads in data: {len(all_ad_ids)}")
    print(f"Train interactions: {len(train_interactions)}")
    print(f"Test interactions: {len(test_interactions)}")
    
    issues = []
    
    if missing_train_users:
        issues.append(f"Train interactions reference {len(missing_train_users)} unknown users")
    if missing_train_ads:
        issues.append(f"Train interactions reference {len(missing_train_ads)} unknown ads")
    if missing_test_users:
        issues.append(f"Test interactions reference {len(missing_test_users)} unknown users")  
    if missing_test_ads:
        issues.append(f"Test interactions reference {len(missing_test_ads)} unknown ads")
    
    # Check for overlap between train and test
    overlapping_interactions = set()
    for train_int in train_interactions:
        train_key = (train_int['username'], train_int['ad_id'])
        for test_int in test_interactions:
            test_key = (test_int['username'], test_int['ad_id'])
            if train_key == test_key:
                overlapping_interactions.add(train_key)
    
    if overlapping_interactions:
        issues.append(f"Found {len(overlapping_interactions)} overlapping user-ad pairs between train and test")
    
    if issues:
        print(f"\n✗ Found {len(issues)} consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ Data consistency check passed!")
        return True

if __name__ == "__main__":
    # Example usage
    users_file = "../../data/reddit_dataset_sentiment.json"
    ads_file = "../../data/ads_dataset_14_unique.json"
    train_file = "../../data/user_ad_interactions_train2.json"
    test_file = "../../data/user_ad_interactions_test2.json"
    
    # Run all checks
    print("Running data analysis and validation...\n")
    
    # Validate format
    format_ok = validate_data_format(users_file, ads_file, train_file)
    
    if format_ok:
        # Analyze statistics
        stats = analyze_data_statistics(users_file, ads_file, train_file)
        
        # Check consistency
        consistency_ok = check_data_consistency(users_file, ads_file, train_file, test_file)
        
        if consistency_ok:
            print(f"\n✅ All data checks passed! Ready for training.")
        else:
            print(f"\n❌ Data consistency issues found. Please fix before training.")
    else:
        print(f"\n❌ Data format issues found. Please fix before training.")