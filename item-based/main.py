#!/usr/bin/env python3
"""
Main script để train và evaluate Item-Based Collaborative Filtering model
với dữ liệu thực của người dùng.

Usage:
    python main.py --train
    python main.py --evaluate
    python main.py --train --evaluate
"""

import argparse
import os
import sys
from item_based_collaborative import ItemBasedCollaborativeFiltering

def check_data_files(users_file, ads_file, train_file, test_file=None):
    """Kiểm tra sự tồn tại của các file dữ liệu"""
    required_files = [
        (users_file, "Users data"),
        (ads_file, "Ads data"), 
        (train_file, "Training interactions")
    ]
    
    if test_file:
        required_files.append((test_file, "Test interactions"))
    
    missing_files = []
    for filepath, description in required_files:
        if not os.path.exists(filepath):
            missing_files.append(f"{description}: {filepath}")
    
    if missing_files:
        print("✗ Missing required data files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    print("✓ All required data files found")
    return True

def main():
    parser = argparse.ArgumentParser(description='Item-Based Collaborative Filtering for Social Media Ads')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--users', default='../../data/reddit_dataset_sentiment.json', help='Path to users data file')
    parser.add_argument('--ads', default='../../data/ads_dataset_14_unique.json', help='Path to ads data file')
    parser.add_argument('--train-interactions', default='../../data/user_ad_interactions_train2.json', 
                       help='Path to training interactions file')
    parser.add_argument('--test-interactions', default='../../data/user_ad_interactions_test2.json',
                       help='Path to test interactions file')
    parser.add_argument('--model-name', default='social_media_ad_cf', help='Model name')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--k', type=int, default=10, help='Number of similar items to consider')
    parser.add_argument('--load-model', help='Path to saved model to load')
    
    args = parser.parse_args()
    
    # Nếu không có action nào được chọn, mặc định train và evaluate
    if not args.train and not args.evaluate and not args.load_model:
        args.train = True
        args.evaluate = True
    
    print("="*80)
    print("ITEM-BASED COLLABORATIVE FILTERING FOR PERSONALIZED SOCIAL MEDIA ADS")
    print("="*80)
    
    # Khởi tạo model
    model = ItemBasedCollaborativeFiltering(model_name=args.model_name)
    
    # Load model nếu được chỉ định
    if args.load_model:
        if os.path.exists(args.load_model):
            if model.load_model(args.load_model):
                print(model.get_model_info())
            else:
                print("✗ Failed to load model")
                return 1
        else:
            print(f"✗ Model file not found: {args.load_model}")
            return 1
    
    # Training
    if args.train:
        print("\n" + "="*60)
        print("TRAINING PHASE")
        print("="*60)
        
        # Kiểm tra file dữ liệu
        if not check_data_files(args.users, args.ads, args.train_interactions):
            print("\n✗ Cannot proceed with training due to missing data files")
            print("\nExpected file structure:")
            print("  data/")
            print("  ├── users.json")
            print("  ├── ads.json")
            print("  ├── train_interactions.json")
            print("  └── test_interactions.json")
            return 1
        
        # Train model
        success = model.train(args.users, args.ads, args.train_interactions)
        
        if success:
            print(model.get_model_info())
        else:
            print("✗ Training failed")
            return 1
    
    # Evaluation
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATION PHASE")
        print("="*60)
        
        if not model.is_trained:
            print("✗ Model must be trained before evaluation")
            return 1
        
        # Kiểm tra test file
        if not os.path.exists(args.test_interactions):
            print(f"✗ Test interactions file not found: {args.test_interactions}")
            return 1
        
        # Evaluate model
        results = model.evaluate(
            test_interactions_file=args.test_interactions,
            threshold=args.threshold,
            k=args.k
        )
        
        if results is None:
            print("✗ Evaluation failed")
            return 1
        
        # Tổng kết
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Model Performance:")
        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"Prediction Results:")
        print(f"  Correct: {results['correct_predictions']}")
        print(f"  Incorrect: {results['incorrect_predictions']}")
        print(f"  Total: {results['correct_predictions'] + results['incorrect_predictions']}")
    
    print("\n✓ Process completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
