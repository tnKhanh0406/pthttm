import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class RecommendationModel:
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError("model_type phải là 'random_forest' hoặc 'logistic_regression'")
    
    def train(self, X, y, test_size=0.2):
        """Train model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def cross_validate(self, X, y, cv=5):
        """Cross validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"Cross Validation Scores: {scores}")
        print(f"Average CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def predict(self, X):
        """Predict"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probability"""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save model"""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        self.model = joblib.load(filepath)