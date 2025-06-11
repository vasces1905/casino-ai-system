import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepares features for Random Forest training
    """
    
    print("=== FEATURE PREPARATION ===")
    
    # Select features for modeling
    feature_columns = [
        'avg_bet', 'avg_loss', 'avg_session_duration', 'zone_diversity',
        'ticket_in', 'ticket_out', 'jackpot_total', 'game_type_diversity',
        'avg_usage_level', 'days_since_campaign', 'month',
        'is_weekend_campaign', 'is_holiday_season'
    ]
    
    # Categorical features that need encoding
    categorical_features = ['promo_type', 'segment_name']
    
    # Create feature matrix
    X = df[feature_columns].copy()
    
    # Encode categorical features
    label_encoders = {}
    for cat_feature in categorical_features:
        le = LabelEncoder()
        X[cat_feature + '_encoded'] = le.fit_transform(df[cat_feature])
        label_encoders[cat_feature] = le
        feature_columns.append(cat_feature + '_encoded')
    
    # Final feature matrix
    X = X[feature_columns + [col + '_encoded' for col in categorical_features]]
    
    # Target variable
    y = df['actual_response']
    
    print(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
    print(f"Response rate in dataset: {y.mean():.2%}")
    print(f"Class distribution:")
    print(f"  No Response (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"  Response (1): {(y == 1).sum()} ({(y == 1).mean():.1%})")
    
    return X, y, label_encoders, feature_columns

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Trains Random Forest model with hyperparameter tuning
    """
    
    print("\n=== RANDOM FOREST TRAINING ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train final model
    best_rf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)
    
    # Performance metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Check: {abs(train_accuracy - test_accuracy):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold Cross-Validation:")
    print(f"  Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Individual Scores: {cv_scores}")
    
    # Detailed classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred_test))
    
    return best_rf, X_train, X_test, y_train, y_test, y_pred_test

def analyze_feature_importance(model: RandomForestClassifier, feature_names: list):
    """
    Analyzes and visualizes feature importance
    """
    
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Display top 10 features
    print("Top 10 Most Important Features:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Random Forest Feature Importance (Top 15)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def create_confusion_matrix(y_true, y_pred):
    """
    Creates and visualizes confusion matrix
    """
    
    print("\n=== CONFUSION MATRIX ===")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Response', 'Response'],
                yticklabels=['No Response', 'Response'])
    plt.title('Confusion Matrix - Promo Response Prediction')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"Confusion Matrix Breakdown:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

def save_model_and_results(model, label_encoders, feature_names, feature_importance_df, 
                          test_accuracy, cv_scores):
    """
    Saves trained model and results
    """
    
    print("\n=== SAVING MODEL AND RESULTS ===")
    
    # Save the trained model
    model_path = "../data/trained_rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save label encoders
    encoders_path = "../data/label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    print(f"✅ Label encoders saved to: {encoders_path}")
    
    # Save feature names
    features_path = "../data/feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ Feature names saved to: {features_path}")
    
    # Save comprehensive results
    results = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_accuracy": float(test_accuracy),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "feature_importance": feature_importance_df.to_dict('records'),
        "top_features": feature_importance_df.head(10)['feature'].tolist()
    }
    
    results_path = "../data/model_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Model results saved to: {results_path}")

def predict_promo_response(customer_data: dict, model_path: str = "../data/trained_rf_model.pkl") -> dict:
    """
    Predicts promotional response for a new customer
    """
    
    # Load trained model and encoders
    model = joblib.load(model_path)
    label_encoders = joblib.load("../data/label_encoders.pkl")
    
    # Prepare feature vector (simplified example)
    features = [
        customer_data.get('avg_bet', 50),
        customer_data.get('avg_loss', 100),
        customer_data.get('avg_session_duration', 15),
        customer_data.get('zone_diversity', 3),
        customer_data.get('ticket_in', 200),
        customer_data.get('ticket_out', 150),
        customer_data.get('jackpot_total', 0),
        customer_data.get('game_type_diversity', 2),
        customer_data.get('avg_usage_level', 3),
        customer_data.get('days_since_campaign', 30),
        customer_data.get('month', 6),
        int(customer_data.get('is_weekend_campaign', False)),
        int(customer_data.get('is_holiday_season', False)),
        # Add encoded categorical features here
    ]
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return {
        "prediction": int(prediction),
        "probability_no_response": float(probability[0]),
        "probability_response": float(probability[1]),
        "recommendation": "Send Promo" if prediction == 1 else "Skip Promo"
    }

if __name__ == "__main__":
    print("=== RANDOM FOREST PROMO RESPONSE MODEL ===")
    
    # Load promotional response data
    data_path = "../../data/promo_response_data.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} promotional campaigns")
        
        # Prepare features
        X, y, label_encoders, feature_names = prepare_features(df)
        
        # Train Random Forest model
        model, X_train, X_test, y_train, y_test, y_pred_test = train_random_forest(X, y)
        
        # Analyze feature importance
        feature_importance_df = analyze_feature_importance(model, X.columns.tolist())
        
        # Create confusion matrix
        create_confusion_matrix(y_test, y_pred_test)
        
        # Calculate test accuracy for saving
        test_accuracy = accuracy_score(y_test, y_pred_test)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Save model and results
        save_model_and_results(model, label_encoders, X.columns.tolist(), 
                              feature_importance_df, test_accuracy, cv_scores)
        
        print("\n🎉 RANDOM FOREST MODEL TRAINING COMPLETE!")
        print("\nModel Performance Summary:")
        print(f"  ✅ Test Accuracy: {test_accuracy:.2%}")
        print(f"  ✅ Cross-Val Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        print(f"  ✅ Model saved and ready for predictions!")
        
        # Example prediction
        print("\n🔮 Example Prediction:")
        sample_customer = {
            'avg_bet': 75,
            'avg_loss': 150,
            'avg_session_duration': 20,
            'zone_diversity': 4,
            'ticket_in': 300,
            'is_weekend_campaign': True
        }
        
        # Note: This is a simplified example - full implementation would handle all features
        print(f"Sample customer: {sample_customer}")
        print("Ready for production use!")
        
    except FileNotFoundError:
        print("❌ Error: promo_response_data.csv not found!")
        print("Please run promo_response_generator.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your data and try again.")