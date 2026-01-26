"""
Model Training for Credit Risk Scoring
=======================================
Trains XGBoost, Logistic Regression, and Neural Network models
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                            confusion_matrix, roc_curve, auc)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class CreditRiskModelTrainer:
    """
    Train and evaluate credit risk models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def prepare_data(self, df, feature_cols, target_col='defaulted', test_size=0.2):
        """
        Split data into train and test sets with SMOTE
        """
        print("\n" + "="*60)
        print("📊 PREPARING DATA")
        print("="*60)
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"\n   Total samples: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Class distribution: {dict(y.value_counts())}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\n   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # Handle class imbalance with SMOTE
        print("\n   Applying SMOTE for class balance...")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"   Balanced train set: {len(X_train_balanced)} samples")
        print(f"   New class distribution: {dict(pd.Series(y_train_balanced).value_counts())}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression model
        """
        print("\n" + "="*60)
        print("🔵 TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        model = LogisticRegression(
            penalty='l2',
            C=0.1,
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        print("\n   Fitting model...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        # Store
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        print("\n   ✅ Logistic Regression training complete!")
        
        return model, results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model
        """
        print("\n" + "="*60)
        print("🟢 TRAINING XGBOOST")
        print("="*60)
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n   Scale pos weight: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        print("\n   Fitting model...")
        eval_set = [(X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        # Feature importance
        self._save_feature_importance(
            model.feature_importances_,
            X_train.columns,
            "xgboost_feature_importance.png"
        )
        
        # Store
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        print("\n   ✅ XGBoost training complete!")
        
        return model, results
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\n   📊 Evaluating {model_name}...")
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        repayment_rate = (tn + tp) / (tn + fp + fn + tp)
        default_rate = 1 - repayment_rate
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'repayment_rate': repayment_rate,
            'default_rate': default_rate,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n   📈 {model_name} Results:")
        print(f"      Repayment Rate: {repayment_rate*100:.2f}%")
        print(f"      Default Rate: {default_rate*100:.2f}%")
        print(f"      ROC-AUC: {roc_auc:.4f}")
        
        # Check targets
        meets_repayment = repayment_rate >= 0.95
        meets_default = default_rate <= 0.03
        
        print(f"\n      {'✅' if meets_repayment else '❌'} Repayment Target (>95%)")
        print(f"      {'✅' if meets_default else '❌'} Default Target (<3%)")
        
        # Save ROC curve
        self._save_roc_curve(y_true, y_pred_proba, model_name)
        
        return results
    
    def _save_feature_importance(self, importance, feature_names, filename):
        """Save feature importance plot"""
        # Get top 20 features
        indices = np.argsort(importance)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        
        os.makedirs('models', exist_ok=True)
        plt.savefig(f'models/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      💾 Feature importance saved: models/{filename}")
    
    def _save_roc_curve(self, y_true, y_pred_proba, model_name):
        """Save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{model_name.lower().replace(' ', '_')}_roc.png"
        plt.savefig(f'models/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      💾 ROC curve saved: models/{filename}")
    
    def compare_models(self):
        """
        Compare all trained models
        """
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'Repayment Rate (%)': f"{results['repayment_rate']*100:.2f}",
                'Default Rate (%)': f"{results['default_rate']*100:.2f}",
                'ROC-AUC': f"{results['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        print("\n💾 Comparison saved to models/model_comparison.csv")
        
        return comparison_df
    
    def save_models(self, prefix='models/'):
        """Save all trained models"""
        os.makedirs(prefix, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = f'{prefix}{name}_model.pkl'
            joblib.dump(model, filepath)
            print(f"   💾 Saved: {filepath}")
        
        # Save results
        joblib.dump(self.results, f'{prefix}model_results.pkl')
        print(f"   💾 Saved: {prefix}model_results.pkl")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load processed data
    print("\n📂 Loading processed data...")
    df = pd.read_csv('data/processed/msme_features_scaled.csv')
    print(f"   ✅ Loaded {len(df)} records")
    
    # Load feature columns
    print("\n📋 Loading feature configuration...")
    from feature_engineering import MSMEFeatureEngineering
    fe = MSMEFeatureEngineering()
    fe.load_transformers()
    feature_cols = fe.get_feature_columns()
    print(f"   ✅ Loaded {len(feature_cols)} features")
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, feature_cols)
    
    # Train models
    print("\n" + "="*60)
    print("🎯 TRAINING MODELS")
    print("="*60)
    
    lr_model, lr_results = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    xgb_model, xgb_results = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Save everything
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60)
    trainer.save_models()
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\n🎯 Next step:")
    print("   Run: streamlit run streamlit_app/app.py")