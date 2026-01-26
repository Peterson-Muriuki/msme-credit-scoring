Feature Engineering for Credit Risk Model
==========================================
Creates derived features from alternative data sources
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class MSMEFeatureEngineering:
    """
    Feature engineering pipeline for MSME credit risk assessment
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_cols = []
        
    def create_features(self, df, is_training=True):
        """
        Create engineered features from raw data
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df = df.copy()
        original_shape = df.shape
        
        # 1. FINANCIAL HEALTH INDICATORS
        print("\n1 Creating financial health features...")
        df['debt_to_income'] = df['loan_amount'] / (df['monthly_revenue'] * df['loan_term_months'])
        df['revenue_per_employee'] = df['monthly_revenue'] / df['num_employees']
        df['loan_to_monthly_revenue'] = df['loan_amount'] / df['monthly_revenue']
        
        # 2. MOBILE MONEY FEATURES
        print("2 Creating mobile money features...")
        df['total_monthly_volume'] = df['avg_monthly_transactions'] * df['avg_transaction_amount']
        df['transaction_consistency'] = df['mobile_money_tenure_months'] * (1 + df['transaction_velocity'])
        df['mm_activity_score'] = (
            df['total_monthly_volume'] / 10000 * 0.4 +
            df['mobile_money_tenure_months'] / 60 * 0.3 +
            (df['transaction_velocity'] + 0.1) * 0.3
        )
        
        # Transaction size categories
        df['transaction_size_category'] = pd.cut(
            df['avg_transaction_amount'], 
            bins=[0, 50, 200, 1000, np.inf],
            labels=['micro', 'small', 'medium', 'large']
        )
        
        # 3. BUSINESS MATURITY SCORE
        print("3 Creating business maturity features...")
        df['maturity_score'] = (
            df['business_age_months'] / 120 * 0.4 +
            df['has_business_permit'] * 0.3 +
            df['has_tax_id'] * 0.3
        )
        
        # 4. CREDIT HISTORY FEATURES
        print("4 Creating credit history features...")
        df['is_repeat_borrower'] = (df['num_previous_loans'] > 0).astype(int)
        df['high_risk_history'] = (
            (df['previous_default'] == 1) | 
            (df['num_previous_loans'] > 4)
        ).astype(int)
        
        # 5. PAYMENT BEHAVIOR COMPOSITE
        print("5 Creating payment behavior features...")
        df['avg_payment_score'] = (
            df['utility_payment_score'] + 
            df['rent_payment_score']
        ) / 2
        df['reliable_payer'] = (df['avg_payment_score'] > 70).astype(int)
        
        # 6. SOCIAL CAPITAL FEATURES
        print("6 Creating social capital features...")
        df['network_strength'] = (
            df['num_business_connections'] / 20 * 0.5 +
            df['social_score'] / 100 * 0.5
        )
        
        # 7. SECTOR-SPECIFIC FEATURES
        print("7 Creating sector-specific features...")
        high_risk_sectors = ['Agriculture', 'Transportation']
        df['high_risk_sector'] = df['sector'].isin(high_risk_sectors).astype(int)
        
        # 8. COUNTRY RISK FEATURES
        print("8 Creating country risk features...")
        low_risk_countries = ['Rwanda', 'Kenya']
        medium_risk_countries = ['Ghana', 'Tanzania']
        
        df['country_risk_tier'] = df['country'].apply(
            lambda x: 'low' if x in low_risk_countries 
            else 'medium' if x in medium_risk_countries 
            else 'high'
        )
        
        # 9. LOAN CHARACTERISTICS
        print("9 Creating loan characteristic features...")
        df['loan_size_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 2000, 5000, 15000, np.inf],
            labels=['small', 'medium', 'large', 'xlarge']
        )
        
        df['loan_term_category'] = pd.cut(
            df['loan_term_months'],
            bins=[0, 6, 12, np.inf],
            labels=['short', 'medium', 'long']
        )
        
        df['interest_rate_category'] = pd.cut(
            df['interest_rate'],
            bins=[0, 0.15, 0.25, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # 10. INTERACTION FEATURES
        print("10 Creating interaction features...")
        df['mature_formal_business'] = (
            (df['business_age_months'] > 24) & 
            (df['has_business_permit'] == 1)
        ).astype(int)
        
        df['trusted_borrower'] = (
            (df['avg_payment_score'] > 70) & 
            (df['social_score'] > 60)
        ).astype(int)
        
        # 11. ENCODE CATEGORICAL VARIABLES
        print("11 Encoding categorical variables...")
        categorical_cols = [
            'country', 'sector', 'transaction_size_category', 
            'country_risk_tier', 'loan_size_category', 
            'loan_term_category', 'interest_rate_category'
        ]
        
        for col in categorical_cols:
            if is_training:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: self.encoders[col].transform([x])[0] 
                    if x in self.encoders[col].classes_ 
                    else -1
                )
        
        # 12. HANDLE MISSING VALUES
        print("12 Handling missing values...")
        df = df.fillna(df.median(numeric_only=True))
        
        print(f"\nFeature engineering complete!")
        print(f"   Original shape: {original_shape}")
        print(f"   New shape: {df.shape}")
        print(f"   Features added: {df.shape[1] - original_shape[1]}")
        
        return df
