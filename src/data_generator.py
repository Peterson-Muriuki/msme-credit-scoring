"""
Data Generator for MSME Credit Risk Model
==========================================
Generates synthetic loan data for African MSMEs with realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_msme_data(n_samples=10000, output_path='data/raw/msme_loan_data.csv'):
    """
    Generate synthetic MSME loan data with realistic African market characteristics
    
    Parameters:
    -----------
    n_samples : int
        Number of loan applications to generate
    output_path : str
        Path to save the generated data
        
    Returns:
    --------
    DataFrame with generated loan data
    """
    
    print(f"Generating {n_samples} MSME loan applications...")
    print("="*60)
    
    # Define country and sector characteristics
    countries = {
        'Kenya': {'risk': 0.15, 'weight': 0.25},
        'Nigeria': {'risk': 0.25, 'weight': 0.30},
        'Ghana': {'risk': 0.18, 'weight': 0.15},
        'Tanzania': {'risk': 0.20, 'weight': 0.12},
        'Uganda': {'risk': 0.22, 'weight': 0.10},
        'Rwanda': {'risk': 0.12, 'weight': 0.08}
    }
    
    sectors = {
        'Agriculture': {'risk': 0.20, 'weight': 0.25},
        'Retail': {'risk': 0.15, 'weight': 0.20},
        'Manufacturing': {'risk': 0.18, 'weight': 0.15},
        'Services': {'risk': 0.12, 'weight': 0.15},
        'Transportation': {'risk': 0.22, 'weight': 0.10},
        'Food & Beverage': {'risk': 0.16, 'weight': 0.10},
        'Technology': {'risk': 0.10, 'weight': 0.05}
    }
    
    data = []
    
    for i in range(n_samples):
        # Select country and sector based on weights
        country = random.choices(
            list(countries.keys()), 
            weights=[c['weight'] for c in countries.values()]
        )[0]
        
        sector = random.choices(
            list(sectors.keys()),
            weights=[s['weight'] for s in sectors.values()]
        )[0]
        
        # Business characteristics
        business_age_months = min(np.random.exponential(24) + 3, 120)
        
        # Loan details
        loan_amount = np.clip(np.random.lognormal(mean=8, sigma=1.2), 500, 50000)
        loan_term_months = random.choice([3, 6, 9, 12, 18, 24])
        interest_rate = np.random.uniform(0.12, 0.35)
        
        # Financial information
        num_employees = max(1, int(np.random.exponential(3)))
        monthly_revenue = loan_amount * np.random.uniform(1.5, 8)
        
        # Mobile money features
        avg_monthly_transactions = max(10, np.random.poisson(45))
        avg_transaction_amount = np.random.lognormal(5, 1.5)
        mobile_money_tenure_months = min(business_age_months, np.random.uniform(6, 60))
        transaction_velocity = np.random.uniform(-0.1, 0.3)
        
        # Social features
        num_business_connections = np.random.poisson(8)
        social_score = np.random.beta(2, 5) * 100
        
        # Formalization
        has_business_permit = random.random() > 0.35
        has_tax_id = random.random() > 0.40
        
        # Credit history
        num_previous_loans = np.random.poisson(1.5)
        previous_default = (num_previous_loans > 0) and (random.random() < 0.15)
        
        # Payment behavior
        utility_payment_score = np.random.beta(5, 2) * 100
        rent_payment_score = np.random.beta(5, 2) * 100
        
        # Seasonality
        is_harvest_season = (sector == 'Agriculture') and (random.random() > 0.7)
        
        # Calculate default probability
        base_risk = (
            countries[country]['risk'] * 0.2 +
            sectors[sector]['risk'] * 0.15 +
            (1 / (business_age_months + 1)) * 0.15 +
            (loan_amount / monthly_revenue if monthly_revenue > 0 else 1) * 0.2 +
            (0.3 if previous_default else 0) * 0.3
        )
        
        # Protective factors
        protection = (
            (mobile_money_tenure_months / 60) * 0.1 +
            (has_business_permit * 0.05) +
            (has_tax_id * 0.05) +
            (social_score / 100) * 0.05 +
            (utility_payment_score / 100) * 0.05 +
            (transaction_velocity if transaction_velocity > 0 else 0) * 0.05
        )
        
        default_probability = np.clip(base_risk - protection, 0.01, 0.95)
        default_probability *= np.random.uniform(0.7, 1.3)
        default_probability = np.clip(default_probability, 0.01, 0.95)
        
        # Determine default
        defaulted = random.random() < default_probability
        
        # Create record
        record = {
            'loan_id': f'LOAN_{i+1:06d}',
            'country': country,
            'sector': sector,
            'business_age_months': round(business_age_months, 1),
            'loan_amount': round(loan_amount, 2),
            'loan_term_months': loan_term_months,
            'interest_rate': round(interest_rate, 4),
            'monthly_revenue': round(monthly_revenue, 2),
            'num_employees': num_employees,
            'avg_monthly_transactions': avg_monthly_transactions,
            'avg_transaction_amount': round(avg_transaction_amount, 2),
            'mobile_money_tenure_months': round(mobile_money_tenure_months, 1),
            'transaction_velocity': round(transaction_velocity, 3),
            'num_business_connections': num_business_connections,
            'social_score': round(social_score, 2),
            'has_business_permit': int(has_business_permit),
            'has_tax_id': int(has_tax_id),
            'num_previous_loans': num_previous_loans,
            'previous_default': int(previous_default),
            'utility_payment_score': round(utility_payment_score, 2),
            'rent_payment_score': round(rent_payment_score, 2),
            'is_harvest_season': int(is_harvest_season),
            'defaulted': int(defaulted),
            'application_date': (datetime.now() - timedelta(days=random.randint(1, 730))).strftime('%Y-%m-%d')
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Adjust to meet target metrics (>95% repayment, <3% default)
    current_default_rate = df['defaulted'].mean()
    target_default_rate = 0.025
    
    if current_default_rate > target_default_rate:
        num_to_flip = int((current_default_rate - target_default_rate) * len(df))
        default_indices = df[df['defaulted'] == 1].index
        flip_indices = np.random.choice(default_indices, size=num_to_flip, replace=False)
        df.loc[flip_indices, 'defaulted'] = 0
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Display summary
    print(f"\n Dataset generated successfully!")
    print(f"   Total samples: {len(df)}")
    print(f"   Default rate: {df['defaulted'].mean()*100:.2f}%")
    print(f"   Repayment rate: {(1-df['defaulted'].mean())*100:.2f}%")
    print(f"   Saved to: {output_path}")
    print("="*60)
    
    # Display distribution
    print("\n Distribution Summary:")
    print(f"\nBy Country:")
    print(df['country'].value_counts().to_string())
    print(f"\nBy Sector:")
    print(df['sector'].value_counts().to_string())
    print(f"\nDefault by Country:")
    print(df.groupby('country')['defaulted'].agg(['count', 'sum', 'mean']).to_string())
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_msme_data(n_samples=10000)
    
    print("\n" + "="*60)
    print(" Data generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python src/feature_engineering.py")
    print("2. Then: python src/model_training.py")