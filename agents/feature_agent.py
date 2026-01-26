"""
Smolagents Feature Analysis Agent
==================================
Uses Smolagents to analyze feature importance and provide insights
"""

from smolagents import CodeAgent, HfApiModel
import pandas as pd
import json

class FeatureAnalysisAgent:
    """
    AI agent for analyzing credit risk features using Smolagents
    """
    
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        """
        Initialize the feature analysis agent
        
        Parameters:
        -----------
        model_name : str
            HuggingFace model to use for the agent
        """
        print("\n🤖 Initializing Smolagent for Feature Analysis...")
        
        # Initialize model
        self.model = HfApiModel(model_id=model_name)
        
        # Create agent
        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            max_steps=5
        )
        
        print("✅ Agent initialized successfully!")
    
    def analyze_feature_importance(self, feature_importance_dict):
        """
        Analyze feature importance and provide business insights
        
        Parameters:
        -----------
        feature_importance_dict : dict
            Dictionary of feature names and their importance scores
            
        Returns:
        --------
        str : Analysis and recommendations
        """
        
        # Create prompt for analysis
        prompt = f"""
        You are a credit risk analyst. Analyze these feature importance scores from a 
        credit risk model for African MSMEs and provide actionable insights:
        
        Feature Importance (top 10):
        {json.dumps(feature_importance_dict, indent=2)}
        
        Please provide:
        1. Key insights about what drives default risk
        2. Business recommendations for loan officers
        3. Data collection priorities for improving the model
        4. Risk mitigation strategies based on top features
        
        Be specific and practical.
        """
        
        print("\n🔍 Analyzing feature importance...")
        result = self.agent.run(prompt)
        
        return result
    
    def generate_risk_profile(self, loan_application):
        """
        Generate a detailed risk profile for a loan application
        
        Parameters:
        -----------
        loan_application : dict
            Dictionary containing loan application details
            
        Returns:
        --------
        str : Risk profile analysis
        """
        
        prompt = f"""
        You are a credit risk analyst. Analyze this MSME loan application and provide 
        a detailed risk assessment:
        
        Application Details:
        {json.dumps(loan_application, indent=2)}
        
        Provide:
        1. Overall risk assessment (Low/Medium/High)
        2. Key risk factors identified
        3. Protective factors identified
        4. Specific recommendations for this application
        5. Additional information needed (if any)
        
        Be thorough and specific.
        """
        
        print("\n📋 Generating risk profile...")
        result = self.agent.run(prompt)
        
        return result
    
    def suggest_feature_engineering(self, current_features, performance_metrics):
        """
        Suggest new features to engineer based on current model performance
        
        Parameters:
        -----------
        current_features : list
            List of current features in the model
        performance_metrics : dict
            Current model performance metrics
            
        Returns:
        --------
        str : Feature engineering suggestions
        """
        
        prompt = f"""
        You are a machine learning engineer specializing in credit risk. 
        
        Current features in the model:
        {', '.join(current_features[:20])}...
        
        Current model performance:
        {json.dumps(performance_metrics, indent=2)}
        
        Suggest:
        1. New features to create from existing data
        2. Alternative data sources to consider
        3. Feature interactions to explore
        4. Domain-specific features for African MSMEs
        5. Potential feature reduction opportunities
        
        Focus on features that could improve the model's ability to identify defaults 
        while maintaining fairness.
        """
        
        print("\n💡 Generating feature engineering suggestions...")
        result = self.agent.run(prompt)
        
        return result


class RiskRecommendationAgent:
    """
    AI agent for generating loan approval recommendations
    """
    
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta"):
        """Initialize the risk recommendation agent"""
        print("\n🤖 Initializing Risk Recommendation Agent...")
        
        self.model = HfApiModel(model_id=model_name)
        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            max_steps=5
        )
        
        print("✅ Agent initialized!")
    
    def generate_recommendation(self, loan_data, default_probability, risk_factors):
        """
        Generate a comprehensive loan recommendation
        
        Parameters:
        -----------
        loan_data : dict
            Loan application details
        default_probability : float
            Predicted default probability
        risk_factors : list
            List of identified risk factors
            
        Returns:
        --------
        str : Detailed recommendation
        """
        
        prompt = f"""
        You are a senior credit analyst at a financial institution serving African MSMEs.
        
        Loan Application:
        - Amount: ${loan_data.get('loan_amount', 0):,.2f}
        - Business: {loan_data.get('sector', 'N/A')} in {loan_data.get('country', 'N/A')}
        - Business Age: {loan_data.get('business_age_months', 0)} months
        
        Model Output:
        - Default Probability: {default_probability*100:.2f}%
        - Repayment Probability: {(1-default_probability)*100:.2f}%
        
        Risk Factors Identified:
        {chr(10).join(f'- {factor}' for factor in risk_factors)}
        
        Provide a comprehensive recommendation including:
        1. Approve/Reject/Review decision with rationale
        2. Suggested loan terms (amount, rate, term) if approving
        3. Conditions or covenants to include
        4. Monitoring requirements
        5. Risk mitigation measures
        
        Be practical and consider both business growth and risk management.
        """
        
        print("\n📝 Generating loan recommendation...")
        result = self.agent.run(prompt)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING SMOLAGENTS INTEGRATION")
    print("="*60)
    
    # Example feature importance
    feature_importance = {
        "previous_default": 0.15,
        "avg_payment_score": 0.12,
        "business_age_months": 0.10,
        "debt_to_income": 0.09,
        "mobile_money_tenure_months": 0.08,
        "social_score": 0.07,
        "total_monthly_volume": 0.07,
        "has_business_permit": 0.06,
        "loan_amount": 0.05,
        "country_risk_tier": 0.04
    }
    
    # Example loan application
    loan_app = {
        "country": "Kenya",
        "sector": "Retail",
        "business_age_months": 18,
        "loan_amount": 5000,
        "monthly_revenue": 8000,
        "has_business_permit": True,
        "previous_default": False,
        "avg_payment_score": 75
    }
    
    print("\n" + "="*60)
    print("Note: Smolagents requires HuggingFace API access.")
    print("For full functionality, set HUGGINGFACE_TOKEN in your environment.")
    print("="*60)
    
    # Uncomment to test with actual API:
    # agent = FeatureAnalysisAgent()
    # analysis = agent.analyze_feature_importance(feature_importance)
    # print("\n📊 Feature Analysis:")
    # print(analysis)
    
    print("\n✅ Smolagents module created successfully!")
    print("   Ready for integration with Streamlit app")