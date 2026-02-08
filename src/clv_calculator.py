"""
Customer Lifetime Value (CLV) Calculator

Implements CLV calculation and value-based customer segmentation.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CLVCalculator:
    """
    Calculate Customer Lifetime Value for telecom customers.
    
    CLV Formula:
    CLV = (Average Monthly Revenue * Gross Margin %) * Average Customer Lifespan
    
    Assumptions:
    - Gross margin: 50% (industry standard for telecom)
    - Discount rate: 10% annually
    - Churn impacts expected lifespan
    """
    
    def __init__(self, gross_margin=0.50, discount_rate=0.10):
        """
        Initialize CLV calculator.
        
        Args:
            gross_margin (float): Profit margin (default 50%)
            discount_rate (float): Annual discount rate (default 10%)
        """
        self.gross_margin = gross_margin
        self.discount_rate = discount_rate
        
    def calculate_clv(self, monthly_charges, tenure, churn_probability=None):
        """
        Calculate Customer Lifetime Value.
        
        Args:
            monthly_charges (float): Monthly revenue from customer
            tenure (int): Current tenure in months
            churn_probability (float, optional): Predicted churn probability
            
        Returns:
            float: Estimated CLV in dollars
        """
        # Base monthly profit
        monthly_profit = monthly_charges * self.gross_margin
        
        # Estimate remaining lifespan
        if churn_probability is not None:
            # Higher churn probability = shorter expected lifespan
            expected_remaining_months = self._estimate_lifespan(
                tenure, churn_probability
            )
        else:
            # Use industry average (24 months) adjusted by current tenure
            expected_remaining_months = max(24 - tenure, 12)
        
        # Calculate present value of future cash flows
        clv = 0
        monthly_discount = (1 + self.discount_rate) ** (1/12)
        
        for month in range(1, int(expected_remaining_months) + 1):
            discounted_profit = monthly_profit / (monthly_discount ** month)
            clv += discounted_profit
        
        # Add value of tenure already completed
        historical_value = monthly_profit * tenure
        
        total_clv = historical_value + clv
        
        return round(total_clv, 2)
    
    def _estimate_lifespan(self, tenure, churn_probability):
        """
        Estimate expected remaining customer lifespan.
        
        Args:
            tenure (int): Current tenure in months
            churn_probability (float): Churn probability (0-1)
            
        Returns:
            float: Expected remaining months
        """
        # Base expectation: inverse of churn probability
        # If 20% churn probability, expected life = 1/0.20 = 5 months
        base_expectation = 1 / max(churn_probability, 0.01) * 12
        
        # Adjust based on loyalty (tenure)
        loyalty_factor = min(tenure / 60, 1.5)  # Cap at 1.5x bonus
        
        expected_months = base_expectation * loyalty_factor
        
        # Bound between 6 and 60 months
        return max(min(expected_months, 60), 6)
    
    def segment_by_value(self, df, clv_column='CLV', percentiles=[33, 67]):
        """
        Segment customers by CLV into tiers.
        
        Args:
            df (pd.DataFrame): Dataframe with CLV column
            clv_column (str): Name of CLV column
            percentiles (list): Percentile thresholds for segmentation
            
        Returns:
            pd.DataFrame: Dataframe with value_segment column
        """
        df = df.copy()
        
        thresholds = np.percentile(df[clv_column], percentiles)
        
        df['value_segment'] = 'Low Value'
        df.loc[df[clv_column] > thresholds[0], 'value_segment'] = 'Medium Value'
        df.loc[df[clv_column] > thresholds[1], 'value_segment'] = 'High Value'
        
        # Log distribution
        value_dist = df['value_segment'].value_counts()
        logger.info("CLV Segmentation:")
        for segment, count in value_dist.items():
            avg_clv = df[df['value_segment'] == segment][clv_column].mean()
            logger.info(f"  {segment}: {count} customers (avg CLV: ${avg_clv:,.2f})")
        
        return df
    
    def calculate_portfolio_metrics(self, df, clv_column='CLV'):
        """
        Calculate aggregate CLV metrics for customer portfolio.
        
        Args:
            df (pd.DataFrame): Customer dataframe with CLV
            clv_column (str): Name of CLV column
            
        Returns:
            dict: Portfolio-level metrics
        """
        metrics = {
            'total_portfolio_value': df[clv_column].sum(),
            'average_clv': df[clv_column].mean(),
            'median_clv': df[clv_column].median(),
            'clv_std': df[clv_column].std(),
            'top_10_percent_value': df.nlargest(int(len(df) * 0.1), clv_column)[clv_column].sum(),
            'bottom_50_percent_value': df.nsmallest(int(len(df) * 0.5), clv_column)[clv_column].sum()
        }
        
        # Calculate concentration
        metrics['top_10_percent_share'] = (
            metrics['top_10_percent_value'] / metrics['total_portfolio_value']
        )
        
        logger.info("\nPORTFOLIO METRICS:")
        logger.info(f"Total Portfolio Value: ${metrics['total_portfolio_value']:,.2f}")
        logger.info(f"Average CLV: ${metrics['average_clv']:,.2f}")
        logger.info(f"Top 10% Customers Hold: {metrics['top_10_percent_share']:.1%} of value")
        
        return metrics
    
    def prioritize_retention(self, df, churn_prob_column='churn_probability',
                           clv_column='CLV', top_n=100):
        """
        Identify top priority customers for retention efforts.
        
        Args:
            df (pd.DataFrame): Customer dataframe
            churn_prob_column (str): Churn probability column
            clv_column (str): CLV column
            top_n (int): Number of top customers to identify
            
        Returns:
            pd.DataFrame: Top priority customers
        """
        df = df.copy()
        
        # Calculate retention priority score
        # High churn probability × High CLV = High priority
        df['retention_priority'] = df[churn_prob_column] * df[clv_column]
        
        # Get top N
        priority_customers = df.nlargest(top_n, 'retention_priority')
        
        total_at_risk_value = priority_customers[clv_column].sum()
        avg_churn_prob = priority_customers[churn_prob_column].mean()
        
        logger.info(f"\nTOP {top_n} RETENTION PRIORITIES:")
        logger.info(f"Total at-risk value: ${total_at_risk_value:,.2f}")
        logger.info(f"Average churn probability: {avg_churn_prob:.1%}")
        
        return priority_customers
