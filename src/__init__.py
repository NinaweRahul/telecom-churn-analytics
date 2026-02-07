"""
Telecom Customer Churn and Lifetime Value Analysis

A comprehensive ene-to-end analytical project to analyze and predict customer churn in the telecom industry. 
This project demonstrates the complete data science pipeline : data preprocessing, exploratory data analysis, feature engineering, predictive modeling, and business intelligence reporting. 
The project is designed to provide actionable insights for telecom companies to reduce churn and maximize customer lifetime value.
"""

__version__ = "1.0.0"
__author__ = "Rahul Ninawe"

from .data_preprocessing import DataPreprocessor
from .churn_model import ChurnPredictor
from .clv_calculator import CLVCalculator
from .segmentation import CustomerSegmentator
from .visualization import BIVisualizer

__all__ = [
    "DataPreprocessor",
    "ChurnPredictor",
    "CLVCalculator",
    "CustomerSegmentator",
    "BIVisualizer"
]