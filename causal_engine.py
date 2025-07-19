import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json
import re
from causal_analysis_state import CausalAnalysisState
from causal_workflow import create_causal_workflow
import warnings
warnings.filterwarnings('ignore')

class CausalAnalysisEngine:
    def __init__(self):
        self.workflow = create_causal_workflow()
    
    def load_and_preprocess_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Load and preprocess data - adapted from your existing function"""
        if data is None:
            try:
                data = pd.read_csv("data/sample_data.csv")
            except FileNotFoundError:
                raise Exception("Sample data file not found. Please upload your own data.")
        
        # Your existing preprocessing logic
        data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
        
        for col in data.select_dtypes(include='object').columns:
            data[col] = data[col].str.replace(",", "", regex=False)
        
        problematic_strings = [' - ', '-', 'na', 'n/a', 'none', '']
        for ps in problematic_strings:
            data.replace(ps, np.nan, inplace=True)
        
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='ignore')
        
        from sklearn.preprocessing import LabelEncoder
        for col in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            if data[col].notnull().any():
                data[col] = le.fit_transform(data[col].astype(str))
            else:
                data[col] = 0.0
        
        # Handle datetime columns
        datetime_cols = data.select_dtypes(include='datetime64[ns]').columns
        for col in datetime_cols:
            base = col
            data[f'{base}_numeric'] = (data[base] - data[base].min()).dt.days.astype('int64')
            data[f'{base}_year'] = data[base].dt.year
            data[f'{base}_month'] = data[base].dt.month
            data[f'{base}_day'] = data[base].dt.day
        
        # Fill missing values
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(data[col].mean())
            elif data[col].dtype == 'object':
                data[col] = data[col].fillna("unknown")
        
        return data
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load sample data"""
        return self.load_and_preprocess_data()
    
    def run_analysis(self, query: str, data: pd.DataFrame, 
                    confidence_level: float = 0.90, 
                    model_type: str = "linear",column_definitions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the complete causal analysis workflow"""
        
        initial_state = {
            "query": query,
            "data": data,
            "treatment": "",
            "outcome": "",
            "controls": [],
            "causal_estimate": 0.0,
            "insights": "",
            "validation_results": {},
            "next_action": "start",
            "business_cases": [],
            "query_validation": {},
            "matched_case": {},
            "quantitative_changes": {},
            "column_definitions": column_definitions or {}
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
