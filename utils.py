import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any

def format_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format analysis results for display"""
    formatted = {}
    
    # Format numeric values
    for key, value in results.items():
        if isinstance(value, float):
            formatted[key] = round(value, 4)
        elif isinstance(value, dict):
            formatted[key] = format_nested_dict(value)
        else:
            formatted[key] = value
    
    return formatted

def format_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Format nested dictionary values"""
    formatted = {}
    for key, value in d.items():
        if isinstance(value, float):
            formatted[key] = round(value, 4)
        elif isinstance(value, dict):
            formatted[key] = format_nested_dict(value)
        else:
            formatted[key] = value
    return formatted

def create_visualizations(results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Create visualizations from analysis results"""
    figs = {}
    
    validation_results = results.get('validation_results', {})
    
    # Feature importance plot
    feature_importance = validation_results.get('feature_importance', {})
    if feature_importance:
        figs['feature_importance'] = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Feature Importance"
        )
    
    # Scenario analysis plot
    scenario_analysis = validation_results.get('scenario_analysis', {})
    if scenario_analysis:
        scenarios = list(scenario_analysis.keys())
        outcome_changes = [scenario_analysis[s].get('outcome_change_percent', 0) for s in scenarios]
        
        figs['scenario_analysis'] = px.bar(
            x=scenarios,
            y=outcome_changes,
            title="Scenario Analysis - Outcome Changes"
        )
    
    # Correlation matrix heatmap
    correlation_matrix = validation_results.get('correlation_matrix', {})
    if correlation_matrix:
        # Convert to DataFrame for plotting
        corr_df = pd.DataFrame(correlation_matrix)
        figs['correlation_matrix'] = px.imshow(
            corr_df,
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
    
    return figs



def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table from results"""
    validation_results = results.get('validation_results', {})
    
    summary_data = {
        'Metric': [
            'Causal Effect (ATE)',
            'Sample Size',
            'Model Score',
            'Treatment Mean',
            'Outcome Mean'
        ],
        'Value': [
            results.get('causal_estimate', 0),
            validation_results.get('sample_size', 0),
            validation_results.get('model_score', 0),
            validation_results.get('baseline_stats', {}).get('treatment_mean', 0),
            validation_results.get('baseline_stats', {}).get('outcome_mean', 0)
        ]
    }
    
    return pd.DataFrame(summary_data)
