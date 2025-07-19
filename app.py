import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from causal_engine import CausalAnalysisEngine
from utils import format_results, create_visualizations
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Causal Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if "column_definitions" not in st.session_state:
    st.session_state.column_definitions = {}

class StreamlitCausalApp:
    def __init__(self):
        self.engine = CausalAnalysisEngine()
        
    def run(self):
        st.title(" Causal Analysis Dashboard")
        st.markdown("*Discover causal relationships in your data with advanced statistical methods*")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Select Page",
                ["Data Upload", "Query Analysis", "Results Dashboard", "Advanced Settings"]
            )  
        
        if page == "Data Upload":
            self.data_upload_page()
        elif page == "Query Analysis":
            self.query_analysis_page()
        elif page == "Results Dashboard":
            self.results_dashboard_page()
        elif page == "Advanced Settings":
            self.advanced_settings_page()
    
    def data_upload_page(self):
        st.header(" Data Upload & Preview")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file containing your data for analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load and preprocess data
                data = pd.read_csv(uploaded_file)
                st.session_state.raw_data = data
                
                # Preprocess data using your existing function
                processed_data = self.engine.load_and_preprocess_data(data)
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                
                # Display data preview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Raw Data Preview")
                    st.dataframe(data.head(), use_container_width=True)
                    st.write(f"Shape: {data.shape}")
                
                with col2:
                    st.subheader("Processed Data Preview")
                    st.dataframe(processed_data.head(), use_container_width=True)
                    st.write(f"Shape: {processed_data.shape}")
                
                # Data summary
                st.subheader("Data Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Total Rows", len(processed_data))
                
                with summary_col2:
                    st.metric("Total Columns", len(processed_data.columns))
                
                with summary_col3:
                    st.metric("Missing Values", processed_data.isnull().sum().sum())
                
                # Column information
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': processed_data.columns,
                    'Type': processed_data.dtypes,
                    'Non-Null Count': processed_data.count(),
                    'Unique Values': processed_data.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        explainer_file = st.file_uploader(
            "Upload a column-definitions JSON (optional)",
            type=["json"],
            help="Maps each column to a friendly description"
        )

        if explainer_file is not None:
            try:
                st.session_state.column_definitions = json.load(explainer_file)
                st.success("Data-explainer loaded!")
            except Exception as e:
                st.error(f"Could not read JSON - {e}")

        
        if st.button("Use Sample Data"):
            try:
                sample_data = self.engine.load_sample_data()
                st.session_state.processed_data = sample_data
                st.session_state.data_loaded = True
                st.success("Sample data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    
    def query_analysis_page(self):
        st.header(" Query Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload data first in the Data Upload page.")
            return
        
        data = st.session_state.processed_data
        
        st.subheader("Enter Your Business Question")
        query = st.text_area(
            "Query",
            placeholder="e.g., What is the effect of square footage on house price?",
            height=100,
            help="Ask a business question about causal relationships in your data"
        )
        
        # Display available columns
        with st.expander("Available Columns"):
            cols = st.columns(3)
            for i, col in enumerate(data.columns):
                with cols[i % 3]:
                    st.write(f"• {col}")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=0.90,
                step=0.01
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ["linear", "forest"],
                help="Choose the underlying model for analysis"
            )
        
        # Run analysis
        if st.button(" Run Analysis", type="primary"):
            if not query.strip():
                st.error("Please enter a query.")
                return
            
            with st.spinner("Analyzing your query..."):
                try:
                    # Run the causal analysis workflow
                    results = self.engine.run_analysis(
                        query=query,
                        data=data,
                        confidence_level=confidence_level,
                        model_type=model_type,
                        column_definitions=st.session_state.column_definitions
                    )
                    
                    st.session_state.analysis_results = results
                    st.session_state.current_query = query
                    
                    st.success("Analysis completed successfully!")
                    
                    # Display quick results
                    if results and 'causal_estimate' in results:
                        st.subheader("Quick Results")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric(
                                "Causal Effect",
                                f"{results['causal_estimate']:.4f}",
                                help="The estimated causal effect of treatment on outcome"
                            )
                        
                        with result_col2:
                            validation = results.get('validation_results', {})
                            sample_size = validation.get('sample_size', 0)
                            st.metric("Sample Size", sample_size)
                        
                        with result_col3:
                            model_score = validation.get('model_score', 0)
                            st.metric("Model Score", f"{model_score:.4f}")
                        
                        st.info("Go to Results Dashboard for detailed analysis.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    def results_dashboard_page(self):
        st.header("📊 Results Dashboard")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis results available. Please run an analysis first.")
            return
        
        results = st.session_state.analysis_results
        query = st.session_state.get('current_query', '')
        
        # Display query
        st.subheader("Analysis Query")
        st.info(f"**Query:** {query}")
        
        # Main results
        st.subheader("Causal Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        validation_results = results.get('validation_results', {})
        
        with col1:
            st.metric(
                "Causal Effect (ATE)",
                f"{results.get('causal_estimate', 0):.4f}",
                help="Average Treatment Effect"
            )
        
        with col2:
            ci = validation_results.get('ate_confidence_interval', {})
            ci_text = f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]"
            st.metric("Confidence Interval", ci_text)
        
        with col3:
            st.metric(
                "Sample Size",
                validation_results.get('sample_size', 0)
            )
        
        with col4:
            st.metric(
                "Model Score",
                f"{validation_results.get('model_score', 0):.4f}"
            )
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Feature importance chart
        feature_importance = validation_results.get('feature_importance', {})
        if feature_importance:
            fig_importance = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                title="Feature Importance",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Scenario analysis
        scenario_analysis = validation_results.get('scenario_analysis', {})
        if scenario_analysis:
            st.subheader("Scenario Analysis")
            
            scenario_data = []
            for scenario, values in scenario_analysis.items():
                scenario_data.append({
                    'Scenario': scenario,
                    'New Treatment Value': values.get('new_treatment_value', 0),
                    'Predicted Outcome': values.get('predicted_outcome_mean', 0),
                    'Outcome Change (%)': values.get('outcome_change_percent', 0)
                })
            
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df, use_container_width=True)
            
            # Scenario visualization
            fig_scenario = px.bar(
                scenario_df,
                x='Scenario',
                y='Outcome Change (%)',
                title="Outcome Change by Scenario"
            )
            st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Detailed insights
        st.subheader("Detailed Insights")
        insights = results.get('insights', '')
        if insights:
            st.markdown(insights)
        else:
            st.info("No detailed insights available.")
        
        # Statistical tests
        statistical_tests = validation_results.get('statistical_tests', {})
        if statistical_tests:
            st.subheader("Statistical Tests")
            
            for test_name, test_results in statistical_tests.items():
                st.write(f"**{test_name.replace('_', ' ').title()}:**")
                
                test_cols = st.columns(len(test_results))
                for i, (key, value) in enumerate(test_results.items()):
                    with test_cols[i]:
                        st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
        
        # Export results
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Results as JSON"):
                results_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name="causal_analysis_results.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Download Summary as CSV"):
                summary_data = {
                    'Query': [query],
                    'Treatment': [results.get('treatment', '')],
                    'Outcome': [results.get('outcome', '')],
                    'Causal_Effect': [results.get('causal_estimate', 0)],
                    'Sample_Size': [validation_results.get('sample_size', 0)],
                    'Model_Score': [validation_results.get('model_score', 0)]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="causal_analysis_summary.csv",
                    mime="text/csv"
                )
    
    def advanced_settings_page(self):
        st.header("Advanced Settings")
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Primary LLM",
                ["mistralai/Mistral-7B-Instruct-v0.2", "deepseek/deepseek-r1-0528:free"],
                help="Choose the primary language model for analysis"
            )
        
        with col2:
            st.selectbox(
                "Secondary LLM",
                ["gemini-2.5-pro", "gpt-4"],
                help="Choose the secondary language model for insight generation"
            )
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.number_input(
                "Random Forest Estimators",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
        
        with col2:
            st.number_input(
                "Ridge Alpha",
                min_value=0.001,
                max_value=10.0,
                value=0.1,
                step=0.001,
                format="%.3f"
            )
        
        with col3:
            st.number_input(
                "Max Control Variables",
                min_value=1,
                max_value=10,
                value=5,
                step=1
            )
        
        # API Configuration
        st.subheader("API Configuration")
        
        with st.expander("API Keys"):
            st.text_input("Together API Key", type="password", help="API key for Together AI")
            st.text_input("OpenAI API Key", type="password", help="API key for OpenAI")
            st.text_input("Google API Key", type="password", help="API key for Google AI")
        
        # Cache settings
        st.subheader("Cache Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable LLM Caching", value=True)
        
        with col2:
            if st.button("Clear Cache"):
                st.success("Cache cleared successfully!")

# Run the app
if __name__ == "__main__":
    app = StreamlitCausalApp()
    app.run()
