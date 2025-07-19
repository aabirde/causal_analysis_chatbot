

from langgraph.graph import StateGraph, END
from causal_analysis_state import CausalAnalysisState
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from dowhy import CausalModel
from econml.dml import LinearDML
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


from llm_data import llm1, llm4  

def extract_quantitative_changes(query):
    """Extract quantitative changes from query text"""
    percentage_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(increase|decrease|change|more|less)'
    absolute_pattern = r'(\d+(?:\.\d+)?)\s*(more|less|increase|decrease|additional|extra)'
    multiplier_pattern = r'(\d+(?:\.\d+)?)\s*(?:times|x)\s*(more|higher|greater|larger)'
    
    changes = {
        "percentage_changes": [],
        "absolute_changes": [],
        "multiplier_changes": [],
        "has_quantitative_target": False
    }
    
    percentage_matches = re.findall(percentage_pattern, query.lower())
    for value, direction in percentage_matches:
        changes["percentage_changes"].append({
            "value": float(value),
            "direction": direction,
            "type": "percentage"
        })
        changes["has_quantitative_target"] = True
    
    absolute_matches = re.findall(absolute_pattern, query.lower())
    for value, direction in absolute_matches:
        changes["absolute_changes"].append({
            "value": float(value),
            "direction": direction,
            "type": "absolute"
        })
        changes["has_quantitative_target"] = True
    
    multiplier_matches = re.findall(multiplier_pattern, query.lower())
    for value, direction in multiplier_matches:
        changes["multiplier_changes"].append({
            "value": float(value),
            "direction": direction,
            "type": "multiplier"
        })
        changes["has_quantitative_target"] = True
    
    return changes

def business_case_validation_agent(state: CausalAnalysisState) -> CausalAnalysisState:
    """Agent to validate business case and extract variables"""
    try:
        column_definitions = state.get("column_definitions") or {}
        

        if not column_definitions:
            print("No data-explainer found. Using column names for analysis.")
            data_columns = state["data"].columns.tolist()
            column_definitions = {col: col.replace('_', ' ').title() for col in data_columns}
        quantitative_changes = extract_quantitative_changes(state["query"])
        

        limited_definitions = {}
        for key, value in list(column_definitions.items())[:100]:
            limited_definitions[key] = value[:200] if isinstance(value, str) else value
        column_names = list(limited_definitions.keys())
        
        # Variable extraction prompt
        extraction_prompt = ChatPromptTemplate.from_template("""
Identify variables from this query and column descriptions:

Query: {query}  
Quantitative Changes: {quantitative_changes}  
Columns: {column_definitions}
Respond in this exact format, with controls being strictly chosen from the columns. Treatment and outcome should be the most relevant variables based on the {query} and {column_definitions}. If no suitable outcome can be found, return "NONE" for outcome. If no suitable treatment can be found, but an outcome is found, then take the closest variable to the outcome as treatment. The variable names should strictly match the column names in the dataset, and synonyms can be used to achieve this objective. Do not return any variable that is not in the dataset. Also approximate variables to the closest match for example, sales becomes revenue.Use synonyms or similar terms to match the dataset columns. Use a business perspective, with revenue being quite important.
Only take from these columns: {column_names}.                                                      
Do not give any explanation or reasoning, just return the variables in the format below:
Treatment: exact_column_name
Outcome: exact_column_name
Controls: column1, column2, column3, column4, column5
                                                             
Strictly adhere to the format above.
""")
        
        extraction_chain = extraction_prompt | llm1
        
        extraction_response = extraction_chain.invoke({
            "query": state["query"][:300],
            "quantitative_changes": str(quantitative_changes)[:200],
            "column_definitions": str(limited_definitions)[:1000],
            "column_names": str(column_names)[:1000]
        })
        
        response_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
        
        # Parse response
        treatment = re.search(r'Treatment:\s*([^\n]+)', response_text)
        outcome = re.search(r'Outcome:\s*([^\n]+)', response_text)
        controls = re.search(r'Controls:\s*([^\n]+)', response_text)
        
        extracted_treatment = treatment.group(1).strip() if treatment else "NONE"
        extracted_outcome = outcome.group(1).strip() if outcome else "NONE"
        extracted_controls = [c.strip().strip("'\"") for c in controls.group(1).split(',')] if controls and controls.group(1) != "NONE" else []
        
        if extracted_treatment == "NONE" or extracted_outcome == "NONE":
            print("Could not extract treatment and outcome.")
            return {**state, "next_action": "complete"}
        
        return {
            **state,
            "treatment": extracted_treatment,
            "outcome": extracted_outcome,
            "controls": extracted_controls[:5],
            "quantitative_changes": quantitative_changes,
            "query_validation": {"is_valid": True},
            "next_action": "causal_analysis"
        }
        
    except Exception as e:
        print(f"Error in variable extraction: {str(e)}")
        return {**state, "next_action": "complete"}

def causal_pipeline_agent(state: CausalAnalysisState) -> CausalAnalysisState:
    """Agent to perform causal analysis"""
    print("Running causal analysis...")
    
    # Validate variables exist in data
    available_columns = list(state["data"].columns)
    missing_vars = []
    
    if state['treatment'] not in available_columns:
        missing_vars.append(f"Treatment: {state['treatment']}")
    
    if state['outcome'] not in available_columns:
        missing_vars.append(f"Outcome: {state['outcome']}")
    
    for control in state['controls']:
        if control not in available_columns:
            missing_vars.append(f"Control: {control}")
    
    if missing_vars:
        error_msg = f"Missing variables in dataset: {', '.join(missing_vars)}"
        print(error_msg)
        return {**state, "insights": f"Analysis Error: {error_msg}", "next_action": "complete"}
    
    # Perform causal analysis
    try:
        df = state["data"][[state["treatment"], state["outcome"]] + state["controls"]].dropna()
        
        if df.empty:
            return {**state, "insights": "No valid data after filtering", "next_action": "complete"}
        
        # Causal model
        causal_model = CausalModel(
            data=df,
            treatment=state["treatment"],
            outcome=state["outcome"],
            common_causes=state["controls"]
        )
        
        identified_estimand = causal_model.identify_effect()
        
        # EconML model
        econ_model = LinearDML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestRegressor(),
            discrete_treatment=False
        )
        
        econ_model.fit(
            Y=df[state["outcome"]],
            T=df[state["treatment"]],
            X=df[state["controls"]]
        )
        
        ate = econ_model.ate(X=df[state["controls"]])
        confidence_interval = econ_model.ate_interval(X=df[state["controls"]], alpha=0.1)
        
        # Additional analysis
        scaler = StandardScaler()
        X = df[[state["treatment"]] + state["controls"]]
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor()
        model.fit(X_scaled, df[state["outcome"]])
        
        feature_importance = dict(zip([state["treatment"]] + state["controls"], model.feature_importances_))
        
        # Compile results
        validation_results = {
            "ate": float(ate[0]) if hasattr(ate, '__len__') else float(ate),
            "ate_confidence_interval": {
                "lower": float(confidence_interval[0][0]) if hasattr(confidence_interval[0], '__len__') else float(confidence_interval[0]),
                "upper": float(confidence_interval[1][0]) if hasattr(confidence_interval[1], '__len__') else float(confidence_interval[1])
            },
            "feature_importance": feature_importance,
            "sample_size": len(df),
            "model_score": float(model.score(X_scaled, df[state["outcome"]])),
            "baseline_stats": {
                "treatment_mean": float(df[state["treatment"]].mean()),
                "outcome_mean": float(df[state["outcome"]].mean()),
                "treatment_std": float(df[state["treatment"]].std()),
                "outcome_std": float(df[state["outcome"]].std())
            }
        }
        
        causal_estimate = validation_results["ate"]
        
        return {
            **state,
            "causal_estimate": causal_estimate,
            "validation_results": validation_results,
            "next_action": "insight_generation"
        }
        
    except Exception as e:
        print(f"Error in causal analysis: {str(e)}")
        return {**state, "insights": f"Analysis Error: {str(e)}", "next_action": "complete"}

def insight_generation_agent(state: CausalAnalysisState) -> CausalAnalysisState:
    """Agent to generate business insights"""
    print("Generating insights...")
    
    insight_prompt = ChatPromptTemplate.from_template("""
You are acting as a dedicated sales analyst XYZ, providing thoughtful, data-driven insights to clients such as investors, developers, or homeowners. 
Your job is to not only break down the numbers but also to help the client feel confident in the decisions they're making—backed by both statistical rigor and business practicality.
Keep a very natural, conversational tone, as if you're explaining the analysis to a friend or colleague who is not a data expert. 
Also, keep the insight organised and structured, so the client can easily follow along and understand the key points.

Client Query
Query: {query}
This section is where we address your key business question—what's driving a particular market behavior or trend. This could relate to pricing, demand shifts, renovation decisions, or development feasibility. Throughout our analysis, we'll keep referring back to this core question to ensure our insights remain focused on your main concern.

Treatment & Outcome Overview
Treatment Variable: {treatment}
This is the factor we're testing as a potential cause for change—something you may be able to control or adjust, such as a property feature (number of bedrooms, square footage, or a location attribute).

Outcome Variable: {outcome}
This represents your goal—what you're aiming to understand or improve. It could be maximizing sale price, reducing days on market, increasing rental income, or enhancing marketability.

Control Variables: {controls}
These are other important variables we've accounted for to ensure our insights about the treatment are reliable. By controlling for these, we can isolate the true effect of the treatment and avoid misleading conclusions.

Causal Analysis Interpretation


Cofactor Analysis: {cofactor_analysis}
Here, we examine whether any control variables interact with the treatment to strengthen or weaken its effect. For instance, the impact of adding a bedroom might be more substantial in urban areas than rural ones, or in newer homes compared to older builds.


Predictive Modeling Support
Feature Importance: {feature_importance}
This shows how influential the treatment variable is in predicting the outcome, even without assuming causality. If {treatment} ranks high, it supports our earlier findings and suggests it's a variable worth prioritizing.


Baseline Stats: {baseline_stats}
These provide market benchmarks for comparison. Knowing averages for square footage, bathroom count, or listing price can guide whether a property is above or below market norms.

Regression Stability: {regression_stability}
This shows whether our conclusions hold across different model specifications. A stable regression suggests that, no matter how we analyze the data, the core insight remains true—giving you higher confidence in your decisions.
                                                      
Control Variable Insights
Control Effects: {control_effects}
Control variables can play a hidden but crucial role. Here, we assess how much each control contributes to the outcome, with all output for controls being numerical for actionable insights. For example, if the number of bathrooms independently increases price by $x while square footage contributes $y per sq ft, that tells us where the real leverage lies. Sometimes, even though a treatment shows an effect, the control variables are doing most of the work in reality. Understanding this helps you decide whether to invest in more space, more amenities, or just better staging.

Also, consider interaction effects from the cofactor analysis. If the impact of adding space is much larger in premium neighborhoods or during peak seasons, those nuances should guide your timing and target audience.

To provide you with comprehensive insights, we'll analyze your data through four distinct analytical lenses. Use one of these tiers.:

Tier 1: Optimization Analysis
Question Focus: What is the optimal level or configuration of the treatment variable to maximize the outcome?

This tier identifies the sweet spot for your treatment variable. We'll determine the optimal range, threshold effects, and diminishing returns patterns. For example, if square footage is your treatment, we'll find the optimal size that maximizes price per square foot, or identify when additional space stops adding proportional value.

Tier 2: Root Cause Analysis (RCA)
Question Focus: What are the underlying drivers and mechanisms that explain why the treatment affects the outcome?

Here, we dig deeper into the "why" behind the relationship. We'll examine mediating factors, identify confounding variables, and explore the causal pathways. This helps you understand not just that something works, but why it works, giving you confidence in applying insights to new situations. Follow the five whys technique to peel back layers of complexity and get to the core drivers.

Tier 3: Forecasting Analysis
Question Focus: How will the treatment-outcome relationship evolve over time, and what future scenarios should we prepare for?

This tier projects how the current relationship might change due to market trends, seasonal patterns, or economic cycles. We'll model different future scenarios and their implications for your strategy, helping you make decisions that remain robust over time.

Tier 4: Intervention Analysis (Direct Treatment to Outcome)
Question Focus: What specific interventions or changes to the treatment variable will yield the highest return on investment?

This is where we get tactical. We'll identify the most cost-effective ways to modify your treatment variable, calculate expected returns for different intervention strategies, and prioritize actions based on effort-to-impact ratios. Also mention which tier question is being answered in the analysis.
Answers to Client Query
Now, let's directly address your query with clear, actionable insights. We'll summarize the key findings and how they relate to your business case, ensuring you have a solid understanding of the data and its implications for your property or investment strategy.
Tailored Business Recommendations
Based on the combined causal and predictive findings, we'll now discuss actionable business recommendations. Every suggestion is grounded in the numbers—no vagueness. Each recommendation will be framed with expected monetary or percentage gains, so you can make informed, confident decisions about your property or investment strategy. Give 5 business recommendations based on the analysis, focusing on how to leverage the treatment variable to achieve the desired outcome, and all should be numerically backed with data from the analysis. It should be quantitiative.

Also, give a conclusion, grounded on quantitative outputs. Give me a very detailed insight
""")
    
    chain = insight_prompt | llm4
    validation_results = state["validation_results"]    
    insights = chain.invoke({
        "query": state["query"][:500],
        "treatment": state["treatment"],
        "outcome": state["outcome"],
        "controls": str(state["controls"])[:200],
        "cofactor_analysis": str(validation_results.get("cofactor_analysis", {}))[:200],
        "baseline_stats": str(validation_results.get("baseline_stats", {}))[:300],
        "feature_importance": str(validation_results.get("feature_importance", {}))[:200],
        "control_effects": str(validation_results.get("control_effects", {}))[:300],
        "regression_stability": str(validation_results.get("regression_stability", {}))[:200]
    })
    
    insights_text = insights.content if hasattr(insights, 'content') else str(insights)
    
    return {
        **state,
        "insights": insights_text,
        "next_action": "complete"
    }

def create_causal_workflow():
    """Create and configure the causal analysis workflow"""
    workflow = StateGraph(CausalAnalysisState)
    
    # Add nodes
    workflow.add_node("business_case_validator", business_case_validation_agent)
    workflow.add_node("causal_pipeline", causal_pipeline_agent)
    workflow.add_node("insight_generator", insight_generation_agent)
    
    # Add edges
    workflow.add_conditional_edges(
        "business_case_validator",
        lambda x: x.get("next_action"),
        {
            "causal_analysis": "causal_pipeline",
            "complete": END
        }
    )
    
    workflow.add_conditional_edges(
        "causal_pipeline",
        lambda x: x.get("next_action"),
        {
            "insight_generation": "insight_generator",
            "complete": END
        }
    )
    
    workflow.add_edge("insight_generator", END)
    workflow.set_entry_point("business_case_validator")
    
    return workflow.compile()
