from typing import TypedDict, List, Any
import pandas as pd

class CausalAnalysisState(TypedDict):
    query: str
    data: pd.DataFrame
    treatment: str
    outcome: str
    controls: List[str]
    causal_estimate: float
    insights: str
    validation_results: dict
    next_action: str
    business_cases: List[dict]
    query_validation: dict
    matched_case: dict
    quantitative_changes: dict
    column_definitions: dict[str, Any]
