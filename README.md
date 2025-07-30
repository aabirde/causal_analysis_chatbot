# Causal Analysis Chatbot

This project is a web-based application that allows users to perform causal analysis on their data. It provides a user-friendly interface to upload data, ask business questions in natural language, and receive detailed causal insights and visualizations. This project has compatibility in both Streamlit and Flask

##  Features

* **CSV Data Upload**: Users can upload their own data in CSV format.
* **Sample Data**: A sample dataset is provided for demonstration purposes.
* **Natural Language Queries**: Users can ask causal questions in plain English (e.g., "What is the effect of ad spend on revenue?").
* **Causal Analysis**: The backend performs causal analysis using statistical models to determine the causal effect of a treatment variable on an outcome variable.
* **Detailed Results Dashboard**: The results are presented in a comprehensive dashboard with key metrics, visualizations, and detailed insights.
* **Insight Generation**: The application generates business-oriented insights and recommendations based on the analysis.
* **Advanced Settings**: Users can configure advanced parameters for the analysis.
* **Export Results**: Results can be exported in JSON and CSV formats.

##  How it Works

1.  **Data Upload**: The user uploads a CSV file or chooses to use the sample data. The data is preprocessed to handle missing values and encode categorical variables.
2.  **Query Analysis**: The user asks a business question. The application's AI identifies the treatment, outcome, and control variables from the query and the data.
3.  **Causal Analysis**: A causal model is built to estimate the effect of the treatment on the outcome while controlling for other variables.
4.  **Insight Generation**: The results of the causal analysis are used to generate detailed insights and business recommendations.
5.  **Results Dashboard**: The results are displayed in a user-friendly dashboard with charts and tables.

##  Getting Started

### Prerequisites

* Python 3.7+
* Flask
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Langchain
* DoWhy
* EconML

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/aabirde/causal_analysis_chatbot.git](https://github.com/aabirde/causal_analysis_chatbot.git)
    cd causal_analysis_chatbot
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application in Flask

1.  **Start the Flask server:**
    ```bash
    python flask_app.py
    ```

2.  **Open your web browser and navigate to:**
    ```
    [http://127.0.0.1:5000](http://127.0.0.1:5000)
    ```

 ### Running the Application in Flask

1.  **Start the Streamlit server:**
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Open your web browser and navigate to:**
    ```
    (http://localhost:8501)
    ```

##  Pages

1.  **Upload Data**: Go to the "Data Upload" page and upload your CSV file, or click "Use Sample Data".
2.  **Analyze**: Go to the "Query Analysis" page and enter a business question in the text area.
3.  **View Results**: Once the analysis is complete, you will be redirected to the "Results Dashboard" to view the detailed insights.
4.  **Settings**: You can configure advanced settings on the "Advanced Settings" page.

##  File Structure

```
.
├── causal_analysis_state.py
├── causal_engine.py
├── causal_workflow.py
├── data
│   ├── dataexplainer.json
│   └── sample_data.csv
├── flask_app.py
├── llm_data.py
├── README.md
├── streamlit_app.py
├── templates
│   ├── base.html
│   ├── index.html
│   ├── query.html
│   ├── results.html
│   ├── settings.html
│   └── upload.html
└── utils.py
```

* `flask_app.py`: The main Flask application file.
* `streamlit_app.py`: The main Streamlit application file.
* `causal_engine.py`: Contains the core logic for the causal analysis.
* `causal_workflow.py`: Defines the causal analysis workflow using state graphs.
* `data/`: Contains the sample data and data dictionary.
* `templates/`: Contains the HTML templates for the web interface.
* `utils.py`: Contains utility functions for formatting results and creating visualizations.
