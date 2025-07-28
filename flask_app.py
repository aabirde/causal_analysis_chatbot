from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, flash, make_response
from flask_session import Session 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import os
import io
import base64
from werkzeug.utils import secure_filename
from causal_engine import CausalAnalysisEngine
from utils import format_results, create_visualizations
import warnings
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize causal engine
causal_engine = CausalAnalysisEngine()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Data upload page"""
    return render_template('upload.html')

@app.route('/query')
def query_page():
    """Query analysis page"""
    if 'data_loaded' not in session or not session['data_loaded']:
        flash('Please upload data first before running analysis.', 'warning')
        return redirect(url_for('upload_page'))
    
    # Get column information for display
    data_info = session.get('data_info', {})
    return render_template('query.html', data_info=data_info)

@app.route('/results')
def results_page():
    """Results dashboard page"""
    if 'analysis_results' not in session:
        flash('No analysis results available. Please run an analysis first.', 'warning')
        return redirect(url_for('query_page'))
    
    try:
        results = session['analysis_results']
        query = session.get('current_query', '')
        
        # Create visualizations
        plots_json = {}
        try:
            figs = create_visualizations(results)
            for key, fig in figs.items():
                plots_json[key] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
            # Continue without plots
        
        return render_template('results.html', 
                             results=results, 
                             query=query, 
                             plots=plots_json)
    except Exception as e:
        print(f"Results page error: {e}")
        flash(f'Error displaying results: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/settings')
def settings_page():
    """Advanced settings page"""
    return render_template('settings.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        print("Upload file request received")  # Debug log
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved to: {filepath}")  # Debug log
            
            # Load and preprocess data
            data = pd.read_csv(filepath)
            print(f"Raw data shape: {data.shape}")  # Debug log
            
            processed_data = causal_engine.load_and_preprocess_data(data)
            print(f"Processed data shape: {processed_data.shape}")  # Debug log
            
            # Store data info in session
            session['data_loaded'] = True
            session['raw_data_shape'] = data.shape
            session['processed_data_shape'] = processed_data.shape
            session['data_columns'] = processed_data.columns.tolist()
            session['missing_values'] = int(processed_data.isnull().sum().sum())
            
            # Create column info
            col_info = {
                'columns': processed_data.columns.tolist(),
                'dtypes': processed_data.dtypes.astype(str).to_dict(),
                'non_null_counts': processed_data.count().to_dict(),
                'unique_values': processed_data.nunique().to_dict()
            }
            session['data_info'] = col_info
            
            # Save processed data temporarily (in production, use database)
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            processed_data.to_csv(processed_filepath, index=False)
            session['processed_data_path'] = processed_filepath
            
            print("Data upload completed successfully")  # Debug log
            
            return jsonify({
                'success': True,
                'message': 'Data uploaded and processed successfully',
                'data_info': {
                    'raw_shape': data.shape,
                    'processed_shape': processed_data.shape,
                    'columns': len(processed_data.columns),
                    'missing_values': int(processed_data.isnull().sum().sum())
                }
            })
        
        return jsonify({'error': 'Invalid file format. Please upload CSV files only.'}), 400
        
    except Exception as e:
        print(f"Upload error: {e}")  # Debug log
        print(traceback.format_exc())  # Full traceback
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/upload_explainer', methods=['POST'])
def upload_explainer():
    """Handle column definitions JSON upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.json'):
            column_definitions = json.load(file)
            session['column_definitions'] = column_definitions
            
            return jsonify({
                'success': True,
                'message': 'Column definitions loaded successfully',
                'columns_count': len(column_definitions)
            })
        
        return jsonify({'error': 'Invalid file format. Please upload JSON files only.'}), 400
        
    except Exception as e:
        print(f"Explainer upload error: {e}")  # Debug log
        return jsonify({'error': f'Error processing explainer file: {str(e)}'}), 500

@app.route('/api/load_sample')
def load_sample_data():
    """Load sample data"""
    try:
        print("Loading sample data...")  # Debug log
        
        sample_data = causal_engine.load_sample_data()
        print(f"Sample data loaded with shape: {sample_data.shape}")  # Debug log
        
        # Store data info in session
        session['data_loaded'] = True
        session['processed_data_shape'] = sample_data.shape
        session['data_columns'] = sample_data.columns.tolist()
        session['missing_values'] = int(sample_data.isnull().sum().sum())
        
        # Create column info
        col_info = {
            'columns': sample_data.columns.tolist(),
            'dtypes': sample_data.dtypes.astype(str).to_dict(),
            'non_null_counts': sample_data.count().to_dict(),
            'unique_values': sample_data.nunique().to_dict()
        }
        session['data_info'] = col_info
        
        # Save sample data temporarily
        sample_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_data.csv')
        sample_data.to_csv(sample_filepath, index=False)
        session['processed_data_path'] = sample_filepath
        
        print("Sample data loading completed")  # Debug log
        
        return jsonify({
            'success': True,
            'message': 'Sample data loaded successfully',
            'data_info': {
                'shape': sample_data.shape,
                'columns': len(sample_data.columns),
                'missing_values': int(sample_data.isnull().sum().sum())
            }
        })
        
    except Exception as e:
        print(f"Sample data error: {e}")  # Debug log
        print(traceback.format_exc())  # Full traceback
        return jsonify({'error': f'Error loading sample data: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def run_analysis():
    """Run causal analysis"""
    try:
        print("Analysis request received")  # Debug log
        
        # Check if request has JSON data
        if not request.is_json:
            print("Request is not JSON")  # Debug log
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        print(f"Request data: {data}")  # Debug log
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        query = data.get('query', '').strip()
        confidence_level = data.get('confidence_level', 0.90)
        model_type = data.get('model_type', 'linear')
        
        print(f"Parsed query: '{query}'")  # Debug log
        print(f"Confidence level: {confidence_level}")  # Debug log
        print(f"Model type: {model_type}")  # Debug log
        
        if not query:
            return jsonify({'error': 'Please enter a query'}), 400
        
        if len(query) < 5:
            return jsonify({'error': 'Query too short. Please provide more details.'}), 400
        
        if 'processed_data_path' not in session:
            return jsonify({'error': 'No data loaded. Please upload data first.'}), 400
        
        # Check if data file exists
        data_path = session['processed_data_path']
        if not os.path.exists(data_path):
            return jsonify({'error': 'Data file not found. Please upload data again.'}), 400
        
        print(f"Loading data from: {data_path}")  # Debug log
        
        # Load processed data
        processed_data = pd.read_csv(data_path)
        column_definitions = session.get('column_definitions', {})
        
        print(f"Data loaded with shape: {processed_data.shape}")  # Debug log
        print(f"Column definitions: {len(column_definitions)} items")  # Debug log
        
        print("Starting causal analysis...")  # Debug log
        
        # Run analysis
        results = causal_engine.run_analysis(
            query=query,
            data=processed_data,
            confidence_level=confidence_level,
            model_type=model_type,
            column_definitions=column_definitions
        )
        
        print("Analysis completed successfully")  # Debug log
        print(f"Results keys: {list(results.keys()) if results else 'None'}")  # Debug log
        session['analysis_results'] = results
        session['current_query'] = query
        
        # Prepare response
        response_data = {
            'success': True,
            'message': 'Analysis completed successfully',
            'results': {
                'causal_effect': results.get('causal_estimate', 0),
                'sample_size': results.get('validation_results', {}).get('sample_size', 0),
                'model_score': results.get('validation_results', {}).get('model_score', 0)
            }
        }
        
        print(f"Response data: {response_data}")  # Debug log
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Analysis error: {e}")  # Debug log
        print(traceback.format_exc())  # Full traceback
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/export_results')
def export_results():
    """Export analysis results as JSON"""
    if 'analysis_results' not in session:
        return jsonify({'error': 'No results to export'}), 400
    
    results = session['analysis_results']
    
    # Create a downloadable JSON file
    output = io.StringIO()
    json.dump(results, output, indent=2, default=str)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name='causal_analysis_results.json',
        mimetype='application/json'
    )

@app.route('/api/export_summary')
def export_summary():
    """Export analysis summary as CSV"""
    if 'analysis_results' not in session:
        return jsonify({'error': 'No results to export'}), 400
    results = session['analysis_results']
    validation_results = results.get('validation_results', {})
    query = session.get('current_query', '')
    
    summary_data = {
        'Query': [query],
        'Treatment': [results.get('treatment', '')],
        'Outcome': [results.get('outcome', '')],
        'Causal_Effect': [results.get('causal_estimate', 0)],
        'Sample_Size': [validation_results.get('sample_size', 0)],
        'Model_Score': [validation_results.get('model_score', 0)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    output = io.StringIO()
    summary_df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        as_attachment=True,
        download_name='causal_analysis_summary.csv',
        mimetype='text/csv'
    )

@app.route('/api/clear_session')
def clear_session():
    """Clear session data"""
    try:
        # Clean up uploaded files
        if 'processed_data_path' in session:
            file_path = session['processed_data_path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        session.clear()
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    except Exception as e:
        print(f"Clear session error: {e}")
        return jsonify({'error': f'Error clearing session: {str(e)}'}), 500

# Debug route to check session data
@app.route('/debug/session')
def debug_session():
    """Debug route to check session data"""
    if app.debug:
        return jsonify({
            'session_keys': list(session.keys()),
            'data_loaded': session.get('data_loaded', False),
            'has_data_path': 'processed_data_path' in session,
            'has_analysis_results': 'analysis_results' in session,
            'data_columns_count': len(session.get('data_columns', [])),
        })
    else:
        return jsonify({'error': 'Debug mode only'}), 403

if __name__ == '__main__':
    print("Starting Flask application in debug mode...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5000)