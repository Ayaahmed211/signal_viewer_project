from dash import html, dcc, Input, Output, State, callback
import os
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyts.image import RecurrencePlot
from scipy.signal import butter, filtfilt, resample
import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.preprocessing import StandardScaler

# ----------------- Ensure uploads folder exists -----------------
os.makedirs("uploads", exist_ok=True)

# ----------------- Load Model and Scaler -----------------
MODEL_PATH = os.path.join('models', 'ecg_classifier_model.keras')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Global variables to store model and scaler
model = None
scaler = None

def load_classifier():
    """Load the trained model and scaler"""
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded successfully from {SCALER_PATH}")
        else:
            print(f"Warning: Scaler file not found at {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading model/scaler: {e}")

# Load model on module import
load_classifier()

# ----------------- Layout -----------------
layout = html.Div(
    style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif', 'padding': '30px'},
    children=[
        html.H1("ECG Signal Viewer (CSV)", 
                style={'color':'#00d9ff','fontSize':'48px','marginBottom':'20px', 'textAlign': 'center'}),

        # Upload Section
        html.Div(
            style={'maxWidth': '1200px', 'margin': '0 auto', 'marginBottom': '30px'},
            children=[
                dcc.Upload(
                    id='upload-ecg',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select CSV File', style={'color': '#00d9ff', 'fontWeight': 'bold'})
                    ]),
                    style={
                        'width': '100%', 'height': '100px', 'lineHeight': '100px',
                        'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                        'textAlign': 'center', 'backgroundColor': '#16213e', 'color': '#00d9ff',
                        'cursor': 'pointer', 'border': '2px dashed #00d9ff'
                    },
                    multiple=False,
                    accept='.csv'
                ),
                html.Div(id='file-list', style={'marginTop': '15px', 'fontSize':'16px','color':'#00d9ff', 'textAlign': 'center'}),
                
                html.Button("Process ECG", id='process-btn', n_clicks=0, style={
                    'marginTop':'20px', 'backgroundColor': '#00d9ff', 'color': '#1a1a2e',
                    'borderRadius': '8px', 'padding': '12px 40px', 'fontSize':'16px',
                    'border':'none', 'cursor':'pointer', 'fontWeight': 'bold',
                    'display': 'block', 'margin': '20px auto'
                }),
            ]
        ),

        # Processing Status
        dcc.Loading(
            id="loading-ecg",
            type="circle",
            color="#00d9ff",
            children=[
                html.Div(id='ecg-output', style={'marginTop': '20px','fontSize':'18px','color':'#00d9ff', 'textAlign': 'center'}),
            ]
        ),

        # Main Content Container
        html.Div(
            id='main-content',
            style={'display': 'none', 'maxWidth': '1600px', 'margin': '0 auto'},
            children=[
                # Health Prediction Section (Top)
                html.Div(
                    id='prediction-section',
                    style={'marginBottom': '30px'},
                    children=[
                        html.Div(
                            id='prediction-card',
                            style={
                                'backgroundColor': '#16213e',
                                'borderRadius': '15px',
                                'padding': '30px',
                                'maxWidth': '700px',
                                'margin': '0 auto',
                                'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.3)',
                                'border': '2px solid #00d9ff'
                            },
                            children=[
                                html.H2("🏥 Health Prediction", style={'color': '#00d9ff', 'marginBottom': '20px', 'textAlign': 'center'}),
                                html.Div(id='prediction-result', style={'fontSize': '32px', 'fontWeight': 'bold', 'marginBottom': '15px', 'textAlign': 'center'}),
                                html.Div(id='prediction-confidence', style={'fontSize': '18px', 'color': '#aaa', 'marginBottom': '10px', 'textAlign': 'center'}),
                                html.Div(id='prediction-details', style={'fontSize': '16px', 'color': '#ccc', 'marginTop': '15px'})
                            ]
                        )
                    ]
                ),

                # Controls and Visualization Section
                html.Div(
                    style={'display': 'flex', 'gap': '20px', 'marginTop': '30px'},
                    children=[
                        # Left Panel - Controls
                        html.Div(
                            style={
                                'backgroundColor': '#16213e',
                                'borderRadius': '15px',
                                'padding': '25px',
                                'width': '300px',
                                'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.2)',
                                'border': '1px solid #00d9ff'
                            },
                            children=[
                                html.H3("ECG Controls", style={'color': '#00d9ff', 'marginBottom': '20px', 'fontSize': '24px'}),
                                
                                # Window Width Control
                                html.Label("Window width (seconds)", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Input(
                                    id='window-width',
                                    type='number',
                                    value=5,
                                    min=1,
                                    max=30,
                                    step=1,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'borderRadius': '8px',
                                        'border': '1px solid #00d9ff',
                                        'backgroundColor': '#0f1626',
                                        'color': '#00d9ff',
                                        'fontSize': '16px',
                                        'marginBottom': '20px'
                                    }
                                ),
                                
                                # Graph Mode Selection
                                html.Label("Graph Mode", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '8px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id='graph-mode',
                                    options=[
                                        {'label': 'ECG Waveform', 'value': 'waveform'},
                                        {'label': 'Polar Graph', 'value': 'polar'},
                                        {'label': 'Recurrence Plot', 'value': 'recurrence'}
                                    ],
                                    value='waveform',
                                    style={
                                        'marginBottom': '20px',
                                        'backgroundColor': '#0f1626',
                                        'color': '#1a1a2e',
                                    },
                                    className='custom-dropdown'
                                ),
                                
                                # Channel Selection
                                html.Label("Select Channels", style={'color': '#00d9ff', 'fontSize': '14px', 'marginBottom': '12px', 'display': 'block'}),
                                dcc.Checklist(
                                    id='channel-selection',
                                    options=[],  # Will be populated after processing
                                    value=[],
                                    style={'color': '#00d9ff', 'marginBottom': '20px'},
                                    labelStyle={'display': 'block', 'marginBottom': '8px', 'cursor': 'pointer'}
                                ),
                                
                                # Start/Stop Streaming Button
                                html.Button(
                                    "Start Streaming",
                                    id='stream-btn',
                                    n_clicks=0,
                                    style={
                                        'width': '100%',
                                        'backgroundColor': '#00d9ff',
                                        'color': '#1a1a2e',
                                        'borderRadius': '8px',
                                        'padding': '15px',
                                        'fontSize': '16px',
                                        'border': 'none',
                                        'cursor': 'pointer',
                                        'fontWeight': 'bold',
                                        'marginTop': '10px'
                                    }
                                ),
                            ]
                        ),
                        
                        # Right Panel - Visualization
                        html.Div(
                            style={'flex': '1'},
                            children=[
                                dcc.Graph(
                                    id='ecg-stream-graph',
                                    style={
                                        'height': '600px',
                                        'backgroundColor': '#16213e',
                                        'borderRadius': '15px',
                                        'boxShadow': '0 4px 20px rgba(0, 217, 255, 0.2)',
                                    },
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        )
                    ]
                ),
                
                # Interval component for streaming
                dcc.Interval(
                    id='interval-component',
                    interval=100,  # Update every 100ms
                    n_intervals=0,
                    disabled=True  # Start disabled
                ),
                
                # Store components
                dcc.Store(id='ecg-data-store'),
                dcc.Store(id='stream-state', data={'streaming': False, 'position': 0})
            ]
        ),

        html.Br(),
        html.Div(
            style={'textAlign': 'center', 'marginTop': '40px'},
            children=[
                dcc.Link(
                    html.Button("Back to Home", style={
                        'backgroundColor': '#00d9ff', 'color': '#1a1a2e', 'borderRadius': '8px',
                        'padding': '12px 40px', 'fontSize':'16px', 'border': 'none', 
                        'cursor': 'pointer', 'fontWeight': 'bold'
                    }), 
                    href='/'
                )
            ]
        )
    ]
)

# ----------------- ECG Functions -----------------
def bandpass(sig, fs, low=0.5, high=40, order=4):
    """Apply bandpass filter to ECG signal"""
    nyq = 0.5 * fs
    low_normalized = low / nyq
    high_normalized = high / nyq
    
    low_normalized = max(0.001, min(low_normalized, 0.999))
    high_normalized = max(low_normalized + 0.001, min(high_normalized, 0.999))
    
    b, a = butter(order, [low_normalized, high_normalized], btype='band')
    return filtfilt(b, a, sig)

def read_csv_file(csv_path):
    """Read CSV file with ECG data"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Expected columns: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
        expected_columns = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Check if expected columns exist
        available_columns = [col for col in expected_columns if col in df.columns]
        
        if not available_columns:
            raise Exception("CSV does not contain expected ECG lead columns (I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6)")
        
        # Extract signals
        signals = df[available_columns].values
        
        # Default sampling frequency (assume 250 Hz if not specified)
        fs = 250
        
        # Create a record-like object
        class CSVRecord:
            def __init__(self, signals, fs, lead_names):
                self.p_signal = signals
                self.fs = fs
                self.sig_name = lead_names
                self.n_sig = signals.shape[1]
        
        return CSVRecord(signals, fs, available_columns)
    
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")

def process_ecg_signals(record):
    """Process all ECG signals"""
    signals = record.p_signal
    fs = record.fs
    lead_names = record.sig_name
    
    processed_signals = []
    for i in range(signals.shape[1]):
        signal = signals[:, i]
        
        # Handle NaN and Inf values
        valid_indices = ~(np.isnan(signal) | np.isinf(signal))
        if not np.any(valid_indices):
            processed_signals.append(np.zeros_like(signal))
            continue
        if not np.all(valid_indices):
            median_val = np.median(signal[valid_indices])
            signal[~valid_indices] = median_val
        
        # Apply bandpass filter
        if np.std(signal) > 1e-6:
            try:
                signal_filtered = bandpass(signal, fs)
            except:
                signal_filtered = signal
        else:
            signal_filtered = signal
        
        processed_signals.append(signal_filtered)
    
    processed_signals = np.column_stack(processed_signals)
    
    # Resample to target frequency (250 Hz)
    target_fs = 250
    if fs != target_fs and fs > 0:
        try:
            n_samples = int(processed_signals.shape[0] * target_fs / fs)
            resampled_signals = []
            for i in range(processed_signals.shape[1]):
                resampled_sig = resample(processed_signals[:, i], n_samples)
                resampled_signals.append(resampled_sig)
            processed_signals = np.column_stack(resampled_signals)
            fs_final = target_fs
        except:
            fs_final = fs
    else:
        fs_final = fs
    
    return processed_signals, fs_final, lead_names

def predict_health_status(signal, fs):
    """Predict if ECG is healthy or diseased"""
    global model, scaler
    
    if model is None or scaler is None:
        return None, None, "Model not loaded"
    
    try:
        target_fs = 100
        target_samples = 1000
        
        # Resample to target frequency
        if fs != target_fs:
            n_samples = int(len(signal) * target_fs / fs)
            signal_resampled = resample(signal, n_samples)
        else:
            signal_resampled = signal
        
        # Adjust length to target samples
        if len(signal_resampled) > target_samples:
            start_idx = (len(signal_resampled) - target_samples) // 2
            signal_processed = signal_resampled[start_idx:start_idx + target_samples]
        elif len(signal_resampled) < target_samples:
            signal_processed = np.pad(signal_resampled, (0, target_samples - len(signal_resampled)), 'constant')
        else:
            signal_processed = signal_resampled
        
        # Scale and predict
        signal_scaled = scaler.transform(signal_processed.reshape(1, -1))
        signal_input = signal_scaled.reshape(1, target_samples, 1)
        prediction_proba = model.predict(signal_input, verbose=0)[0][0]
        prediction_class = 1 if prediction_proba > 0.5 else 0
        
        is_healthy = (prediction_class == 1)
        confidence = prediction_proba if is_healthy else (1 - prediction_proba)
        
        return is_healthy, confidence, None
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

def generate_rp_data(signal, max_samples=400):
    """Generate recurrence plot data"""
    try:
        if len(signal) > max_samples:
            start_idx = len(signal) // 4
            signal_subset = signal[start_idx:start_idx + max_samples]
        else:
            signal_subset = signal
        
        signal_mean = np.mean(signal_subset)
        signal_std = np.std(signal_subset)
        
        if signal_std > 1e-8:
            signal_normalized = (signal_subset - signal_mean) / signal_std
        else:
            signal_normalized = signal_subset - signal_mean
        
        signal_2d = signal_normalized.reshape(1, -1)
        rp = RecurrencePlot(dimension=2, time_delay=2, percentage=15)
        X_rp = rp.fit_transform(signal_2d)[0]
        
        return X_rp
    except Exception as e:
        print(f"Error generating RP: {e}")
        return np.zeros((100, 100))

# ----------------- Callbacks -----------------

@callback(
    Output('file-list', 'children'),
    Input('upload-ecg', 'contents'),
    State('upload-ecg', 'filename')
)
def show_file_list(contents, filename):
    if not contents or not filename:
        return "No files uploaded yet."
    return f"Uploaded file: {filename}"

@callback(
    [Output('ecg-data-store', 'data'),
     Output('ecg-output', 'children'),
     Output('main-content', 'style'),
     Output('channel-selection', 'options'),
     Output('channel-selection', 'value'),
     Output('prediction-result', 'children'),
     Output('prediction-confidence', 'children'),
     Output('prediction-details', 'children')],
    Input('process-btn', 'n_clicks'),
    [State('upload-ecg', 'contents'),
     State('upload-ecg', 'filename')]
)
def process_ecg(n_clicks, contents, filename):
    empty_outputs = (None, "", {'display':'none'}, [], [], "", "", "")
    
    if n_clicks == 0 or not contents or not filename:
        return empty_outputs
    
    # Check if it's a CSV file
    if not filename.endswith('.csv'):
        return (None, "❌ Please upload a CSV file", {'display':'none'}, [], [], "", "", "")
    
    # Clear uploads directory
    for file in os.listdir("uploads"):
        try:
            os.unlink(os.path.join("uploads", file))
        except:
            pass
    
    # Save file
    try:
        data = contents.split(',')[1] if ',' in contents else contents
        csv_path = os.path.join("uploads", filename)
        with open(csv_path, "wb") as f:
            f.write(base64.b64decode(data))
    except Exception as e:
        return (None, f"❌ Error saving file: {e}", {'display':'none'}, [], [], "", "", "")
    
    # Read and process CSV
    try:
        record = read_csv_file(csv_path)
        signals, fs, lead_names = process_ecg_signals(record)
    except Exception as e:
        return (None, f"❌ Error processing CSV: {e}", {'display':'none'}, [], [], "", "", "")
    
    # Store data
    ecg_data = {
        'signals': signals.tolist(),
        'fs': fs,
        'lead_names': lead_names,
    }
    
    # Create channel options (all available leads)
    channel_options = [{'label': f'Lead {name}', 'value': i} for i, name in enumerate(lead_names)]
    default_channels = [1] if len(lead_names) > 1 else [0]  # Default to Lead II if available
    
    # Make prediction using Lead II (index 1) if available, otherwise first lead
    prediction_lead_idx = 1 if len(lead_names) > 1 and 'II' in lead_names else 0
    
    prediction_signal = signals[:, prediction_lead_idx]
    is_healthy, confidence, error_msg = predict_health_status(prediction_signal, fs)
    
    if error_msg or is_healthy is None:
        prediction_result = html.Div("⚠️ Prediction Unavailable", style={'color': '#FFA500'})
        prediction_conf = "Model not available"
        prediction_det = "Please ensure model files are in models/ directory"
    else:
        if is_healthy:
            prediction_result = html.Div("✅ HEALTHY", style={'color': '#2ECC71'})
            prediction_conf = f"Confidence: {confidence*100:.1f}%"
            prediction_det = html.Div([
                html.P("📋 Analysis Summary:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'color': '#00d9ff'}),
                html.Ul([
                    html.Li(f"Based on Lead {lead_names[prediction_lead_idx]} analysis"),
                    html.Li("ECG pattern shows normal characteristics"),
                    html.Li(f"Model confidence: {confidence*100:.1f}%"),
                    html.Li("No significant abnormalities detected")
                ], style={'textAlign': 'left', 'color': '#ccc'})
            ])
        else:
            prediction_result = html.Div("⚠️ ABNORMAL", style={'color': '#E74C3C'})
            prediction_conf = f"Confidence: {confidence*100:.1f}%"
            prediction_det = html.Div([
                html.P("📋 Analysis Summary:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'color': '#00d9ff'}),
                html.Ul([
                    html.Li(f"Based on Lead {lead_names[prediction_lead_idx]} analysis"),
                    html.Li("ECG pattern shows potential abnormalities"),
                    html.Li(f"Model confidence: {confidence*100:.1f}%"),
                    html.Li("⚕️ Recommendation: Consult a healthcare professional", style={'color': '#E74C3C'})
                ], style={'textAlign': 'left', 'color': '#ccc'})
            ])
    
    return (ecg_data, 
            "✅ ECG processed successfully! Configure controls and click 'Start Streaming'",
            {'display': 'block'},
            channel_options,
            default_channels,
            prediction_result,
            prediction_conf,
            prediction_det)

@callback(
    [Output('stream-btn', 'children'),
     Output('interval-component', 'disabled'),
     Output('stream-state', 'data')],
    Input('stream-btn', 'n_clicks'),
    State('stream-state', 'data')
)
def toggle_streaming(n_clicks, stream_state):
    if n_clicks == 0:
        return "Start Streaming", True, {'streaming': False, 'position': 0}
    
    if stream_state is None:
        stream_state = {'streaming': False, 'position': 0}
    
    is_streaming = stream_state.get('streaming', False)
    
    if is_streaming:
        return "Start Streaming", True, {'streaming': False, 'position': 0}
    else:
        return "Stop Streaming", False, {'streaming': True, 'position': 0}

@callback(
    [Output('ecg-stream-graph', 'figure'),
     Output('stream-state', 'data', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('graph-mode', 'value'),
     Input('window-width', 'value'),
     Input('channel-selection', 'value')],
    [State('ecg-data-store', 'data'),
     State('stream-state', 'data')],
    prevent_initial_call=True
)
def update_stream(n_intervals, graph_mode, window_width, selected_channels, ecg_data, stream_state):
    if ecg_data is None or not selected_channels:
        return go.Figure(), stream_state
    
    signals = np.array(ecg_data['signals'])
    fs = ecg_data['fs']
    lead_names = ecg_data['lead_names']
    
    # Update streaming position
    if stream_state is None:
        stream_state = {'streaming': True, 'position': 0}
    
    position = stream_state.get('position', 0)
    samples_per_update = int(fs * 0.1)  # 100ms of data
    window_samples = int(window_width * fs)
    
    # Calculate window
    start = position
    end = min(position + window_samples, signals.shape[0])
    
    # Loop if reached end
    if end >= signals.shape[0]:
        position = 0
        start = 0
        end = min(window_samples, signals.shape[0])
    
    # Advance position
    new_position = position + samples_per_update
    if new_position >= signals.shape[0]:
        new_position = 0
    
    time_window = np.arange(start, end) / fs
    
    if graph_mode == 'waveform':
        # Create waveform plot
        n_channels = len(selected_channels)
        fig = make_subplots(
            rows=n_channels, cols=1,
            subplot_titles=[f'Lead {lead_names[ch]}' for ch in selected_channels if ch < len(lead_names)],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        colors = ['#00d9ff', '#ff006e', '#8338ec', '#fb5607', '#3a86ff', '#06ffa5']
        for idx, ch in enumerate(selected_channels):
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                fig.add_trace(
                    go.Scatter(
                        x=time_window,
                        y=signal_window,
                        mode='lines',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        showlegend=False
                    ),
                    row=idx+1, col=1
                )
        
        fig.update_layout(
            title="ECG Waveform (Streaming)",
            plot_bgcolor='#0f1626',
            paper_bgcolor='#16213e',
            font=dict(color='#00d9ff'),
            height=max(400, 150 * n_channels),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        for i in range(n_channels):
            fig.update_xaxes(gridcolor='#2a2a4e', row=i+1, col=1)
            fig.update_yaxes(gridcolor='#2a2a4e', title_text="mV", row=i+1, col=1)
        
        fig.update_xaxes(title_text="Time (seconds)", row=n_channels, col=1)
    
    elif graph_mode == 'polar':
        # Create polar plot for selected channels
        fig = go.Figure()
        
        colors = ['#00d9ff', '#ff006e', '#8338ec', '#fb5607', '#3a86ff', '#06ffa5']
        for idx, ch in enumerate(selected_channels):
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                theta = np.linspace(0, 360, len(signal_window))
                
                # Offset to ensure positive radius
                r_min = np.min(signal_window)
                r_offset = abs(r_min) + 0.1 if r_min < 0 else 0
                r_values = signal_window + r_offset
                
                fig.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta,
                    mode='lines',
                    name=f'Lead {lead_names[ch]}' if ch < len(lead_names) else f'Lead {ch+1}',
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="ECG Polar Representation (Streaming)",
            polar=dict(
                bgcolor='#0f1626',
                radialaxis=dict(
                    visible=True,
                    gridcolor='#2a2a4e',
                    color='#00d9ff'
                ),
                angularaxis=dict(
                    visible=True,
                    gridcolor='#2a2a4e',
                    color='#00d9ff'
                )
            ),
            paper_bgcolor='#16213e',
            font=dict(color='#00d9ff'),
            height=600,
            showlegend=True,
            legend=dict(
                bgcolor='#0f1626',
                bordercolor='#00d9ff',
                borderwidth=1
            )
        )
    
    elif graph_mode == 'recurrence':
        # Create recurrence plot for selected channels
        n_channels = len(selected_channels)
        
        if n_channels == 1:
            fig = go.Figure()
            ch = selected_channels[0]
            if ch < signals.shape[1]:
                signal_window = signals[start:end, ch]
                rp_data = generate_rp_data(signal_window)
                
                fig.add_trace(go.Heatmap(
                    z=rp_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Recurrence",
                        titleside="right",
                        tickfont=dict(color='#00d9ff'),
                        titlefont=dict(color='#00d9ff')
                    )
                ))
            
            fig.update_layout(
                title=f"Recurrence Plot - Lead {lead_names[ch] if ch < len(lead_names) else ch+1}",
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=600,
                margin=dict(l=40, r=40, t=80, b=40)
            )
        else:
            # Multiple recurrence plots in grid
            cols = min(2, n_channels)
            rows = (n_channels + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f'Lead {lead_names[ch]}' if ch < len(lead_names) else f'Lead {ch+1}' 
                               for ch in selected_channels],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for idx, ch in enumerate(selected_channels):
                row = idx // cols + 1
                col = idx % cols + 1
                
                if ch < signals.shape[1]:
                    signal_window = signals[start:end, ch]
                    rp_data = generate_rp_data(signal_window)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=rp_data,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Recurrence Plots (Streaming)",
                plot_bgcolor='#0f1626',
                paper_bgcolor='#16213e',
                font=dict(color='#00d9ff'),
                height=max(500, 300 * rows),
                margin=dict(l=40, r=40, t=80, b=40)
            )
        
        # Remove axis labels for recurrence plots
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
    
    else:
        fig = go.Figure()
    
    return fig, {'streaming': True, 'position': new_position}