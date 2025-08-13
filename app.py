from flask import Flask, render_template, request, redirect, url_for, session, flash
import psycopg2
from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import logging
from scipy.signal import savgol_filter
import os
import logging
from flask import Flask, jsonify, request, render_template
import psutil
import GPUtil
from datetime import datetime, timedelta
import time
import json
from collections import deque
from flask import Flask, render_template, request,jsonify
from xhtml2pdf import pisa
from io import BytesIO
import json
import google.generativeai as gemini
from flask import Flask, render_template, redirect, url_for, session, request, jsonify
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
import json
import google.generativeai as genai
import re
from flask import Flask, render_template, request, jsonify
import json
import re
import random
import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai

################################################################################################################################################################################################################################################################

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Example DB connection
conn = psycopg2.connect("dbname=carbon_footprint user=postgres password=Chinu@248")
cursor = conn.cursor()
#############################################        basic login, signup etc        ####################################################################################################################################################################################################################

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['POST'])
def signup():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash('Missing email or password.')
            return redirect(url_for('home'))

        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Email already registered. Try logging in.')
            return redirect(url_for('home'))

        cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)', (name, email, password))

        conn.commit()
        flash('Signup successful!')
        return redirect(url_for('home'))

    except Exception as e:
        conn.rollback()  # Rollback the transaction to avoid further errors
        flash(f'Error: {e}')
        return redirect(url_for('home'))
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        flash('Missing email or password.')
        return redirect(url_for('home'))

    cursor.execute('SELECT * FROM users WHERE email = %s AND password = %s', (email, password))
    user = cursor.fetchone()

    if user:
        session['user'] = user[0]  # Assuming user[0] is the user id
        session['user_name'] = user[3]  # Assuming user[3] is the name
        flash('Login successful!')
    else:
        flash('Invalid credentials!')
    return redirect(url_for('home'))
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!')
    return redirect(url_for('home'))

####################################################    Forecast      ######################################################################################################################################################################################################
   
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Enhanced power draw modeling with more controlled values
POWER_DRAW = {
    "CPU_IDLE": 35,        # Watts at idle
    "CPU_LOAD": 85,        # Watts at full load
    "RAM_PER_GB": 0.375,   # More precise scaling per GB
    "GPU_IDLE": 30,        # Watts at idle
    "GPU_LOAD": 250,       # Watts at full load
    "SSD_PER_TB": 4,       # Watts per TB for SSD
    "HDD_PER_TB": 7,       # Watts per TB for HDD
    "NETWORKING": 15       # Base networking equipment power
}

# Environmental and cost factors
EMISSION_FACTOR = 0.7      # kg CO2 per kWh (can vary by region)
COST_PER_KWH = 8.5         # ₹ per kWh

# Server room climate baseline
BASELINE_TEMP = 22         # Celsius - optimal temperature
BASELINE_HUMIDITY = 45     # Percent - optimal humidity

def smooth_data(data, window_size=7, polyorder=2):
    """Apply Savitzky-Golay filter to smooth the data with more conservative settings."""
    if len(data) > window_size:
        return savgol_filter(data, window_size, polyorder)
    return data

def detect_anomalies(data, threshold=2.5):  # Increased threshold
    """Detect anomalies using z-score method with higher threshold."""
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

def correct_anomalies(data, threshold=2.5):
    """Correct anomalies by replacing them with rolling median using larger window."""
    anomalies = detect_anomalies(data, threshold)
    if np.any(anomalies):
        # Use rolling median with larger window
        median_values = pd.Series(data).rolling(window=7, center=True, min_periods=1).median().values
        corrected_data = np.where(anomalies, median_values, data)
        return corrected_data
    return data

def calculate_server_power(cpu_util, ram_gb, num_gpus, gpu_util, ssd_tb, hdd_tb):
    """Calculate server power consumption based on component utilization."""
    power = (
        POWER_DRAW["CPU_IDLE"] + (cpu_util * (POWER_DRAW["CPU_LOAD"] - POWER_DRAW["CPU_IDLE"])) +
        ram_gb * POWER_DRAW["RAM_PER_GB"] +
        num_gpus * (POWER_DRAW["GPU_IDLE"] + gpu_util * (POWER_DRAW["GPU_LOAD"] - POWER_DRAW["GPU_IDLE"])) +
        ssd_tb * POWER_DRAW["SSD_PER_TB"] +
        hdd_tb * POWER_DRAW["HDD_PER_TB"] +
        POWER_DRAW["NETWORKING"]
    )
    return power

def calculate_cooling_overhead(room_temp, humidity, power_load):
    """More gradual cooling power overhead calculation."""
    temp_diff = max(0, room_temp - BASELINE_TEMP)
    humidity_factor = 1 + 0.003 * abs(humidity - BASELINE_HUMIDITY)  # Reduced impact
    cop = 3.5 - 0.05 * temp_diff  # More gradual COP reduction
    cop = max(2.0, cop)  # Higher minimum COP
    pue = 1 + (humidity_factor / cop)
    return power_load * (pue - 1) * 0.8  # Additional damping factor

def create_hourly_pattern(base_load, date):
    """Create realistic hourly load pattern with smoother transitions."""
    hourly_pattern = [
        0.85, 0.80, 0.78, 0.75, 0.78, 0.85,  # Smoother overnight transition
        0.90, 1.05, 1.15, 1.12, 1.08, 1.15,   # Reduced peak
        1.12, 1.08, 1.15, 1.12, 1.08, 1.05,   # Smoother daytime
        1.00, 0.95, 0.90, 0.87, 0.85, 0.85    # Smoother evening
    ]
    
    return [base_load * factor for factor in hourly_pattern]

def apply_seasonal_effects(date, base_consumption):
    """Apply more conservative seasonal effects."""
    month = date.month
    day = date.day
    
    # Create continuous factor that changes smoothly between months
    month_factor = month + day/30  # Approximate position in year
    
    # Much more subtle seasonal pattern - reduced from 0.1 to 0.03
    seasonal_factor = 1 + 0.03 * np.sin((month_factor - 1) * 2 * np.pi / 12)
    
    return base_consumption * seasonal_factor

def generate_anomalies(base_data):
    """Generate more controlled anomalies with smoother transitions."""
    anomaly_prob = 0.015  # Reduced probability
    result = base_data.copy()
    
    for i in range(1, len(result)-2):  # Leave buffer at ends
        if np.random.random() < anomaly_prob:
            # More controlled anomaly magnitude (5-15% change)
            anomaly_type = np.random.choice(['spike', 'dip'], p=[0.6, 0.4])
            if anomaly_type == 'spike':
                factor = np.random.uniform(1.05, 1.15)  # Reduced from 1.2-1.5
            else:
                factor = np.random.uniform(0.85, 0.95)  # Reduced from 0.6-0.8
            
            # Apply with smoother transition
            result[i] *= factor
            result[i+1] *= (factor * 0.95)  # Diminishing effect
            result[i+2] *= (factor * 0.9)   # Further diminishing
    
    return result

def apply_hardware_aging(data, days_span):
    """More gradual hardware aging effect."""
    aging_factors = np.linspace(1.0, 1.005, len(data))  # Reduced from 1.02
    return data * aging_factors

def preprocess_data(df):
    """Enhanced preprocessing with additional smoothing."""
    y_values = df['y'].values
    
    # First correct anomalies with higher threshold
    corrected_y = correct_anomalies(y_values, threshold=2.5)
    
    # Then apply smoothing with larger window
    smoothed_y = smooth_data(corrected_y, window_size=7)
    
    # Ensure no negative values and cap extreme highs
    smoothed_y = np.maximum(smoothed_y, 1.0)
    smoothed_y = np.minimum(smoothed_y, np.percentile(smoothed_y, 99))
    
    df['y'] = smoothed_y
    return df

def predict_with_prophet(df, periods=30):
    """Generate forecast using Facebook Prophet with more conservative settings."""
    df = preprocess_data(df.copy())
    
    # Change from logistic to linear growth - more stable for this application
    model = Prophet(
        growth='linear',  # Changed from logistic to linear
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,  # Slightly increased for more flexibility
        seasonality_prior_scale=8,     # Reduced for more stable seasonality
        interval_width=0.95,
        changepoint_range=0.8          # Adjusted to avoid edge effects
    )
    
    # Remove carrying capacity since we're using linear growth
    # df['cap'] = df['y'].max() * 1.2
    
    # Use more conservative seasonality
    model.add_seasonality(name='daily', period=1, fourier_order=3)
    
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    # Remove cap for linear model
    # future['cap'] = df['y'].max() * 1.2
    
    forecast = model.predict(future)
    
    # Ensure realistic values with smoother transition
    last_actual = df['y'].iloc[-1]
    
    # Apply a more conservative transition factor - prevent dramatic shifts
    for i in range(min(10, periods)):
        blend_factor = min(0.3, (i + 1) / 15)  # Much more gradual increase
        forecast.loc[len(df) + i, 'yhat'] = (1 - blend_factor) * last_actual + blend_factor * forecast.loc[len(df) + i, 'yhat']
    
    # Apply more conservative upper/lower bounds - prevent extreme changes
    max_change_rate = 0.015  # Maximum 1.5% change per day
    
    for i in range(1, periods):
        idx = len(df) + i
        prev_idx = len(df) + i - 1
        max_change = forecast.loc[prev_idx, 'yhat'] * max_change_rate
        
        # Ensure forecast doesn't change too rapidly
        if forecast.loc[idx, 'yhat'] > forecast.loc[prev_idx, 'yhat'] + max_change:
            forecast.loc[idx, 'yhat'] = forecast.loc[prev_idx, 'yhat'] + max_change
        elif forecast.loc[idx, 'yhat'] < forecast.loc[prev_idx, 'yhat'] - max_change:
            forecast.loc[idx, 'yhat'] = forecast.loc[prev_idx, 'yhat'] - max_change
    
    # Apply final bounds
    forecast['yhat'] = forecast['yhat'].clip(lower=df['y'].min() * 0.95, upper=df['y'].max() * 1.05)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=df['y'].min() * 0.9)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=df['y'].min() * 0.9, upper=df['y'].max() * 1.1)
    
    return forecast


def predict_with_sarima(data, periods=30):
    """Generate forecast using SARIMA model with more stable configuration."""
    try:
        # Enhanced preprocessing
        data = smooth_data(data, window_size=7)
        data = correct_anomalies(data, threshold=2.5)
        
        model = SARIMAX(
            data,
            order=(1, 1, 1),          # Simpler model
            seasonal_order=(1, 1, 0, 7),  # Reduced complexity
            enforce_stationarity=True,
            enforce_invertibility=True,
            initialization='approximate_diffuse'  # More stable initialization
        )
        results = model.fit(disp=False)
        
        forecast = results.get_forecast(steps=periods)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Apply realistic bounds
        mean_forecast = np.maximum(mean_forecast, 1.0)
        mean_forecast = np.minimum(mean_forecast, np.max(data) * 1.3)
        conf_int.iloc[:, 0] = np.maximum(conf_int.iloc[:, 0], 1.0)
        conf_int.iloc[:, 1] = np.minimum(conf_int.iloc[:, 1], np.max(data) * 1.5)
        
        return mean_forecast, conf_int
    except Exception as e:
        logger.error(f"SARIMA prediction failed: {e}")
        return None, None

@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    if request.method == "POST":
        try:
            # Get form data and process it
            num_servers = int(request.form["num_servers"])
            ram_gb = int(request.form["ram_gb"])
            num_gpus = int(request.form["num_gpus"])
            avg_cpu_load = float(request.form["avg_load"]) / 100
            avg_gpu_load = float(request.form.get("gpu_load", 50)) / 100
            ssd_tb = float(request.form.get("ssd_storage", 1))
            hdd_tb = float(request.form.get("hdd_storage", 0))
            
            room_temp = float(request.form["room_temp"])
            humidity = float(request.form.get("humidity", 45))
            
            # Calculate base power with bounds checking
            power_per_server = min(1000, max(50, calculate_server_power(
                avg_cpu_load, ram_gb, num_gpus, avg_gpu_load, ssd_tb, hdd_tb
            )))

            today = datetime.today()
            dates = [today - timedelta(days=i) for i in reversed(range(60))]
            usage_data = []
            
            for date in dates:
                daily_base = power_per_server * num_servers
                hourly_power = create_hourly_pattern(daily_base, date)
                daily_kwh = sum(hourly_power) / 1000
                daily_kwh = apply_seasonal_effects(date, daily_kwh)
                usage_data.append({
                    "ds": date.strftime("%Y-%m-%d"),
                    "y": round(daily_kwh, 2)
                })
            
            df = pd.DataFrame(usage_data)
            
            # Apply effects with more control
            df['y'] = apply_hardware_aging(df['y'].values, 60)
            df['y'] = generate_anomalies(df['y'].values)
            
            # Calculate cooling with bounds
            cooling_overhead = [min(y * 0.3, calculate_cooling_overhead(room_temp, humidity, y)) for y in df['y']]
            df['y'] = df['y'] + cooling_overhead
            df.dropna(inplace=True)
            
            # Additional validation
            if df.shape[0] < 7:
                return "Error: Not enough valid data points to predict. Please adjust your inputs!"
            
            # Apply final smoothing and clipping
            df['y'] = df['y'].clip(lower=1.0, upper=df['y'].quantile(0.99))
            df['y'] = smooth_data(df['y'].values, window_size=5)
            
            # Generate forecasts
            forecast = predict_with_prophet(df, periods=30)
            prediction = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
            prediction["ds"] = prediction["ds"].dt.strftime("%Y-%m-%d")
            prediction["yhat"] = prediction["yhat"].apply(lambda x: round(max(1.0, min(x, df['y'].max() * 1.5)), 2))
            prediction["yhat_lower"] = prediction["yhat_lower"].apply(lambda x: round(max(1.0, x), 2))
            prediction["yhat_upper"] = prediction["yhat_upper"].apply(lambda x: round(max(1.0, x), 2))
            prediction["co2"] = prediction["yhat"] * EMISSION_FACTOR
            prediction["cost"] = prediction["yhat"] * COST_PER_KWH

            sarima_forecast, sarima_conf = predict_with_sarima(df['y'].values, periods=30)
            if sarima_forecast is not None:
                prediction["sarima"] = [round(max(1.0, min(x, df['y'].max() * 1.3)), 2) for x in sarima_forecast]
            
            # Calculate summary statistics
            total_kwh_7_days = round(df.tail(7)["y"].sum(), 2)
            total_co2_7_days = round(total_kwh_7_days * EMISSION_FACTOR, 2)
            total_cost_7_days = round(total_kwh_7_days * COST_PER_KWH, 2)
            
            forecast_kwh_30_days = round(prediction["yhat"].sum(), 2)
            forecast_co2_30_days = round(forecast_kwh_30_days * EMISSION_FACTOR, 2)
            forecast_cost_30_days = round(forecast_kwh_30_days * COST_PER_KWH, 2)
            
            # Prepare chart data with bounds checking
            chart_data = {
                "dates": df["ds"].tolist() + prediction["ds"].tolist(),
                "historical": df["y"].tolist() + [None] * 30,
                "forecast": [None] * len(df) + prediction["yhat"].tolist(),
                "lower_bound": [None] * len(df) + prediction["yhat_lower"].tolist(),
                "upper_bound": [None] * len(df) + prediction["yhat_upper"].tolist()
            }
            
            if "sarima" in prediction.columns:
                chart_data["sarima"] = [None] * len(df) + prediction["sarima"].tolist()
            
            # Log data for debugging
            logger.info(f"Historical data range: {min(df['y'])} - {max(df['y'])}")
            logger.info(f"Forecast range: {min(prediction['yhat'])} - {max(prediction['yhat'])}")
            
            return render_template(
                "forecast_result.html",
                data=prediction.to_dict(orient="records"),
                chart_data=json.dumps(chart_data),
                total_power=total_kwh_7_days,
                total_co2=total_co2_7_days,
                total_cost=total_cost_7_days,
                forecast_power=forecast_kwh_30_days,
                forecast_co2=forecast_co2_30_days,
                forecast_cost=forecast_cost_30_days,
                server_config={
                    "servers": num_servers,
                    "ram": ram_gb,
                    "gpus": num_gpus,
                    "cpu_load": avg_cpu_load * 100,
                    "gpu_load": avg_gpu_load * 100,
                    "storage": f"{ssd_tb} TB SSD, {hdd_tb} TB HDD",
                    "room_temp": room_temp,
                    "humidity": humidity
                }
            )

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return f"Error: {str(e)}"

    return render_template("forecast.html")


#########################################   Game footprint               ##############################################################################################################################################################################################################

session_data = {
    'start_time': None,
    'total_energy': 0,  # in Watt-hours
    'last_update': None,
    'current_game': None,
    'emission_history': deque(maxlen=30)  # Stores last 30 emissions readings
}

# Emission factors (kg CO₂ per kWh)
EMISSION_FACTORS = {
    'india': 0.7,      # India's grid average
    'global': 0.475,   # Global average
    'renewable': 0.05  # Renewable energy
}

# Game database with typical power profiles
GAME_DATABASE = {
    'valorant': {
        'name': 'Valorant',
        'typical_power': 200,
        'resolution': '1080p',
        'fps': 144,
        'graphics': 'medium',
        'emission_class': 'low'
    },
    'cyberpunk 2077': {
        'name': 'Cyberpunk 2077',
        'typical_power': 400,
        'resolution': '1440p',
        'fps': 60,
        'graphics': 'high',
        'emission_class': 'medium'
    },
    'microsoft flight simulator': {
        'name': 'Microsoft Flight Simulator',
        'typical_power': 500,
        'resolution': '4K',
        'fps': 45,
        'graphics': 'ultra',
        'emission_class': 'high'
    }
}

# Optimization tips
OPTIMIZATION_TIPS = [
    {
        'type': 'cpu',
        'title': 'Reduce CPU Usage',
        'message': 'Close background applications to reduce CPU load',
        'impact': 'Can reduce power consumption by 10-20%',
        'difficulty': 'easy',
        'savings_percent': 15
    },
    {
        'type': 'gpu',
        'title': 'Lower Graphics Settings',
        'message': 'Reduce shadow quality and anti-aliasing',
        'impact': 'Can reduce GPU power by 20-30%',
        'difficulty': 'easy',
        'savings_percent': 25
    },
    {
        'type': 'fps',
        'title': 'Cap FPS',
        'message': 'Limit FPS to your monitor refresh rate',
        'impact': 'Can reduce power by 15-25%',
        'difficulty': 'easy',
        'savings_percent': 20
    },
    {
        'type': 'resolution',
        'title': 'Lower Resolution',
        'message': 'Play at 1080p instead of 4K when possible',
        'impact': 'Can reduce power by 30-40%',
        'difficulty': 'easy',
        'savings_percent': 35
    },
    {
        'type': 'system',
        'title': 'Enable Power Saving',
        'message': 'Use your system power saving features',
        'impact': 'Can reduce total power by 5-10%',
        'difficulty': 'easy',
        'savings_percent': 7
    }
]

def get_hardware_info():
    """Get detailed hardware information with error handling"""
    try:
        # CPU Info
        cpu_freq = psutil.cpu_freq()
        cpu_info = {
            'model': f"{cpu_freq.current if cpu_freq else 'Unknown'} MHz",
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'usage': psutil.cpu_percent(interval=1),
            'temperature': get_cpu_temperature(),
            'power': get_cpu_power()
        }

        # GPU Info
        gpus = GPUtil.getGPUs()
        gpu_info = []
        if gpus:
            for gpu in gpus:
                gpu_info.append({
                    'model': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100,
                    'power': gpu.powerDraw if hasattr(gpu, 'powerDraw') else None
                })
        else:
            gpu_info.append({
                'model': 'No GPU detected', 
                'load': 0, 
                'power': 0
            })

        return {
            'cpu': cpu_info,
            'gpu': gpu_info,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Hardware info error: {str(e)}")
        return {'error': str(e)}

def get_cpu_temperature():
    """Get CPU temperature if available"""
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
        return None
    except Exception as e:
        logger.warning(f"CPU temp error: {str(e)}")
        return None

def get_cpu_power():
    """Get CPU power consumption in Watts"""
    try:
        # Try to get actual power if available
        if hasattr(psutil, "cpu_power"):
            return psutil.cpu_power()
        
        # Estimate based on usage and TDP (65W for typical gaming CPU)
        cpu_percent = psutil.cpu_percent(interval=1)
        return (cpu_percent / 100) * 65
    except Exception as e:
        logger.warning(f"CPU power error: {str(e)}")
        return None

def calculate_power_usage():
    """Calculate current system power usage in Watts"""
    try:
        # Get CPU power (fallback to estimation if needed)
        cpu_power = get_cpu_power() or (psutil.cpu_percent(interval=1) / 100) * 65
        
        # Get GPU power (sum all GPUs)
        gpu_power = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                if hasattr(gpu, 'powerDraw') and gpu.powerDraw is not None:
                    gpu_power += gpu.powerDraw
                else:
                    # Estimate GPU power (150W TDP * load)
                    gpu_power += gpu.load * 150
        
        total_power = cpu_power + gpu_power
        
        # Update session energy (Watt-hours)
        now = datetime.now()
        if session_data['last_update']:
            time_diff = (now - session_data['last_update']).total_seconds() / 3600
            session_data['total_energy'] += total_power * time_diff
        
        session_data['last_update'] = now
        
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'gpu_usage': gpus[0].load * 100 if gpus else 0,
            'cpu_power': cpu_power,
            'gpu_power': gpu_power,
            'total_power': total_power,
            'total_energy': session_data['total_energy'],
            'session_duration': (now - (session_data['start_time'] or now)).total_seconds(),
            'current_game': session_data['current_game'],
            'timestamp': now.isoformat()
        }
    except Exception as e:
        logger.error(f"Power calculation error: {str(e)}")
        return {'error': str(e)}

def update_emission_history():
    """Update emission history with current power usage"""
    try:
        power_data = calculate_power_usage()
        if 'total_power' in power_data and session_data['last_update']:
            # Calculate emissions for the last update interval
            time_diff = (datetime.now() - session_data['last_update']).total_seconds() / 3600
            if time_diff > 0:  # Only update if some time has passed
                power = power_data['total_power']
                region = 'india'  # Default region
                emissions = calculate_emissions(power, time_diff, region)
                
                if emissions is not None:
                    session_data['emission_history'].append({
                        'emissions_kg': emissions,
                        'power_watts': power,
                        'time_hours': time_diff,
                        'region': region,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"Updated emission history: {power}W for {time_diff}h = {emissions}kg CO₂")
    except Exception as e:
        logger.error(f"Update emission history error: {str(e)}")

def get_optimization_suggestions():
    """Get personalized optimization tips based on current usage"""
    try:
        hardware = get_hardware_info()
        suggestions = []
        
        # CPU-related tips
        if hardware['cpu']['usage'] > 80:
            suggestions.append({
                **OPTIMIZATION_TIPS[0],
                'potential_savings': hardware['cpu']['power'] * (OPTIMIZATION_TIPS[0]['savings_percent'] / 100)
            })
        
        # GPU-related tips
        if hardware['gpu'] and hardware['gpu'][0]['load'] > 70:
            suggestions.append({
                **OPTIMIZATION_TIPS[1],
                'potential_savings': hardware['gpu'][0].get('power', 150) * (OPTIMIZATION_TIPS[1]['savings_percent'] / 100)
            })
        
        # Always include general tips
        suggestions.extend(OPTIMIZATION_TIPS[2:])
        
        # Calculate potential savings
        power_data = calculate_power_usage()
        if 'total_power' in power_data:
            for tip in suggestions:
                if 'savings_percent' in tip and 'potential_savings' not in tip:
                    tip['potential_savings'] = power_data['total_power'] * (tip['savings_percent'] / 100)
        
        return suggestions
    except Exception as e:
        logger.error(f"Optimization suggestions error: {str(e)}")
        return []

def calculate_emissions(power_watts, hours, region='india'):
    """Calculate CO₂ emissions in kg"""
    try:
        emission_factor = EMISSION_FACTORS.get(region, 0.7)
        energy_kwh = (power_watts * hours) / 1000
        emissions = energy_kwh * emission_factor
        return emissions
    except Exception as e:
        logger.error(f"Emission calculation error: {str(e)}")
        return None

@app.route('/gamefootprint')
def gamefootprint():
    """Initialize session if needed"""
    if not session_data['start_time']:
        session_data['start_time'] = datetime.now()
        session_data['last_update'] = datetime.now()
    return render_template('gamefootprint.html')

@app.route('/api/hardware', methods=['GET'])
def hardware_info():
    """Endpoint for hardware information"""
    return jsonify(get_hardware_info())

@app.route('/api/power', methods=['GET'])
def power_usage():
    """Endpoint for real-time power data"""
    power_data = calculate_power_usage()
    update_emission_history()  # Update emission history with latest data
    return jsonify(power_data)

@app.route('/api/emissions', methods=['POST'])
def emissions_calculation():
    """Calculate emissions with optional parameters"""
    try:
        data = request.get_json()
        power = float(data.get('power_watts', 0))
        hours = float(data.get('hours', 0))
        region = data.get('region', 'india')
        
        emissions = calculate_emissions(power, hours, region)
        if emissions is None:
            raise ValueError("Invalid emission calculation")
        
        # Store this emission data point in history
        session_data['emission_history'].append({
            'emissions_kg': emissions,
            'power_watts': power,
            'hours': hours,
            'region': region,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Added manual emission calculation: {power}W for {hours}h = {emissions}kg CO₂")
        
        # Get optimization suggestions
        suggestions = get_optimization_suggestions()
        
        return jsonify({
            'emissions_kg': emissions,
            'power_watts': power,
            'hours': hours,
            'emission_factor': EMISSION_FACTORS.get(region, 0.7),
            'optimization_suggestions': suggestions,
            'current_game': session_data['current_game'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Emission API error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/game', methods=['POST'])
def set_current_game():
    """Set the currently playing game"""
    try:
        data = request.get_json()
        game_name = data.get('name', '').lower()
        
        if game_name in GAME_DATABASE:
            session_data['current_game'] = GAME_DATABASE[game_name]
        else:
            session_data['current_game'] = {
                'name': data.get('name', 'Custom Game'),
                'typical_power': data.get('power', 300),
                'resolution': data.get('resolution', '1080p'),
                'fps': data.get('fps', 60),
                'graphics': data.get('graphics', 'medium'),
                'emission_class': 'custom'
            }
        
        return jsonify({
            'status': 'success',
            'game': session_data['current_game']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/optimization', methods=['GET'])
def optimization_tips():
    """Get optimization suggestions"""
    return jsonify({
        'suggestions': get_optimization_suggestions(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/session', methods=['GET'])
def session_info():
    """Get current session information"""
    # Make sure emission history is properly formatted for JSON
    emission_history = list(session_data['emission_history'])
    
    return jsonify({
        'start_time': session_data['start_time'].isoformat() if session_data['start_time'] else None,
        'duration': (datetime.now() - (session_data['start_time'] or datetime.now())).total_seconds(),
        'total_energy': session_data['total_energy'],
        'current_game': session_data['current_game'],
        'emission_history': emission_history,
        'emissions_count': len(emission_history)
    })

@app.route('/api/emission_history', methods=['GET'])
def get_emission_history():
    """Get emission history data for charts"""
    history = list(session_data['emission_history'])
    return jsonify({
        'history': history,
        'count': len(history),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Reset the current session"""
    try:
        session_data['start_time'] = datetime.now()
        session_data['total_energy'] = 0
        session_data['last_update'] = datetime.now()
        session_data['emission_history'].clear()
        
        return jsonify({
            'status': 'success',
            'message': 'Session data cleared',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
####################################   product life cycle tracker         ###############################################################################################################################################################################################################

gemini.configure(api_key="")
@app.route('/product-life-cycle', methods=['GET'])
def product_life_cycle():
    return render_template('product_life.html')

# Load device CO2 data
with open("static/devices.json") as f:
    device_data = json.load(f)

@app.route('/submit-device', methods=['POST'])
def submit_device():
    name = request.form.get("name", "User")
    device_type = request.form['device_type']
    brand = request.form['brand']
    model = request.form['model']
    years = int(request.form['years'])

    try:
        profile = device_data[device_type][brand][model]
    except KeyError:
        return jsonify({"error": "Device data not found."}), 400

    co2_manufacture = profile["manufacture"]
    co2_usage = profile["usage_per_year"] * years
    co2_disposal = profile["disposal"]
    total_co2 = co2_manufacture + co2_usage + co2_disposal

    # Benchmark assumes a new device each year
    annual_benchmark = profile["manufacture"] + profile["usage_per_year"] + profile["disposal"]
    benchmark_total = annual_benchmark * years
    co2_saved = max(0, round(benchmark_total - total_co2, 2))

    return jsonify({
        "device": f"{brand} {model}",
        "co2_manufacture": co2_manufacture,
        "co2_usage": round(co2_usage, 2),
        "co2_disposal": co2_disposal,
        "total": round(total_co2, 2),
        "co2_saved": co2_saved
    })

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    device = request.form.get("device", "")
    years = request.form.get("years", "")
    user_query = request.form.get("query", "")

    context_query = (
    f"The user has been using a {device} for approximately {years} years. "
    f"Taking into account environmental sustainability and the ecological impact of prolonged device usage, "
    f"respond to the following query with practical and concise insights:\n"
    f"{user_query}"
)


    chat = gemini.GenerativeModel("gemini-2.0-flash").start_chat()
    response = chat.send_message(context_query)

    return jsonify({"reply": response.text})
############################################    file transfer       ###############################################################################################################################################################################################################


genai.configure(api_key="")

# Network and data transfer constants (in grams CO2 per GB)
TRANSFER_METHODS = {
    "Internet Transfer": {
        "carbon_per_gb": 50,  # grams CO2 per GB
        "description": "Standard internet data transfer"
    },
    "Cloud Provider Internal": {
        "carbon_per_gb": 20,
        "description": "Transfer within same cloud provider"
    },
    "Same Region Transfer": {
        "carbon_per_gb": 10,
        "description": "Transfer within same data center region"
    },
    "Same Zone Transfer": {
        "carbon_per_gb": 5,
        "description": "Transfer within same availability zone"
    }
}

@app.route('/file-transfer' , methods=['GET'])
def file_transfer():
    return render_template('file.html', transfer_methods=TRANSFER_METHODS.keys())

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    # Get transfer method carbon intensity
    method_data = TRANSFER_METHODS[data['transfer_method']]
    carbon_per_gb = method_data["carbon_per_gb"]
    
    # Calculate carbon footprint (convert grams to kg)
    data_gb = float(data['data_size'])
    carbon_footprint = (carbon_per_gb * data_gb) / 1000  # kg CO2
    
    # Calculate equivalent emissions
    equivalents = get_equivalents(carbon_footprint)
    
    return jsonify({
        'carbon_footprint': round(carbon_footprint, 4),  # kg CO2
        'equivalents': equivalents,
        'method_info': method_data["description"]
    })

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    
    # Get transfer method carbon intensity
    method_data = TRANSFER_METHODS[data['transfer_method']]
    carbon_per_gb = method_data["carbon_per_gb"]
    
    # Calculate carbon footprint (convert grams to kg)
    data_gb = float(data['data_size'])
    carbon_footprint = (carbon_per_gb * data_gb) / 1000  # kg CO2

    # Create a prompt for the generative model
    prompt = f"""
    Suggest 3–5 actionable tips to reduce the carbon footprint for file transfers.
    The user transferred:
    - Data Size: {data['data_size']} GB
    - Transfer Method: {data['transfer_method']}
    - Carbon Footprint: {carbon_footprint} kg CO2
    """

    
    model = genai.GenerativeModel("gemini-2.0-flash")  # Replace with your model initialization
    response = model.generate_content(prompt)  # Generate content based on the prompt
    tips = response.text.strip().split("\n")  # Parse the response into a list of tips

    # Mock response for demonstration purposes
    

    return jsonify({'tips': tips})

def get_equivalents(kg_co2):
    """Convert kg CO2 to relatable equivalents"""
    return {
        'car_miles': round(kg_co2 * 2.5, 2),  # Average car emissions
        'smartphone_charges': round(kg_co2 * 50, 0),  # Based on 0.02kg per full charge
        'tree_days': round(kg_co2 * 10, 1)  # Tree absorbs ~0.1kg CO2 per day
    }



    suggestions = []
    
    # Suggestion for storage optimization
    if float(data['storage_size']) > 10:
        suggestions.append(f"Consider archiving or deleting unused data to reduce from {data['storage_size']}TB")
    
    # Suggestion for provider selection
    if data['provider'] == "Microsoft Azure Blob Storage" and current_footprint > 100:
        suggestions.append("Google Cloud Storage has lower energy consumption per TB in most regions")
    
    # Suggestion for region selection
    current_region_carbon = CLOUD_PROVIDERS[data['provider']]["regions"][data['region']]
    lowest_carbon_region = min(
        CLOUD_PROVIDERS[data['provider']]["regions"].items(),
        key=lambda x: x[1]
    )
    if current_region_carbon > lowest_carbon_region[1] * 1.2:
        suggestions.append(f"Consider switching to {lowest_carbon_region[0]} region with {lowest_carbon_region[1]} kg/kWh intensity")
    
    if not suggestions:
        suggestions.append("Your cloud storage configuration is relatively efficient!")
    
    return suggestions
################################################################################################################################################################################################################################################################

STREAMING_QUALITIES = {
    "Low (480p)": {
        "data_rate": 0.7,  # GB per hour
        "description": "Standard definition quality"
    },
    "Medium (720p)": {
        "data_rate": 1.5,
        "description": "HD quality"
    },
    "High (1080p)": {
        "data_rate": 3.0,
        "description": "Full HD quality"
    },
    "Ultra (4K)": {
        "data_rate": 7.0,
        "description": "Ultra HD quality"
    }
}

# Device types and their power consumption (in watts)
DEVICE_TYPES = {
    "Smartphone": {
        "power": 2,
        "description": "Mobile device streaming"
    },
    "Tablet": {
        "power": 4,
        "description": "Medium-sized tablet"
    },
    "Laptop": {
        "power": 15,
        "description": "Standard laptop"
    },
    "Desktop PC": {
        "power": 30,
        "description": "Computer with external monitor"
    },
    "Smart TV": {
        "power": 50,
        "description": "Large screen television"
    }
}

# Network transmission factors (grams CO2 per GB)
NETWORK_TYPES = {
    "Mobile Data (4G/5G)": 50,
    "WiFi (Home)": 30,
    "WiFi (Public)": 40,
    "Wired (Ethernet)": 20
}

# Add this function before your route definitions
def get_equivalents(carbon_kg):
    """
    Convert carbon emissions to everyday equivalents
    """
    return {
        'car_miles': round(carbon_kg * 2.5, 1),  # 1 kg CO2 = ~2.5 miles driven
        'plastic_bottles': round(carbon_kg * 63, 0),  # 1 kg CO2 = ~63 plastic bottles
        'tree_days': round(carbon_kg * 1.2, 1)  # 1 kg CO2 = ~1.2 days of tree absorption
    }

@app.route('/streaming', methods=['GET'])
def streaming():
    return render_template('streaming.html', 
                         qualities=STREAMING_QUALITIES.keys(),
                         devices=DEVICE_TYPES.keys(),
                         networks=NETWORK_TYPES.keys())

@app.route('/calculate_streaming', methods=['POST'])
def calculate_streaming():
    data = request.json
    
    # Get streaming quality data rate
    quality_data = STREAMING_QUALITIES[data['quality']]
    data_rate = quality_data["data_rate"]
    
    # Get device power consumption
    device_data = DEVICE_TYPES[data['device']]
    device_power = device_data["power"]
    
    # Get network carbon intensity
    network_carbon = NETWORK_TYPES[data['network']]  # grams CO2 per GB
    
    # Calculate total data transferred
    hours = float(data['hours'])
    total_data = data_rate * hours  # GB
    
    # Calculate carbon from data transfer (convert grams to kg)
    transfer_carbon = (total_data * network_carbon) / 1000  # kg CO2
    
    # Calculate carbon from device usage (kWh then to kg CO2)
    device_energy = (device_power * hours) / 1000  # kWh
    device_carbon = device_energy * 0.475  # Using global average of 0.475 kg/kWh
    
    # Total carbon footprint
    total_carbon = transfer_carbon + device_carbon
    
    # Calculate equivalent emissions
    equivalents = get_equivalents(total_carbon)
    
    return jsonify({
        'total_carbon': round(total_carbon, 4),
        'transfer_carbon': round(transfer_carbon, 4),
        'device_carbon': round(device_carbon, 4),
        'total_data': round(total_data, 2),
        'equivalents': equivalents,
        'quality_info': quality_data["description"],
        'device_info': device_data["description"]
    })

@app.route('/optimize_streaming', methods=['POST'])
def optimize_streaming():
    data = request.json

    # Reuse calculations from calculate endpoint
    quality_data = STREAMING_QUALITIES[data['quality']]
    data_rate = quality_data["data_rate"]
    device_data = DEVICE_TYPES[data['device']]
    device_power = device_data["power"]
    hours = float(data['hours'])
    network_carbon = NETWORK_TYPES[data['network']]
    
    total_data = data_rate * hours
    transfer_carbon = (total_data * network_carbon) / 1000
    device_energy = (device_power * hours) / 1000
    device_carbon = device_energy * 0.475
    total_carbon = transfer_carbon + device_carbon

    # Create prompt for Gemini
    prompt = f"""
    Based on these streaming metrics, suggest 3-4 specific ways to reduce carbon footprint:
    Current Settings:
    - Quality: {data['quality']} ({data_rate} GB/hour)
    - Device: {data['device']} ({device_power} watts)
    - Network: {data['network']}
    - Hours: {hours}
    Current Impact:
    - Data Transfer: {round(transfer_carbon, 3)} kg CO2
    - Device Usage: {round(device_carbon, 3)} kg CO2
    - Total Carbon: {round(total_carbon, 3)} kg CO2

    Provide actionable suggestions to reduce environmental impact.
    """

    # Generate suggestions using Gemini
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    # Clean asterisks from suggestions
    raw_suggestions = response.text.split('\n')
    suggestions = [re.sub(r"\*", "", line).strip() for line in raw_suggestions if line.strip()]

    return jsonify({
        'impact_metrics': {
            'data_transfer_carbon': round(transfer_carbon, 4),
            'device_carbon': round(device_carbon, 4),
            'total_carbon': round(total_carbon, 4),
            'total_data': round(total_data, 2)
        },
        'current_settings': {
            'quality': data['quality'],
            'device': data['device'],
            'network': data['network'],
            'hours': hours
        },
        'optimization_suggestions': suggestions
    })

################################################################################################################################################################################################################################################################



# Cloud provider data (storage power in W/TB, PUE, and carbon intensity)
CLOUD_PROVIDERS = {
    "AWS S3": {
        "storage_power": 0.65,  # W per TB
        "pue": 1.2,            # Power Usage Effectiveness
        "regions": {
            "US East (Virginia)": 0.415,
            "EU (Ireland)": 0.378,
            "Asia Pacific (Mumbai)": 0.708
        }
    },
    "Google Cloud Storage": {
        "storage_power": 0.55,
        "pue": 1.1,
        "regions": {
            "US (Iowa)": 0.415,
            "EU (Belgium)": 0.378,
            "Asia (Mumbai)": 0.708
        }
    },
    "Microsoft Azure Blob Storage": {
        "storage_power": 0.7,
        "pue": 1.125,
        "regions": {
            "US East (Virginia)": 0.415,
            "West Europe (Netherlands)": 0.378,
            "Asia Pacific (Mumbai)": 0.708
        }
    }
}

@app.route('/cloud' , methods=['GET'])
def cloud():
    return render_template('cloud.html', providers=CLOUD_PROVIDERS.keys())

@app.route('/get_regions/<provider>')
def get_regions(provider):
    return jsonify(list(CLOUD_PROVIDERS[provider]["regions"].keys()))

@app.route('/calculate_cloud', methods=['POST'])
def calculate_cloud():
    data = request.json

    provider_data = CLOUD_PROVIDERS[data['provider']]
    carbon_intensity = provider_data["regions"][data['region']]

    # Calculate energy consumption (kWh per year)
    storage_tb = float(data['storage_size'])
    power_w = storage_tb * provider_data["storage_power"] * provider_data["pue"]
    energy_kwh = (power_w * 24 * 365) / 1000  # Convert to kWh/year

    # Calculate carbon footprint (kg CO2 per year)
    carbon_footprint = energy_kwh * carbon_intensity

    # Calculate monthly equivalent
    monthly_footprint = carbon_footprint / 12

    # Optimization suggestions
    suggestions = generate_suggestions(data, carbon_footprint)

    return jsonify({
        'carbon_footprint': round(carbon_footprint, 2),
        'monthly_footprint': round(monthly_footprint, 2),
        'energy_consumption': round(energy_kwh, 2),
        'suggestions': suggestions
    })
@app.route('/optimize_cloud', methods=['POST'])
def optimize_cloud():
    import re
    data = request.json
    data_used = float(data['data_used'])
    energy_consumed = float(data['energy_consumed'])
    carbon_emission = float(data['carbon_emission'])

    prompt = f"""
    Suggest 3–5 actionable tips to reduce the carbon footprint for streaming content.
    The user streamed content using:
    - Data: {data_used} GB
    - Energy: {energy_consumed} kWh
    - CO₂: {carbon_emission} kg
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    # Remove all ** markers completely instead of escaping them
    clean_text = response.text.strip().replace('**', '')
    tips = clean_text.split("\n")

    return jsonify({'tips': tips})
def generate_suggestions(data, current_footprint):
    suggestions = []
    
    # Suggestion for storage optimization
    if float(data['storage_size']) > 10:
        suggestions.append(f"Consider archiving or deleting unused data to reduce from {data['storage_size']}TB")
    
    # Suggestion for provider selection
    if data['provider'] == "Microsoft Azure Blob Storage" and current_footprint > 100:
        suggestions.append("Google Cloud Storage has lower energy consumption per TB in most regions")
    
    # Suggestion for region selection
    current_region_carbon = CLOUD_PROVIDERS[data['provider']]["regions"][data['region']]
    lowest_carbon_region = min(
        CLOUD_PROVIDERS[data['provider']]["regions"].items(),
        key=lambda x: x[1]
    )
    if current_region_carbon > lowest_carbon_region[1] * 1.2:
        suggestions.append(f"Consider switching to {lowest_carbon_region[0]} region with {lowest_carbon_region[1]} kg/kWh intensity")
    
    if not suggestions:
        suggestions.append("Your cloud storage configuration is relatively efficient!")
    
    return suggestions
################################  ChatBOT  #######################################################################################################################################################################################################################

# Load environment variables
load_dotenv()

# Initialize Flask app


# Configure logging


# Load knowledge base
with open('knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Define regex-based formula detection
formula_patterns = {
    r"(electricity.*formula|formula.*electricity|calculate.*electricity)": 
        "Electricity Emissions = Energy Consumption (kWh) × Emission Factor (kg CO₂/kWh)",
    r"(data transfer.*formula|formula.*data transfer|calculate.*data transfer)": 
        "Data Transfer Emissions = Data Transferred (GB) × Emission Factor (g CO₂/GB)",
    r"(file storage.*formula|formula.*file storage|calculate.*file storage)": 
        "File Storage Emissions = Storage Size (GB) × Duration (hours) × Emission Factor (g CO₂/GB/hr)",
    r"(cloud compute.*formula|formula.*cloud compute|calculate.*cloud compute)": 
        "Cloud Compute Emissions = Compute Time (hours) × Power Consumption (kW) × Emission Factor (kg CO₂/kWh)",
    r"(all formulas|show formulas|list formulas|give.*formulas)": 
        """Here are key sustainability-related emissions formulas:

1. **Electricity Emissions**  
   = Energy Consumption (kWh) × Emission Factor (kg CO₂/kWh)

2. **Data Transfer Emissions**  
   = Data Transferred (GB) × Emission Factor (g CO₂/GB)

3. **File Storage Emissions**  
   = Storage Size (GB) × Duration (hours) × Emission Factor (g CO₂/GB/hr)

4. **Cloud Compute Emissions**  
   = Compute Time (hours) × Power (kW) × Emission Factor (kg CO₂/kWh)

Would you like help using one of these formulas?"""
}

def get_gemini_response(query):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        system_prompt = """You are a specialized sustainability assistant focusing on digital carbon emissions and SDGs 7, 9, 12, and 13.
        Provide concise, accurate information about sustainability, clean energy, responsible consumption, 
        climate action, and technology's environmental impact. Keep responses informative, practical, and under 150 words."""
        chat = model.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I understand. I'm now a specialized sustainability assistant focusing on digital carbon emissions and SDGs."]}
        ])
        response = chat.send_message(query)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return None

def get_bot_response(user_message):
    user_message = user_message.lower()

    # Check formula queries using regex
    for pattern, response in formula_patterns.items():
        if re.search(pattern, user_message):
            return response

    # === Predefined pattern responses ===
    greetings = ['hello', 'hi', 'hey', 'greetings', 'howdy']
    if any(greeting in user_message for greeting in greetings):
        return random.choice([
            "Hello! I'm your Sustainability Assistant. How can I help you today?",
            "Hi there! I'm here to answer your questions about sustainability and SDGs. What would you like to know?",
            "Greetings! I'm your guide for sustainable development topics. What are you curious about?"
        ])

    if ('tech' in user_message or 'digital' in user_message) and ('carbon' in user_message or 'emission' in user_message):
        if any(word in user_message for word in ['what is', 'explain', 'tell me about', 'define', 'what are']):
            return """Tech carbon emissions (aka digital carbon footprint) refer to the environmental impact of digital activities like data centers, video streaming, cloud computing, and device usage.

Would you like to know how to reduce your digital emissions?"""
        elif any(phrase in user_message for phrase in ['how to reduce', 'ways to reduce', 'reduce', 'minimize']):
            return """Ways to reduce tech emissions include:
- Power-saving mode on devices
- Stream in lower quality
- Use renewable energy
- Clear unnecessary files and emails
- Optimize cloud usage

Would you like help applying these?"""
        elif any(word in user_message for word in ['impact', 'effect', 'damage', 'problem']):
            return """Digital emissions impact the planet by:
- Raising electricity use
- Requiring rare materials
- Increasing e-waste
- Contributing to climate change

Need strategies to reduce impact?"""

    if 'carbon' in user_message or 'emission' in user_message:
        if 'cycle' in user_message and any(word in user_message for word in ['what', 'explain', 'define']):
            return """The carbon cycle is nature's way of moving carbon through the air, oceans, and land. It includes photosynthesis, respiration, decomposition, and fossil fuel formation."""
        elif any(word in user_message for word in ['define', 'what is', 'explain']):
            return """Carbon is a vital element in all life and plays a central role in climate change due to CO₂ emissions."""
        elif any(phrase in user_message for phrase in ['how to reduce', 'ways to reduce', 'minimize']):
            return """To reduce carbon emissions:
- Use renewable energy
- Use public transport
- Consume sustainably
- Reduce energy usage

Want detailed tips?"""

    if 'sdg' in user_message:
        if '7' in user_message or 'clean energy' in user_message:
            return """SDG 7 promotes affordable and clean energy for all, focusing on renewable sources and energy efficiency."""
        elif '9' in user_message or 'innovation' in user_message:
            return """SDG 9 encourages sustainable industrialization, infrastructure, and innovation."""
        elif '12' in user_message or 'consumption' in user_message:
            return """SDG 12 aims to ensure sustainable consumption and reduce environmental degradation."""
        elif '13' in user_message or 'climate' in user_message:
            return """SDG 13 focuses on urgent climate action and strengthening resilience to environmental risks."""

    if 'digital' in user_message and ('carbon' in user_message or 'emission' in user_message):
        return """Digital emissions include the CO₂ generated by devices, data centers, internet usage, and cloud services. Want tips to reduce them?"""

    goodbyes = ['bye', 'goodbye', 'see you', 'exit', 'quit']
    if any(goodbye in user_message for goodbye in goodbyes):
        return random.choice([
            "Goodbye! Feel free to return if you have more sustainability questions.",
            "Take care! Remember that small sustainable actions make a big difference.",
            "Bye for now! Keep thinking green!"
        ])

    thanks = ['thank', 'thanks', 'appreciate']
    if any(thank in user_message for thank in thanks):
        return random.choice([
            "You're welcome! I'm happy to help with sustainability topics.",
            "Glad I could be of assistance! Any other questions about sustainable development?",
            "My pleasure! Together we can make a more sustainable future."
        ])

    help_requests = ['help', 'assist', 'guide', 'what can you do']
    if any(help_req in user_message for help_req in help_requests) and len(user_message) < 15:
        return """I can help with:
- Emissions formulas
- Digital carbon footprint
- Climate change actions
- Sustainable goals like SDG 7, 9, 12, 13

Ask me anything!"""

    # Fallback to Gemini
    gemini_response = get_gemini_response(user_message)
    if gemini_response:
        return gemini_response

    return random.choice([
        "I'm not sure I understand. Could you rephrase your question?",
        "I don't have that specific info yet. Try asking about sustainability formulas or SDGs.",
        "Still learning about that area. Try something like 'formula for electricity emissions'."
    ])

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def respond():
    user_message = request.json.get('message', '')
    response = get_bot_response(user_message)
    return jsonify({'response': response})




################################################################################################################################################################################################################################################################
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
