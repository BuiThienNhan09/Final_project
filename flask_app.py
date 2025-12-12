import os
import io
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import numpy as np
import pandas as pd
import joblib
import time
import urllib.request
import urllib.error
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me')

MODEL_PATH = 'flight_price_model.keras'
PREPROCESSOR_PATH = 'datasets/processed/preprocessor.pkl'

# Exchange rate management
# Default fallback (can be overridden by env var or remote fetch)
EXCHANGE_RATE_INR_TO_VND = float(os.environ.get('EXCHANGE_RATE_INR_TO_VND', '287.5'))
# How often (seconds) to refresh rate from remote API. Default 6 hours.
EXCHANGE_REFRESH_SECONDS = int(os.environ.get('EXCHANGE_REFRESH_SECONDS', str(6 * 3600)))
# Cache
_last_rate_fetch = 0.0
_cached_rate = EXCHANGE_RATE_INR_TO_VND

def fetch_remote_rate(base='INR', symbols='VND'):
    """Fetch latest exchange rate from exchangerate.host (returns float)"""
    url = f"https://api.exchangerate.host/latest?base={base}&symbols={symbols}"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.load(resp)
            # expected structure: {'rates': {'VND': 287.5}, ...}
            rate = data.get('rates', {}).get(symbols)
            if rate:
                return float(rate)
    except urllib.error.URLError:
        pass
    except Exception:
        pass
    return None

def get_current_rate():
    """Return cached exchange rate, refreshing from remote if stale.
    Falls back to env/default if fetch fails."""
    global _last_rate_fetch, _cached_rate
    now = time.time()
    if now - _last_rate_fetch > EXCHANGE_REFRESH_SECONDS:
        rate = fetch_remote_rate('INR', 'VND')
        if rate is not None and rate > 0:
            _cached_rate = float(rate)
            _last_rate_fetch = now
        else:
            # if fetch failed, only update the timestamp to avoid hammering remote
            _last_rate_fetch = now
    return _cached_rate

# Load model & preprocessor at startup
_model = None
_preprocessor = None

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    try:
        _model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load model: {e}")
    try:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        print(f"Warning: failed to load preprocessor: {e}")
else:
    print("Model or preprocessor not found; predictions will not work until files exist.")

FEATURE_COLUMNS = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']

def predict_df(df: pd.DataFrame):
    if _preprocessor is None or _model is None:
        raise RuntimeError('Model or preprocessor not loaded')
    X = _preprocessor.transform(df)
    preds_log = _model.predict(X, verbose=0)
    preds = np.expm1(preds_log[:, 0])
    return preds

@app.route('/')
def index():
    # show exchange rate note on the index page as well (use cached/remote rate)
    current_rate = get_current_rate()
    rate_str = f"1 INR = {current_rate:,.2f} VND"
    return render_template('index.html', exchange_rate=rate_str, exchange_value=current_rate)


@app.route('/rate')
def rate():
    """Return current cached exchange rate (INR->VND) as JSON."""
    try:
        current_rate = get_current_rate()
        return jsonify({'rate': float(current_rate), 'base': 'INR', 'target': 'VND'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # single-row form submission
    if 'file' in request.files and request.files['file'].filename:
        # CSV batch upload
        f = request.files['file']
        try:
            df = pd.read_csv(f)
        except Exception as e:
            flash(f'Failed to read CSV: {e}', 'danger')
            return redirect(url_for('index'))
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            flash(f'Missing columns in CSV: {missing}', 'warning')
            return redirect(url_for('index'))
        preds = predict_df(df[FEATURE_COLUMNS])
        df['predicted_price'] = preds
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

    # otherwise single form
    data = {}
    # map form names from the existing template to the model feature names
    # template uses 'flight_class' for ticket class
    form = request.form
    airline = form.get('airline', '')
    source_city = form.get('source_city', '')
    destination_city = form.get('destination_city', '')
    departure_time = form.get('departure_time', '')
    arrival_time = form.get('arrival_time', '')
    stops = form.get('stops', '')
    flight_class = form.get('flight_class', form.get('class', ''))
    # duration in the template may be entered in hours (e.g. 2.5), convert to minutes if reasonable
    try:
        duration_val = float(form.get('duration', '0'))
    except Exception:
        duration_val = 0.0
    if duration_val <= 24:
        duration_mins = float(duration_val) * 60.0
    else:
        duration_mins = float(duration_val)
    try:
        days_left = int(float(form.get('days_left', '0')))
    except Exception:
        days_left = 0

    data = {
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [flight_class],
        'duration': [duration_mins],
        'days_left': [days_left]
    }
    df = pd.DataFrame(data)
    try:
        preds = predict_df(df)
        price = float(preds[0])
    except Exception as e:
        flash(f'Prediction failed: {e}', 'danger')
        return redirect(url_for('index'))

    # Allow overriding exchange rate from the form (single or batch). If provided and valid, use it.
    rate_input = request.form.get('exchange_rate')
    try:
        rate_used = float(rate_input) if (rate_input is not None and str(rate_input).strip() != '') else get_current_rate()
    except Exception:
        rate_used = get_current_rate()

    # Compute VND conversion and format strings for display
    price_inr = price
    price_vnd = price_inr * rate_used
    price_inr_str = f"₹ {price_inr:,.0f}"
    price_vnd_str = f"₫ {price_vnd:,.0f}"
    rate_str = f"1 INR = {rate_used:,.2f} VND"

    return render_template('result.html', price_inr=price_inr_str, price_vnd=price_vnd_str, rate=rate_str, inputs=df.iloc[0].to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8501)), debug=True)
