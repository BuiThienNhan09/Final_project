import os
import importlib.util

# Load flask_app.py by path to avoid import issues when running from scripts/
here = os.path.dirname(__file__)
flask_app_path = os.path.abspath(os.path.join(here, '..', 'flask_app.py'))
spec = importlib.util.spec_from_file_location('flask_app', flask_app_path)
flask_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flask_app)
app = flask_app.app

sample = {
    'airline': 'Indigo',
    'source_city': 'Delhi',
    'destination_city': 'Mumbai',
    'departure_time': 'Morning',
    'arrival_time': 'Afternoon',
    'stops': 'zero',
    'flight_class': 'Economy',
    'duration': '2.5',
    'days_left': '15'
}

with app.test_client() as client:
    resp = client.post('/predict', data=sample, follow_redirects=True)
    print('STATUS:', resp.status_code)
    text = resp.get_data(as_text=True)
    print(text[:2000])
    # If large, also write to a debug file
    with open('scripts/test_predict_output.html', 'w', encoding='utf-8') as f:
        f.write(text)
    print('\nWrote full response to scripts/test_predict_output.html')
