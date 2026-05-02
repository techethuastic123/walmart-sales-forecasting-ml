import joblib
import numpy as np
import os

MODEL_PATH = 'model(1).pkl'

def test_load_and_predict():
    if not os.path.exists(MODEL_PATH):
        print('Model file not found:', MODEL_PATH)
        return 1
    data = joblib.load(MODEL_PATH)
    print('Loaded object type:', type(data))
    if isinstance(data, dict) and 'model' in data:
        model = data['model']
        features = data.get('features')
        print('Saved features:', features)
    else:
        model = data
        features = None
        print('No features metadata found')

    # Print model input expectations if available
    try:
        coef_len = len(getattr(model, 'coef_', []))
        print('Model expects number of features (approx):', coef_len)
    except Exception:
        pass

    X = np.array([[1, 1, 0, 20.0, 3.0, 250.0, 5.0, 2024, 1, 1]], dtype=float)
    print('Test input shape:', X.shape)
    try:
        pred = model.predict(X)[0]
        print('Prediction OK:', pred)
        return 0
    except Exception as e:
        print('Prediction failed:', e)
        return 2

if __name__ == '__main__':
    raise SystemExit(test_load_and_predict())
