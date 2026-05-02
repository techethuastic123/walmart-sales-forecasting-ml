import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def _safe_read_train(path="train.csv"):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
            print(f"Loaded training data from {path}, rows={len(df)}")
            return df
        except Exception as e:
            print(f"Error reading {path}: {e}")
    # fallback: return None
    return None


def _make_synthetic(n=200):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'Store': rng.randint(1, 10, size=n),
        'Dept': rng.randint(1, 20, size=n),
        'IsHoliday': rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        'Temperature': rng.normal(20, 5, size=n),
        'Fuel_Price': rng.normal(3, 0.2, size=n),
        'CPI': rng.normal(250, 5, size=n),
        'Unemployment': rng.normal(5, 1, size=n),
        'Date': pd.date_range('2020-01-01', periods=n, freq='W'),
    })
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    # generate Weekly_Sales with some signal
    df['Weekly_Sales'] = (
        df['Store'] * 1000 + df['Dept'] * 50 + df['IsHoliday'] * 2000 +
        (25 - df['Temperature']) * 10 + rng.normal(0, 500, size=n)
    )
    print(f"Created synthetic dataset rows={n}")
    return df


def preprocess(df):
    # ensure Date exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df['Date'] = pd.to_datetime('2020-01-01')

    # IsHoliday cleaning
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(str).str.upper()
        df['IsHoliday'] = df['IsHoliday'].replace({
            'TRUE': 1, 'FALSE': 0, 'FALS': 0, 'T': 1, 'F': 0
        })
        df['IsHoliday'] = pd.to_numeric(df['IsHoliday'], errors='coerce').fillna(0).astype(int)
    else:
        df['IsHoliday'] = 0

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    # fill numeric NaNs with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df


def train_and_save(df, out_path='model(1).pkl'):
    features = [
        'Store',
        'IsHoliday',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'Year',
        'Month',
        'Week'
    ]
    features = [c for c in features if c in df.columns]
    if 'Weekly_Sales' not in df.columns:
        raise ValueError('No Weekly_Sales column found in data')

    X = df[features]
    y = df['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    # save model together with feature list so the serving app can construct inputs
    payload = {
        'model': model,
        'features': features
    }
    joblib.dump(payload, out_path)
    print(f"Saved model + feature metadata to {out_path}")


def main():
    df = _safe_read_train('train.csv')
    if df is None:
        # try other likely filenames
        for alt in ['walmart.csv', 'test.csv', 'stores.csv']:
            df = _safe_read_train(alt)
            if df is not None:
                break

    if df is None:
        df = _make_synthetic(300)

    df = preprocess(df)

    try:
        train_and_save(df, out_path='model(1).pkl')
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == '__main__':
    main()