import gradio as gr
import joblib
import numpy as np
import os
# optional data and plotting libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from io import BytesIO
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    BytesIO = None
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False

# Attempt to load the model and capture any load error for user-friendly messages
model_path = "model(1).pkl"
model = None
model_features = []
load_error = None
model_expected_n_features = None

# Default feature order (used as a fallback if model file has no metadata)
DEFAULT_FEATURE_ORDER = [
    'Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price',
    'CPI', 'Unemployment', 'Year', 'Month', 'Week'
]


def _load_model(path):
    global model, load_error, model_path, model_features
    global model_expected_n_features
    model_path = path or model_path
    if os.path.exists(model_path):
        try:
            loaded = joblib.load(model_path)
            # If the saved object is a dict with metadata, unpack it
            if isinstance(loaded, dict) and 'model' in loaded:
                model = loaded['model']
                # prefer explicit features metadata
                model_features = loaded.get('features')
                # infer feature count from model if metadata not present
                if model_features is None:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = list(model.feature_names_in_)
                    elif hasattr(model, 'coef_'):
                        model_expected_n_features = len(getattr(model, 'coef_', []))
                        model_features = DEFAULT_FEATURE_ORDER[:model_expected_n_features]
                    else:
                        model_features = DEFAULT_FEATURE_ORDER
                load_error = None
                features_str = ', '.join(map(str, model_features)) if model_features else ''
                return f"Loaded model: {os.path.basename(model_path)}", features_str
            # otherwise assume it's a raw sklearn estimator
            elif hasattr(loaded, 'predict'):
                model = loaded
                # try to infer feature names/count from the estimator
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    model_expected_n_features = len(model_features)
                elif hasattr(model, 'coef_'):
                    model_expected_n_features = len(getattr(model, 'coef_', []))
                    model_features = DEFAULT_FEATURE_ORDER[:model_expected_n_features]
                else:
                    model_features = DEFAULT_FEATURE_ORDER
                load_error = None
                features_str = ', '.join(map(str, model_features)) if model_features else ''
                return f"Loaded legacy model: {os.path.basename(model_path)}", features_str
            else:
                model = None
                model_features = []
                load_error = f"Unrecognized model format in {model_path}"
                return load_error, ''
        except Exception as e:
            model = None
            model_features = []
            load_error = str(e)
            return f"Error loading model: {e}", ''
    else:
        model = None
        model_features = []
        load_error = f"model file not found at {model_path}"
        return load_error, ''


# initial attempt
_load_model(model_path)

# 🟦 TAB 1: Prediction
def predict(store, dept, holiday, temp, fuel, cpi, unemployment, year, month, week):
    # Input validation and conversion
    try:
        store = int(store)
        dept = int(dept)
        holiday = 1 if bool(holiday) else 0
        temp = float(temp)
        fuel = float(fuel)
        cpi = float(cpi)
        unemployment = float(unemployment)
        year = int(year)
        month = int(month)
        week = int(week)
    except Exception as e:
        return f"Input conversion error: {e}", ""

    if not (1 <= month <= 12):
        return "Month must be 1-12", ""
    if not (1 <= week <= 53):
        return "Week must be 1-53", ""

    # If the model failed to load, return a helpful message instead of crashing
    if model is None:
        return f"Model not loaded: {load_error}", ""

    # Build input vector according to the trained model's feature list
    # map available inputs (several key variants) to a dict
    provided = {
        'store': store,
        'Store': store,
        'dept': dept,
        'Dept': dept,
        'is_holiday': holiday,
        'IsHoliday': holiday,
        'holiday': holiday,
        'Temperature': temp,
        'temperature': temp,
        'Fuel_Price': fuel,
        'fuel': fuel,
        'CPI': cpi,
        'cpi': cpi,
        'Unemployment': unemployment,
        'unemployment': unemployment,
        'Year': year,
        'year': year,
        'Month': month,
        'month': month,
        'Week': week,
        'week': week
    }

    X_row = []
    missing = []
    for feat in model_features or []:
        # try exact, then lowercase, then replace spaces/underscores
        val = None
        if feat in provided:
            val = provided[feat]
        else:
            key_variants = [feat.lower(), feat.replace(' ', '_'), feat.replace('_', ' ').lower()]
            for kv in key_variants:
                if kv in provided:
                    val = provided[kv]
                    break
        if val is None:
            # fallback to zero when feature not provided by UI
            missing.append(feat)
            val = 0
        X_row.append(val)

    if missing:
        # still proceed but inform the user in the output
        missing_msg = f"(Warning: missing inputs for features: {missing}; using 0)"
    else:
        missing_msg = ""

    X = np.array([X_row], dtype=float)
    # If model expects a different number of features, adapt by truncating or padding
    if model is not None:
        expected = model_expected_n_features
        # try to infer expected features from model if not set
        if expected is None:
            if hasattr(model, 'feature_names_in_'):
                expected = len(getattr(model, 'feature_names_in_', []))
            elif hasattr(model, 'coef_'):
                expected = len(getattr(model, 'coef_', []))
        if expected is not None:
            if X.shape[1] > expected:
                X = X[:, :expected]
                missing_msg += f" (Note: truncated to first {expected} features)"
            elif X.shape[1] < expected:
                # pad with zeros
                pad = np.zeros((X.shape[0], expected - X.shape[1]))
                X = np.hstack([X, pad])
                missing_msg += f" (Note: padded with {expected - X.shape[1]} zeros)"
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        return f"Prediction error: {e}", ""

    try:
        pred_value = float(pred)
    except Exception:
        pred_value = pred

    if isinstance(pred_value, (int, float)):
        if pred_value < 5000:
            level = "Low Demand"
        elif pred_value < 15000:
            level = "Medium Demand"
        else:
            level = "High Demand"
        pred_str = f"{pred_value:.2f} {missing_msg}" if missing_msg else f"{pred_value:.2f}"
    else:
        level = "Unknown"
        pred_str = str(pred_value)

    return pred_str, level


# 🟩 TAB 2
def analysis():
    return "Sales trends, top stores, seasonal behavior (use the 'Show analysis' button to render charts)"


def _fig_to_image(fig):
    if not (MATPLOTLIB_AVAILABLE and PIL_AVAILABLE):
        return None
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def run_analysis(sample_n=500):
    if not PANDAS_AVAILABLE:
        return "pandas not installed. Install with: pip install pandas matplotlib Pillow", None
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        return "matplotlib or Pillow not installed. Install with: pip install matplotlib Pillow", None

    # Load and merge data to compute aggregate charts
    try:
        train = pd.read_csv('train.csv', parse_dates=['Date'])
        wm = pd.read_csv('walmart.csv', parse_dates=['Date'])
    except Exception as e:
        return f"Error loading data: {e}", None

    df = pd.merge(train, wm, on=['Store', 'Date'], how='left')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    # Aggregate monthly sales
    monthly = df.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index()
    monthly['YM'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str) + '-01')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(monthly['YM'], monthly['Weekly_Sales'], marker='o')
    ax.set_title('Total Weekly Sales by Month')
    ax.set_ylabel('Total Weekly Sales')
    ax.grid(True)

    img = _fig_to_image(fig)
    summary = f"Records merged: {len(df)}\nMonthly points: {len(monthly)}"
    return summary, img


def compute_performance(sample_n=500):
    if model is None:
        return "Model not loaded; cannot compute performance.", None
    if not PANDAS_AVAILABLE:
        return "pandas not installed. Install with: pip install pandas matplotlib Pillow", None
    if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
        return "matplotlib or Pillow not installed. Install with: pip install matplotlib Pillow", None

    try:
        train = pd.read_csv('train.csv', parse_dates=['Date'])
        wm = pd.read_csv('walmart.csv', parse_dates=['Date'])
    except Exception as e:
        return f"Error loading data: {e}", None

    df = pd.merge(train, wm, on=['Store', 'Date'], how='left')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['IsHoliday'] = df['IsHoliday_x'].astype(str).map({'TRUE': 1, 'FALSE': 0}) if 'IsHoliday_x' in df.columns else df['IsHoliday']

    feats = model_features or DEFAULT_FEATURE_ORDER
    X_rows = []
    y = []
    for _, r in df.iterrows():
        row = []
        for f in feats:
            if f in r and pd.notna(r[f]):
                row.append(r[f])
            else:
                # try common lower/alt names
                alt = f.lower()
                if alt in r and pd.notna(r[alt]):
                    row.append(r[alt])
                else:
                    row.append(0)
        X_rows.append(row)
        y.append(r['Weekly_Sales'])
        if len(X_rows) >= sample_n:
            break

    if len(X_rows) == 0:
        return "No data available for performance computation.", None

    X = np.array(X_rows, dtype=float)
    # adapt to model expected features
    expected = model_expected_n_features
    if expected is None:
        if hasattr(model, 'feature_names_in_'):
            expected = len(getattr(model, 'feature_names_in_', []))
        elif hasattr(model, 'coef_'):
            expected = len(getattr(model, 'coef_', []))
    if expected is not None:
        if X.shape[1] > expected:
            X = X[:, :expected]
        elif X.shape[1] < expected:
            pad = np.zeros((X.shape[0], expected - X.shape[1]))
            X = np.hstack([X, pad])

    try:
        preds = model.predict(X)
    except Exception as e:
        return f"Prediction failed during performance compute: {e}", None

    y = np.array(y[: len(preds)])
    mae = float(np.mean(np.abs(y - preds)))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))

    # plot actual vs predicted for sample
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(y, label='Actual', alpha=0.7)
    ax.plot(preds, label='Predicted', alpha=0.7)
    ax.set_title('Actual vs Predicted (sample)')
    ax.legend()
    ax.grid(True)

    img = _fig_to_image(fig)
    summary = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nSample size: {len(y)}"
    return summary, img

# 🟨 TAB 3
def performance():
    return "Linear Regression Model\nMAE + RMSE shown from training"

# 🟥 TAB 4
def forecast():
    return "Future 7–30 day demand forecast simulation"

with gr.Blocks() as app:

    gr.Markdown("# 🛒 Walmart Sales Forecasting Dashboard")

    with gr.Tab("🧾 Prediction"):
        # Model controls
        model_file = gr.Textbox(label="Model filename", value=model_path)
        reload_btn = gr.Button("Reload Model")
        # call loader to get initial values for status and feature list
        status_val, features_val = _load_model(model_path)
        model_status = gr.Textbox(label="Model status", value=status_val if status_val else ("Error: " + str(load_error)), interactive=False)
        model_features_box = gr.Textbox(label="Model features (expected)", value=features_val, interactive=False)

        # Inputs with labels and brief guidance
        store = gr.Number(label="Store ID", value=1, precision=0)
        dept = gr.Number(label="Department ID", value=1, precision=0)
        holiday = gr.Checkbox(label="Is Holiday? (check for yes)")
        temp = gr.Number(label="Temperature (°C)", value=20.0)
        fuel = gr.Number(label="Fuel Price ($)", value=3.0)
        cpi = gr.Number(label="CPI Index", value=250.0)
        unemployment = gr.Number(label="Unemployment Rate (%)", value=5.0)
        year = gr.Number(label="Year", value=2024, precision=0)
        month = gr.Number(label="Month (1-12)", value=1, precision=0)
        week = gr.Number(label="Week of Year (1-53)", value=1, precision=0)

        btn = gr.Button("Predict")
        out1 = gr.Textbox(label="Predicted Sales", interactive=False)
        out2 = gr.Textbox(label="Demand Level", interactive=False)

        # Hook up reload button to try loading a different filename at runtime
        reload_btn.click(_load_model, inputs=[model_file], outputs=[model_status, model_features_box])

        btn.click(
            predict,
            inputs=[store, dept, holiday, temp, fuel, cpi, unemployment, year, month, week],
            outputs=[out1, out2]
        )

    with gr.Tab("📊 Analysis"):
        analysis_button = gr.Button("Show analysis")
        analysis_text = gr.Textbox(value=analysis(), label="Analysis summary", interactive=False)
        analysis_plot = gr.Image(label="Analysis plot")
        analysis_button.click(lambda *_: run_analysis(), inputs=[], outputs=[analysis_text, analysis_plot])

    with gr.Tab("🤖 Model Performance"):
        perf_button = gr.Button("Compute Metrics")
        perf_text = gr.Textbox(value=performance(), label="Model performance summary", interactive=False)
        perf_plot = gr.Image(label="Performance plot")
        perf_button.click(lambda *_: compute_performance(), inputs=[], outputs=[perf_text, perf_plot])

    with gr.Tab("📈 Forecasting"):
        forecast_button = gr.Button("Run forecast")
        forecast_text = gr.Textbox(value=forecast(), label="Forecast summary", interactive=False)
        forecast_plot = gr.Image(label="Forecast plot")
        def _run_forecast():
            # simple forecast: use model to predict next 14 days using last known features
            if model is None:
                return "Model not loaded; cannot forecast.", None
            if not PANDAS_AVAILABLE:
                return "pandas not installed. Install with: pip install pandas", None
            try:
                train = pd.read_csv('train.csv', parse_dates=['Date'])
                wm = pd.read_csv('walmart.csv', parse_dates=['Date'])
            except Exception as e:
                return f"Error loading data: {e}", None

            df = pd.merge(train, wm, on=['Store', 'Date'], how='left')
            df = df.sort_values('Date')
            last = df.iloc[-1]
            # build simple repeated features for next 14 weekly points
            weeks = 14
            X_rows = []
            for i in range(weeks):
                row = []
                for f in (model_features or DEFAULT_FEATURE_ORDER):
                    if f == 'Year':
                        row.append(last['Date'].year + (i // 52))
                    elif f == 'Month':
                        m = ((last['Date'].month - 1 + i) % 12) + 1
                        row.append(m)
                    elif f == 'Week':
                        w = int((last['Date'].isocalendar().week + i - 1) % 53) + 1
                        row.append(w)
                    elif f in last and pd.notna(last[f]):
                        row.append(last[f])
                    else:
                        alt = f.lower()
                        row.append(last[alt] if alt in last and pd.notna(last[alt]) else 0)
                X_rows.append(row)

            X = np.array(X_rows, dtype=float)
            expected = model_expected_n_features
            if expected is None:
                if hasattr(model, 'feature_names_in_'):
                    expected = len(getattr(model, 'feature_names_in_', []))
                elif hasattr(model, 'coef_'):
                    expected = len(getattr(model, 'coef_', []))
            if expected is not None:
                if X.shape[1] > expected:
                    X = X[:, :expected]
                elif X.shape[1] < expected:
                    pad = np.zeros((X.shape[0], expected - X.shape[1]))
                    X = np.hstack([X, pad])

            try:
                preds = model.predict(X)
            except Exception as e:
                return f"Forecast failed: {e}", None

            # simple plot
            if MATPLOTLIB_AVAILABLE and PIL_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(range(len(preds)), preds, marker='o')
                ax.set_title('Forecast (next 14 points)')
                ax.set_ylabel('Predicted Weekly Sales')
                ax.grid(True)
                img = _fig_to_image(fig)
            else:
                img = None

            return f"Forecast points: {len(preds)}", img

        forecast_button.click(lambda *_: _run_forecast(), inputs=[], outputs=[forecast_text, forecast_plot])

app.launch(share=True)