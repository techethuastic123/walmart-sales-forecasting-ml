import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

FEATURES = ['Store','Dept','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Year','Month','Week']

def create_and_save(path='model(1).pkl'):
    n = 1000
    X = np.zeros((n, len(FEATURES)))
    # Store, Dept
    X[:,0] = np.random.randint(1, 50, size=n)
    X[:,1] = np.random.randint(1, 100, size=n)
    # IsHoliday
    X[:,2] = np.random.randint(0, 2, size=n)
    # Temperature
    X[:,3] = np.random.uniform(-10, 40, size=n)
    # Fuel_Price
    X[:,4] = np.random.uniform(2, 5, size=n)
    # CPI
    X[:,5] = np.random.uniform(200, 300, size=n)
    # Unemployment
    X[:,6] = np.random.uniform(3, 15, size=n)
    # Year
    X[:,7] = np.random.randint(2018, 2025, size=n)
    # Month
    X[:,8] = np.random.randint(1, 13, size=n)
    # Week
    X[:,9] = np.random.randint(1, 54, size=n)

    # synthetic target
    y = (
        X[:,0]*10 + X[:,1]*5 + X[:,3]*20 + X[:,4]*100 + X[:,6]*50 + X[:,8]*100
        + np.random.normal(0, 1000, n)
    )

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump({'model': model, 'features': FEATURES}, path)
    print(f"Saved dummy model to {path}")

if __name__ == '__main__':
    create_and_save()
