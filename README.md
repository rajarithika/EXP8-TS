# EXPERIMENT 8: TIME SERIES MODELING AND FORECASTING WITH ARMA PROCESSES

**Name:** RAJA RITHIKA M

**Reg. No.:** 2305001029

## 1. Aim

To model and forecast a time series using an ARMA(p, q) process.

## 2. Software Required

- Python 3
- Google Colab or Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn

## 3. Procedure

1. Import the required libraries.
2. Generate an ARMA(1,1) time series.
3. Plot the series and examine ACF and PACF.
4. Split the data into training and testing sets.
5. Fit an ARMA(1,1) model.
6. Forecast future values and calculate MAE.
7. Compare candidate models using AIC and BIC.

## 4. Program

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error

# Generate ARMA(1,1) time series
np.random.seed(42)

ar = np.array([1, -0.7])
ma = np.array([1, 0.4])

arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=200)
series = pd.Series(data)

# Plot time series
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Simulated ARMA(1,1) Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, ax=ax[1])
ax[0].set_title("ACF")
ax[1].set_title("PACF")
plt.tight_layout()
plt.show()

# Split data into training and testing
train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# Fit ARMA(1,1)
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()

print(model_fit.summary())
print("\nEstimated Parameters:")
print(model_fit.params)

# Forecast test period
forecast = model_fit.forecast(steps=len(test))
print("\nForecast Values:")
print(forecast)

# Calculate MAE
mae = mean_absolute_error(test, forecast)
print("\nMean Absolute Error:", round(mae, 4))

# Plot actual and forecast values
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Data")
plt.plot(test.index, np.asarray(forecast), label="ARMA Forecast")
plt.title("ARMA(1,1) Forecast")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Compare candidate models
fit_ar1 = ARIMA(train, order=(1, 0, 0)).fit()
fit_ma1 = ARIMA(train, order=(0, 0, 1)).fit()
fit_arma11 = ARIMA(train, order=(1, 0, 1)).fit()
fit_arma21 = ARIMA(train, order=(2, 0, 1)).fit()

print("\nModel Comparison")
for name, fitted in [
    ("AR(1)", fit_ar1),
    ("MA(1)", fit_ma1),
    ("ARMA(1,1)", fit_arma11),
    ("ARMA(2,1)", fit_arma21)
]:
    print(
        name,
        "| AIC:", round(fitted.aic, 3),
        "| BIC:", round(fitted.bic, 3)
    )

print("\nARMA modeling and forecasting completed.")
```

## 5. Expected Output

- Simulated ARMA(1,1) time-series graph.
<img width="978" height="458" alt="image" src="https://github.com/user-attachments/assets/a0a88c08-7c8e-4500-87fb-2f5674a3bcbd" />

- ACF and PACF plots.
<img width="980" height="335" alt="image" src="https://github.com/user-attachments/assets/75f3e6dc-937a-43ab-bc19-7e43ee8ba59f" />

- Training and testing observations.
<img width="352" height="58" alt="image" src="https://github.com/user-attachments/assets/c5cd9dd0-c901-4625-a24d-8402ed18b047" />

- ARMA model summary and estimated parameters.
<img width="857" height="500" alt="image" src="https://github.com/user-attachments/assets/2fabbe32-9ce7-4670-a06f-76c9d8e6b6b4" />

- Forecast values for the test period.
<img width="925" height="703" alt="image" src="https://github.com/user-attachments/assets/df9b75c6-9604-4bf5-9ce1-aa1d02bc886c" />
<img width="1065" height="431" alt="image" src="https://github.com/user-attachments/assets/bfe1704b-c8c3-42a9-a588-3c2309b545d2" />

- Mean Absolute Error (MAE).
<img width="420" height="23" alt="image" src="https://github.com/user-attachments/assets/3be45ce3-c9fb-4cc6-887e-5e1e1005f2fa" />


- Actual-versus-forecast graph.
<img width="972" height="456" alt="image" src="https://github.com/user-attachments/assets/5d61fa6c-cff1-4c25-9a29-b26cec205140" />


- AIC and BIC values for candidate models.
<img width="642" height="175" alt="image" src="https://github.com/user-attachments/assets/086b0bf2-fa9d-45d8-8480-e8338ac5a5e5" />


## 6. Observation

The simulated time series fluctuates around an approximately constant mean. ACF and PACF plots help examine its correlation structure. The ARMA model is fitted using training data, and its forecast is compared with the test data using MAE.

AIC and BIC are used to compare candidate models fitted to the same training data. Lower values are generally preferred, but forecast accuracy and diagnostic checks should also be considered.

## 7. Result

Thus, an ARMA(1,1) time-series model was simulated and fitted using Python. Forecasting was performed, the Mean Absolute Error was calculated, and candidate models were compared using AIC and BIC.
