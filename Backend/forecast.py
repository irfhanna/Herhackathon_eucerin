import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# LOAD CSV
# ---------------------------
def load_csv(path, date_column="timestamp"):
    df = pd.read_csv(path)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    return df


# ---------------------------
# CREATE TIME SERIES
# ---------------------------
def create_time_series(df, date_column="timestamp", text_column="clean_text", concern_keyword="acne"):
    """
    Counts how often a keyword (e.g., 'acne', 'dry skin') appears per day.
    """
    df["contains_keyword"] = df[text_column].str.contains(concern_keyword, case=False, na=False)

    ts = (
        df.set_index(date_column)
          .resample("D")["contains_keyword"]
          .sum()
    )

    return ts


# ---------------------------
# FORECAST FUTURE TRENDS
# ---------------------------
def forecast_trend(ts, steps=30):
    """
    Forecast next X days based on ARIMA.
    """
    model = ARIMA(ts, order=(3,1,2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)
    return forecast


# ---------------------------
# VISUALIZE TREND
# ---------------------------
def plot_trend(ts, forecast, concern_keyword):
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values, label="Historical Data")
    plt.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
    plt.title(f"Trend Forecast for '{concern_keyword}' Mentions")
    plt.xlabel("Date")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def run_forecasting_pipeline(
    csv_path,
    concern_keyword="acne",
    date_column="timestamp",
    text_column="clean_text",
    forecast_days=15
):
    print("📄 Loading CSV...")
    df = load_csv(csv_path, date_column)

    print(f"🔍 Extracting time series for keyword: '{concern_keyword}'")
    ts = create_time_series(df, date_column, text_column, concern_keyword)

    print(ts.tail())

    print("🔮 Forecasting future trend...")
    forecast = forecast_trend(ts, steps=forecast_days)

    print("\n📊 Forecast Results:")
    print(forecast)

    plot_trend(ts, forecast, concern_keyword)

    print("\n📌 Insight:")
    direction = "increasing 📈" if forecast.mean() > ts.mean() else "decreasing 📉"
    print(f"The interest in '{concern_keyword}' is predicted to be **{direction}** over the next {forecast_days} days.")

    return ts, forecast


# ---------------------------
# EXECUTE
# ---------------------------
if __name__ == "__main__":
    run_forecasting_pipeline(
        csv_path="./skin_social_media_data.csv",     # <-- CHANGE THIS
        concern_keyword="gel",       # set any keyword: "pigmentation", "dry skin", "ceramide"
        forecast_days=15
    )
