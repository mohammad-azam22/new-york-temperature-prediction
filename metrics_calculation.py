import datetime
import time
import random
import pandas as pd
import psycopg
import joblib

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift, DriftedColumnsCount, MissingValueCount

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

SEND_TIMEOUT = 1
rand = random.Random()

# -----------------------------
# Load data and model
# -----------------------------
reference_data = pd.read_csv("./data/reference_data.csv", index_col=0)
reference_data.date = pd.to_datetime(reference_data.date)

with open("models/random_forest_meteo.bin", "rb") as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_csv("./data/2025-01-01_2025-12-31.csv", index_col=0)
raw_data.date = pd.to_datetime(raw_data.date)

begin = datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

numerical_cols = [
    'dew_point_2m', 'pressure_msl', 'surface_pressure', 'cloud_cover',
    'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'visibility',
    'weather_code', 'precipitation', 'rain', 'snowfall', 'wind_speed_10m',
    'wind_gusts_10m', 'wind_direction_10m', 'vapour_pressure_deficit',
    'et0_fao_evapotranspiration'
]

categorical_cols = ["evapotranspiration"]

data_definition = DataDefinition(
    numerical_columns=numerical_cols + ['prediction'],
    categorical_columns=categorical_cols
)

report = Report(metrics=[
    ValueDrift(column='prediction'),
    DriftedColumnsCount(),
    MissingValueCount(column='prediction'),
])

# -----------------------------
# SQL
# -----------------------------
create_table_statement = """
CREATE TABLE IF NOT EXISTS metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_cols INTEGER,
    share_missing_values FLOAT,
    mean_absolute_error FLOAT,
    root_mean_squared_error FLOAT,
    r2_score FLOAT
);
"""

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "postgres",
    "password": "example",
    "dbname": "test"
}

# -----------------------------
# Tasks
# -----------------------------
def prep_db():
    # Create database if it does not exist
    with psycopg.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dbname="postgres",
        autocommit=True
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;",
                (DB_CONFIG["dbname"],)
            )
            exists = cur.fetchone()
            if not exists:
                cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']};")

    # Create metrics table
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS metrics;")
            cur.execute(create_table_statement)
        conn.commit()


def calculate_metrics_postgres(i):
    current_data = raw_data[
        (raw_data.date >= begin + datetime.timedelta(i)) &
        (raw_data.date < begin + datetime.timedelta(i + 1))
    ]

    current_data = current_data.fillna(0)
    current_data["prediction"] = model.predict(
        current_data[numerical_cols + categorical_cols]
    )

    reference_dataset = Dataset.from_pandas(reference_data, data_definition)
    current_dataset = Dataset.from_pandas(current_data, data_definition)

    snapshot = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset
    )
    result = snapshot.dict()

    timestamp = begin + datetime.timedelta(i)
    prediction_drift = result["metrics"][0]["value"]
    num_drifted_cols = result["metrics"][1]["value"]["count"]
    share_missing_values = result["metrics"][2]["value"]["share"]
    mae = mean_absolute_error(current_data["temperature_2m"], current_data["prediction"])
    rmse = root_mean_squared_error(current_data["temperature_2m"], current_data["prediction"])
    r2 = r2_score(current_data["temperature_2m"], current_data["prediction"])

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics (
                    timestamp,
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_values,
                    mean_absolute_error,
                    root_mean_squared_error,
                    r2_score
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    timestamp,
                    prediction_drift,
                    num_drifted_cols,
                    share_missing_values,
                    mae,
                    rmse,
                    r2
                )
            )
        conn.commit()


# -----------------------------
# Flow
# -----------------------------
def batch_monitoring_backfill():
    prep_db()

    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)

    upper_limit = 366 if begin.year == 2024 else 365

    for i in range(0, upper_limit):
        calculate_metrics_postgres(i)

        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()

        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)

        while last_send < new_send:
            last_send += datetime.timedelta(seconds=SEND_TIMEOUT)

        print(f"Data sent for: {begin + datetime.timedelta(i)} | Number of rows sent: {len(raw_data[(raw_data.date >= begin + datetime.timedelta(i)) & (raw_data.date < begin + datetime.timedelta(i + 1))])}")


if __name__ == "__main__":
    batch_monitoring_backfill()
