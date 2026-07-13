-- Enable Timescale extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

--------------------------------------------------
-- Patients table
--------------------------------------------------

CREATE TABLE IF NOT EXISTS patients (
    patient_id SERIAL PRIMARY KEY,
    name TEXT,
    age INT,
    gender TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

--------------------------------------------------
-- Sensor data table (UPDATED)
--------------------------------------------------

CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL,
    patient_id INTEGER NOT NULL,

    hr DOUBLE PRECISION,
    temp DOUBLE PRECISION,
    rr DOUBLE PRECISION,
    spo2 DOUBLE PRECISION,

    hrv DOUBLE PRECISION,
    rrv DOUBLE PRECISION,
    movement DOUBLE PRECISION,

    timestamp TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (id, timestamp)
);

--------------------------------------------------
-- Convert to Timescale hypertable
--------------------------------------------------

SELECT create_hypertable('sensor_data', 'timestamp', if_not_exists => TRUE);

--------------------------------------------------
-- Predictions table (UPDATED)
--------------------------------------------------

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER,

    timestamp TIMESTAMPTZ DEFAULT NOW(),

    predicted_sepsis BOOLEAN,
    current_risk_score DOUBLE PRECISION,

    risk_scores JSONB,
    score_timestamps JSONB
);