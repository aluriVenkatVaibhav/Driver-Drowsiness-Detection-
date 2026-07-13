from .db_connection import get_connection
import json

def insert_patient(name, age, gender):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO patients (name, age, gender)
        VALUES (%s,%s,%s)
        RETURNING patient_id
        """,
        (name, age, gender)
    )

    patient_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return patient_id


def insert_sensor_data(patient_id, hr, temp, rr, spo2, hrv, rrv, movement, timestamp):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO sensor_data
        (patient_id, hr, temp, rr, spo2, hrv, rrv, movement, timestamp)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (patient_id, hr, temp, rr, spo2, hrv, rrv, movement, timestamp))

    conn.commit()
    cur.close()
    conn.close()

def insert_prediction(patient_id, predicted_sepsis, current_risk_score, risk_scores, score_timestamps):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions
        (patient_id, predicted_sepsis, current_risk_score, risk_scores, score_timestamps)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        patient_id,
        predicted_sepsis,
        current_risk_score,
        json.dumps(risk_scores),
        json.dumps([str(ts) for ts in score_timestamps])
    ))

    conn.commit()
    cur.close()
    conn.close()


def get_latest_vitals(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT hr, rr, spo2, temp, hrv, rrv, timestamp
        FROM sensor_data
        WHERE patient_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (patient_id,)
    )

    result = cur.fetchone()

    cur.close()
    conn.close()

    return result


def get_day_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('5 minutes', timestamp) AS bucket,
            AVG(hr),
            AVG(spo2),
            AVG(temp),
            AVG(rr),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '1 day'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


def get_week_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('30 minutes', timestamp) AS bucket,
            AVG(hr),
            AVG(spo2),
            AVG(temp),
            AVG(rr),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '7 days'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


def get_month_timeline(patient_id):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            time_bucket('1 hours', timestamp) AS bucket,
            AVG(hr),
            AVG(spo2),
            AVG(temp),
            AVG(rr),
            AVG(hrv),
            AVG(rrv)
        FROM sensor_data
        WHERE patient_id = %s
        AND timestamp > NOW() - INTERVAL '30 days'
        GROUP BY bucket
        ORDER BY bucket
        """,
        (patient_id,)
    )

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results