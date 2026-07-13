import os
import psycopg2

DB_URL = os.getenv("DATABASE_URL")

def get_connection():
    if DB_URL is None:
        raise Exception("DATABASE_URL not set")
    return psycopg2.connect(DB_URL)