from database.db_connection import get_connection

def init_database():

    conn = get_connection()
    cur = conn.cursor()

    with open("backend/database/schema.sql", "r") as f:
        cur.execute(f.read())

    conn.commit()

    cur.close()
    conn.close()

    print("Database initialized successfully")


if __name__ == "__main__":
    init_database()