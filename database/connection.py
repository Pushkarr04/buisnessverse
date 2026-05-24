import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from utils.data_generator import generate_business_data

# Define database file path
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "business_pulse.db")

# Create SQLite SQLAlchemy connection engine
# Use check_same_thread=False to support Streamlit's multi-threaded requests safely
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db_session():
    """
    Context manager to yield a clean database session and close it afterwards.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def get_raw_connection():
    """
    Returns a raw sqlite3 connection for operations that benefit from direct DB-API execution.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_and_seed_db(force=False):
    """
    Initializes the database, checks if tables are populated, and seeds realistic data if empty.
    Creates essential secondary indexes to optimize analytical SQL window functions and complex queries.
    """
    # Create database directory if it doesn't exist
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    # Check if tables exist and have data
    db_needs_seeding = True
    if not force and os.path.exists(DB_PATH):
        try:
            with engine.connect() as conn:
                res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='sales';"))
                table_exists = res.fetchone()
                if table_exists:
                    res_count = conn.execute(text("SELECT count(*) FROM sales;"))
                    count = res_count.fetchone()[0]
                    if count > 0:
                        db_needs_seeding = False
                        print(f"Database already initialized and contains {count} transactions.")
        except Exception as e:
            print("Error validating existing database, re-seeding...", e)

    if db_needs_seeding:
        print("Initializing database and seeding multi-year business data. Please wait...")
        
        # Generate datasets
        df_p, df_c, df_s, df_m, df_f = generate_business_data()
        
        # Write tables to SQLite
        with engine.begin() as conn:
            df_p.to_sql("products", conn, if_exists="replace", index=False)
            df_c.to_sql("customers", conn, if_exists="replace", index=False)
            df_s.to_sql("sales", conn, if_exists="replace", index=False)
            df_m.to_sql("marketing_campaigns", conn, if_exists="replace", index=False)
            df_f.to_sql("financials", conn, if_exists="replace", index=False)
            
            # Create indexes to speed up advanced SQL query analysis (CTEs, Window Functions, Joins)
            print("Creating performance indexes...")
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_cust ON sales(customer_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_prod ON sales(product_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cust_signup ON customers(signup_date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cust_segment ON customers(segment);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_prod_cat ON products(category);"))
            
        print("Database seeded and optimized successfully!")

# Ensure database is created and seeded upon import
init_and_seed_db()

def check_custom_tables_exist():
    """
    Checks if the custom business tables are created and populated.
    """
    required_tables = ["custom_sales", "custom_products", "custom_customers", "custom_marketing_campaigns", "custom_financials"]
    try:
        with engine.connect() as conn:
            for table in required_tables:
                res = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';"))
                if not res.fetchone():
                    return False
                res_count = conn.execute(text(f"SELECT count(*) FROM {table};"))
                if res_count.fetchone()[0] == 0:
                    return False
            
            # Also check if custom_sales has the required column 'sale_date' to ensure it's not a raw unmapped upload
            res_cols = conn.execute(text("PRAGMA table_info(custom_sales);"))
            cols = [col[1] for col in res_cols.fetchall()]
            if 'sale_date' not in cols or 'product_id' not in cols or 'total_amount' not in cols:
                return False
            
            return True
    except Exception:
        pass
    return False


