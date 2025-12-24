import pandas as pd
import os
import glob
import psycopg2
import io
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# --- Configuration ---
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBEDDING_MODEL = "models/text-embedding-004" 
VECTOR_DIMENSION = 768 

# Database Connection Details
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"

def import_raw_data():
    # 1. Load Data
    path = r"C:\RenewableEnergyAI\RenewableEnergyRevolution\data\countrywise"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # Select your columns (excluding embedding for now)
    col_interested = ['iso_code','country','year', \
                  'biofuel_consumption',\
                  'coal_consumption',\
                  'coal_production',\
                  'electricity_demand',\
                  'biofuel_electricity',\
                  'coal_electricity',\
                  'fossil_electricity',\
                  'gas_electricity',\
                  'hydro_electricity',\
                  'nuclear_electricity',\
                  'oil_electricity',\
                  'other_renewable_exc_biofuel_electricity',\
                  'other_renewable_electricity',\
                  'renewables_electricity',\
                  'solar_electricity',\
                  'wind_electricity',\
                  'electricity_generation',\
                  'fossil_fuel_consumption',\
                  'gas_consumption',\
                  'gas_production',\
                  'hydro_consumption',\
                  'low_carbon_consumption',\
                  'nuclear_consumption',\
                  'oil_consumption',\
                  'oil_production',\
                  'other_renewables_cons_change_pct',\
                  'other_renewables_share_energy',\
                  'other_renewables_cons_change_twh',\
                  'other_renewable_consumption',\
                  'other_renewables_share_elec',\
                  'primary_energy_consumption',\
                  'renewables_share_elec',\
                  'renewables_share_energy',\
                  'renewables_consumption',\
                  'solar_share_elec',\
                  'solar_share_energy',\
                  'solar_consumption',\
                  'wind_share_elec',\
                  'wind_share_energy',\
                  'wind_consumption',]
    raw_df = df[col_interested].copy()

    # 2. Bulk Copy to Postgres
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()
    try:
        cur.execute("DROP TABLE IF EXISTS allcountryenergy;")
        # Create table with only text/numeric columns
        cols_sql = ", ".join([f'"{c}" TEXT' for c in raw_df.columns])
        cur.execute(f"CREATE TABLE allcountryenergy ({cols_sql});")

        buffer = io.StringIO()
        raw_df.to_csv(buffer, index=False, header=False, sep='|')
        buffer.seek(0)
        
        cur.copy_expert("COPY allcountryenergy FROM STDIN WITH (FORMAT CSV, DELIMITER '|');", buffer)
        conn.commit()
        print("Raw data import successful.")
    finally:
        cur.close()
        conn.close()

import_raw_data()


def add_embeddings_to_db():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()

    try:
        # 1. Prepare the table for vectors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("ALTER TABLE allcountryenergy ADD COLUMN IF NOT EXISTS embedding vector(768);")
        conn.commit()

        # 2. Fetch distinct countries that need embeddings
        cur.execute("SELECT DISTINCT country FROM allcountryenergy WHERE embedding IS NULL;")
        countries = [row[0] for row in cur.fetchall()]

        # 3. Generate and Update in batches
        batch_size = 50
        for i in range(0, len(countries), batch_size):
            batch = countries[i:i+batch_size]
            res = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768)
            )
            
            for country, emb_obj in zip(batch, res.embeddings):
                # Update all rows for this country
                cur.execute(
                    "UPDATE allcountryenergy SET embedding = %s WHERE country = %s;",
                    (emb_obj.values, country)
                )
            conn.commit()
            print(f"Updated embeddings for {i + len(batch)} countries...")

    finally:
        cur.close()
        conn.close()

add_embeddings_to_db()