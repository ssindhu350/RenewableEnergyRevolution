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

def import_data():
    # --- 1. Data Loading ---
    CSV_File = r"C:\RenewableEnergyAI\RenewableEnergyRevolution\data\SolarEnergy\solar_electricity_by_country_consumption.csv"
    raw_df = pd.read_csv(CSV_File)
    
    # 2. Bulk Copy to Postgres
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()
    try:
        cur.execute("DROP TABLE IF EXISTS allcountrysolarconsumption;")
        # Create table with only text/numeric columns
        cols_sql = ", ".join([f'"{c}" TEXT' for c in raw_df.columns])
        cur.execute(f"CREATE TABLE allcountrysolarconsumption ({cols_sql});")

        buffer = io.StringIO()
        raw_df.to_csv(buffer, index=False, header=False, sep='|')
        buffer.seek(0)
        
        cur.copy_expert("COPY allcountrysolarconsumption FROM STDIN WITH (FORMAT CSV, DELIMITER '|');", buffer)
        conn.commit()
        print("Raw data import successful.")
    finally:
        cur.close()
        conn.close()

import_data()

# --- 2. Batch Embedding Generation and loading (Gemini) ---

def add_embeddings_to_db():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()

    try:
        # 1. Prepare the table for vectors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("ALTER TABLE allcountrysolarconsumption ADD COLUMN IF NOT EXISTS embedding vector(768);")
        conn.commit()

        # 2. Fetch distinct countries that need embeddings
        cur.execute("SELECT DISTINCT country FROM allcountrysolarconsumption WHERE embedding IS NULL;")
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
                    "UPDATE allcountrysolarconsumption SET embedding = %s WHERE country = %s;",
                    (emb_obj.values, country)
                )
            conn.commit()
            print(f"Updated embeddings for {i + len(batch)} countries...")

    finally:
        cur.close()
        conn.close()

add_embeddings_to_db()
