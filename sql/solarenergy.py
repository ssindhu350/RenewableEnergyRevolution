import pandas as pd
import os
import psycopg2
import io
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
    CSV_File_cons = r"C:\RenewableEnergyAI\RenewableEnergyRevolution\data\SolarEnergy\solar_electricity_by_country_consumption.csv"
    raw_cons_df = pd.read_csv(CSV_File_cons)
    
    CSV_File_prod = r"C:\RenewableEnergyAI\RenewableEnergyRevolution\data\SolarEnergy\solar_energy_by_country_production.csv"
    raw_prod_df = pd.read_csv(CSV_File_prod)
    
    CSV_File_invst = r"C:\RenewableEnergyAI\RenewableEnergyRevolution\data\SolarEnergy\solar_energy_with_investments_TWh.csv"
    raw_invst_df = pd.read_csv(CSV_File_invst)
    
    # 2. Bulk Copy to Postgres
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()
    try:
        # Drop and recreate allcountrysolarconsumption to ensure it's clean
        cur.execute("DROP TABLE IF EXISTS allcountrysolarconsumption;")
        # Create table with only text/numeric columns
        cols_sql = ", ".join([f'"{c}" TEXT' for c in raw_cons_df.columns])
        cur.execute(f"CREATE TABLE allcountrysolarconsumption ({cols_sql});")

        buffer = io.StringIO()
        raw_cons_df.to_csv(buffer, index=False, header=False, sep='|')
        buffer.seek(0)
        
        cur.copy_expert("COPY allcountrysolarconsumption FROM STDIN WITH (FORMAT CSV, DELIMITER '|');", buffer)
        conn.commit()

        # Drop and recreate allcountrysolarproduction to ensure it's clean
        cur.execute("DROP TABLE IF EXISTS allcountrysolarproduction;")
        # Create table with only text/numeric columns
        cols_sql = ", ".join([f'"{c}" TEXT' for c in raw_prod_df.columns])
        cur.execute(f"CREATE TABLE allcountrysolarproduction ({cols_sql});")

        buffer = io.StringIO()
        raw_prod_df.to_csv(buffer, index=False, header=False, sep='|')
        buffer.seek(0)
        
        cur.copy_expert("COPY allcountrysolarproduction FROM STDIN WITH (FORMAT CSV, DELIMITER '|');", buffer)
        conn.commit()

        # Drop and recreate allcountrysolarinvestment to ensure it's clean
        cur.execute("DROP TABLE IF EXISTS allcountrysolarinvestment;")
        # Create table with only text/numeric columns
        cols_sql = ", ".join([f'"{c}" TEXT' for c in raw_invst_df.columns])
        cur.execute(f"CREATE TABLE allcountrysolarinvestment ({cols_sql});")

        buffer = io.StringIO()
        raw_invst_df.to_csv(buffer, index=False, header=False, sep='|')
        buffer.seek(0)
        
        cur.copy_expert("COPY allcountrysolarinvestment FROM STDIN WITH (FORMAT CSV, DELIMITER '|');", buffer)
        conn.commit()
        print("Raw data Solar Energy import successful.")
    finally:
        cur.close()
        conn.close()

import_data()

# --- 2. Batch Embedding Generation and loading (Gemini) ---

def add_embeddings_to_db():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cur = conn.cursor()

    try:
        # 1. Prepare the allcountrysolarconsumption for vectors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("ALTER TABLE allcountrysolarconsumption ADD COLUMN IF NOT EXISTS embedding vector(768);")
        conn.commit()

        # 2. Fetch distinct countries that need embeddings
        cur.execute("SELECT DISTINCT country FROM allcountrysolarconsumption WHERE embedding IS NULL;")
        countries = [row[0] for row in cur.fetchall()]

        # 3. Generate and Update in batches for allcountrysolarconsumption
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
            
        # 1. Prepare the allcountrysolarproduction for vectors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("ALTER TABLE allcountrysolarproduction ADD COLUMN IF NOT EXISTS embedding vector(768);")
        conn.commit()

        # 2. Fetch distinct countries that need embeddings
        cur.execute("SELECT DISTINCT Country FROM allcountrysolarproduction WHERE embedding IS NULL;")
        countries = [row[0] for row in cur.fetchall()]

        # 3. Generate and Update in batches for allcountrysolarproduction
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
                    "UPDATE allcountrysolarproduction SET embedding = %s WHERE Country = %s;",
                    (emb_obj.values, country)
                )
            conn.commit()
            print(f"Updated embeddings for {i + len(batch)} countries...")
        
        # 1. Prepare the allcountrysolarinvestments for vectors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("ALTER TABLE allcountrysolarinvestment ADD COLUMN IF NOT EXISTS embedding vector(768);")
        conn.commit()

        # 2. Fetch distinct countries that need embeddings
        cur.execute("SELECT DISTINCT country FROM allcountrysolarinvestment WHERE embedding IS NULL;")
        countries = [row[0] for row in cur.fetchall()]

        # 3. Generate and Update in batches for allcountrysolarinvestments
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
                    "UPDATE allcountrysolarinvestment SET embedding = %s WHERE country = %s;",
                    (emb_obj.values, country)
                )
            conn.commit()
            print(f"Updated embeddings for {i + len(batch)} countries...")
        

    finally:
        cur.close()
        conn.close()

add_embeddings_to_db()
