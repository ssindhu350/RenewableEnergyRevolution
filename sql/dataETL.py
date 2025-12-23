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

joined_files = os.path.join("C:\RenewableEnergyAI\RenewableEnergyRevolution\data\countrywise", "*.csv")
joined_list = glob.glob(joined_files)

    # Using a list comprehension
list_of_dfs = [pd.read_csv(filename) for filename in joined_list]
combined_df = pd.concat(list_of_dfs, ignore_index=True)

# data = pd.DataFrame(combined_df)

# print(combined_df.head())
# print(combined_df.info())
# print(combined_df.describe())
# print(data.shape)
# print(len(data))

# --- Data Transformation / ETL ---
col_interested = ['country','iso_code','year', \
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

renew_data = combined_df[col_interested].copy()


# --- 2. Batch Embedding Generation (Gemini) ---
def get_embeddings_gemini(text_list, batch_size=100):
    """
    Fetches embeddings using Google Gemini. 
    Note: Gemini has a limit on the number of strings per request (typically 100).
    """
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        print(f"Embedding batch {i} to {i + len(batch)}...")
        # Using the new SDK client structure
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=VECTOR_DIMENSION
                )
            )
            # The new SDK returns a list of embedding objects
            all_embeddings.extend([e.values for e in result.embeddings])
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            all_embeddings.extend([[0.0] * VECTOR_DIMENSION] * len(batch))
    return all_embeddings
            
# Generate embeddings (using 'Country' as the source text)
# renew_data['embedding'] = get_embeddings_gemini(renew_data['country'].astype(str).tolist())

# Crucial: Format the embedding list into a Postgres-friendly string format: [0.1, 0.2, ...]
# renew_data['embedding'] = renew_data['embedding'].apply(lambda x: str(x).replace(' ', ''))

# Use .loc for the safest way to assign a new column
print("Generating embeddings...")
renew_data.loc[:, 'embedding'] = get_embeddings_gemini(renew_data['country'].astype(str).tolist())

# CRITICAL FIX: Convert vector to string and remove all spaces
# Postgres vector type hates spaces inside the brackets like [0.1, 0.2]
renew_data.loc[:, 'embedding'] = renew_data['embedding'].apply(lambda x: "[" + ",".join(map(str, x)) + "]")


# --- 3. PSQL Bulk Import Function ---
def psql_bulk_copy(df, table_name):
    # Establish raw psycopg2 connection for COPY command
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
    cursor = conn.cursor()
    
    try:
        # Prepare the table
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute(f"Truncate TABLE {table_name};")
        
        # Build dynamic CREATE TABLE string based on DataFrame columns
        # We ensure the 'embedding' column is typed as vector(1536)
        col_types = []
        for col in df.columns:
            if col == 'embedding':
                col_types.append(f'"{col}" vector({VECTOR_DIMENSION})')
            else:
                col_types.append(f'"{col}" TEXT') # Defaulting to text for CSV data
        
        # cursor.execute(f"CREATE TABLE {table_name} ({', '.join(col_types)});")

        # Create an in-memory string buffer (virtual CSV file)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, sep=',', quoting=1)
        buffer.seek(0)
        
        # Execute the COPY command (much faster than to_sql)
        print(f"Starting bulk copy of {len(df)} rows...")
        copy_sql = f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, DELIMITER '|', QUOTE '\"');"
        cursor.copy_expert(copy_sql, buffer)
        
        conn.commit()
        print(f"Bulk import to '{table_name}' completed successfully.")
        
    except Exception as e:
        conn.rollback()
        print(f"Bulk copy failed: {e}")
    finally:
        cursor.close()
        conn.close()

# Execute Bulk Import
psql_bulk_copy(combined_df, 'allcountryenergy')