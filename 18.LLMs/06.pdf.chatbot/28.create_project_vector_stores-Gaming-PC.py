# This script requires:
# pip install mysql-connector-python sentence-transformers faiss-cpu numpy

import os
import mysql.connector
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# === DATABASE CONFIG ===
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}
TABLE_NAME = 'project_summaries'

# === OUTPUT & MODEL CONFIG ===
OUTPUT_DIR = r'D:\OEssam\for.3.project.chatbot'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# === DATABASE FUNCTIONS ===
def fetch_data_from_db():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True) # Fetch as dictionaries
        query = f"SELECT project_name, pdf_filename, final_summary, page_summaries FROM {TABLE_NAME} WHERE final_summary IS NOT NULL AND final_summary != ''"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        return data
    except mysql.connector.Error as err:
        print(f"Error connecting to or fetching from database: {err}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()

# === VECTOR STORE FUNCTIONS ===
def create_and_save_project_vector_store(project_name, project_data, embedder, output_dir):
    print(f"Processing project: {project_name} ({len(project_data)} documents)")
    if not project_data:
        print(f"  No data to process for {project_name}.")
        return

    texts_to_embed = [
        f"Document Name: {doc['pdf_filename']}\n\nFinal Summary:\n{doc['final_summary']}"
        for doc in project_data
    ]
    
    print(f"  Generating embeddings for {len(texts_to_embed)} final summaries...")
    embeddings = embedder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
    
    if embeddings.size == 0:
        print(f"  No embeddings generated for {project_name}. Skipping vector store creation.")
        return

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    index.add(np.array(embeddings, dtype='float32'))
    
    # Sanitize project_name for filename
    safe_project_name = project_name.lower().replace(' ', '_').replace('-', '_').replace('.', '')
    
    index_path = os.path.join(output_dir, f"{safe_project_name}.faiss")
    metadata_path = os.path.join(output_dir, f"{safe_project_name}_metadata.pkl")
    
    print(f"  Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"  Saving metadata to: {metadata_path}")
    # Metadata will be the list of original project_data dictionaries for this project
    with open(metadata_path, 'wb') as f_meta:
        pickle.dump(project_data, f_meta)
    print(f"  Successfully created vector store for {project_name}.")

# === MAIN EXECUTION ===
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Fetching data from database...")
    all_data = fetch_data_from_db()
    if not all_data:
        print("No data fetched from database. Exiting.")
        return

    # Group data by project_name
    projects = {}
    for row in all_data:
        project_name = row['project_name']
        if project_name not in projects:
            projects[project_name] = []
        projects[project_name].append(row)
    
    print(f"Found data for {len(projects)} projects: {', '.join(projects.keys())}")

    for project_name, data_list in projects.items():
        create_and_save_project_vector_store(project_name, data_list, embedder, OUTPUT_DIR)
    
    print("\nPreprocessing complete. Vector stores created.")

if __name__ == "__main__":
    main()
