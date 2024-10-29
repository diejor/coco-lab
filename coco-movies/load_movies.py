import os
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import math

# ================== Configuration ==================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv()

# Hugging Face Repository Configuration
REPO_ID = "MongoDB/embedded_movies"  # Replace with your actual repo ID if different
FILENAME = "sample_mflix.embedded_movies.json"  # Replace with the actual filename if different

# Output Configuration
OUTPUT_CLEANED_JSON = 'movies_cleaned.json'

# =====================================================

def download_dataset(repo_id: str, filename: str) -> str:
    """
    Downloads a specific file from a Hugging Face repository.

    :param repo_id: The repository ID on Hugging Face.
    :param filename: The name of the file to download.
    :return: The local path to the downloaded file.
    """
    try:
        logger.info(f"Downloading '{filename}' from repository '{repo_id}'...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        logger.info(f"Downloaded '{filename}' to '{file_path}'.")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download '{filename}' from '{repo_id}': {e}")
        exit(1)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a JSON file into a pandas DataFrame.

    :param file_path: The local path to the JSON file.
    :return: A pandas DataFrame containing the dataset.
    """
    try:
        logger.info(f"Loading dataset from '{file_path}'...")
        df = pd.read_json(file_path)
        logger.info(f"Dataset loaded with {len(df)} records.")
        return df
    except ValueError as ve:
        logger.error(f"ValueError while loading JSON: {ve}")
        exit(1)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading the dataset: {e}")
        exit(1)

def replace_nan(obj):
    """
    Recursively replace NaN values in a nested dictionary or list with empty strings.

    :param obj: The object to process (dict, list, or other).
    :return: The processed object with NaNs replaced.
    """
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return ""
        else:
            return obj
    elif pd.isna(obj):
        return ""
    else:
        return obj

def main():
    # Step 1: Download the dataset
    dataset_path = download_dataset(REPO_ID, FILENAME)

    # Step 2: Load the dataset into pandas DataFrame
    df = load_dataset(dataset_path)

    # Step 3: Convert DataFrame to list of records
    records = df.to_dict(orient='records')
    logger.info(f"Total movie records to process: {len(records)}")

    # Step 4: Replace NaN values with empty strings (recursively)
    logger.info("Replacing NaN values with empty strings...")
    cleaned_records = [replace_nan(record) for record in records]

    # Step 5: Save the cleaned records to JSON
    logger.info(f"Saving cleaned records to '{OUTPUT_CLEANED_JSON}'...")
    try:
        with open(OUTPUT_CLEANED_JSON, 'w', encoding='utf-8') as f:
            json.dump(cleaned_records, f, ensure_ascii=False, indent=4)
        logger.info(f"Cleaned data saved to '{OUTPUT_CLEANED_JSON}'.")
    except Exception as e:
        logger.error(f"Failed to save cleaned data to JSON: {e}")

if __name__ == "__main__":
    main()

