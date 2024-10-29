import os
import json
import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

# ================== Configuration ==================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv()

# Primo API Configuration
PRIMO_API_KEY = os.getenv('PRIMO_API_KEY')
if not PRIMO_API_KEY:
    logger.error("PRIMO_API_KEY environment variable not set.")
    exit(1)

# Input and Output Configuration
INPUT_CLEANED_JSON = 'movies_cleaned.json'
OUTPUT_AVAILABLE_JSON = 'movies_available.json'
OUTPUT_AVAILABLE_EXCEL = 'movies_available.xlsx'
OUTPUT_MISSING_EMBEDDINGS_LOG = 'missing_plot_embeddings.log'  # Log file for missing embeddings

# Threading Configuration
MAX_THREADS = 10  # Adjust based on your system and API rate limits

# =====================================================

def load_cleaned_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the cleaned dataset from a JSON file into a pandas DataFrame.

    :param file_path: The local path to the cleaned JSON file.
    :return: A pandas DataFrame containing the cleaned dataset.
    """
    try:
        logger.info(f"Loading cleaned dataset from '{file_path}'...")
        df = pd.read_json(file_path)
        logger.info(f"Cleaned dataset loaded with {len(df)} records.")
        return df
    except ValueError as ve:
        logger.error(f"ValueError while loading JSON: {ve}")
        exit(1)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading the cleaned dataset: {e}")
        exit(1)

def fetch_movie_availability(query: str, api_key: str, offset: int = 0, limit: int = 10) -> dict:
    """
    Fetches search results from the Primo API for a given movie title.

    :param query: The movie title to search for.
    :param api_key: Your Primo API key.
    :param offset: Pagination offset.
    :param limit: Number of results to fetch per request.
    :return: JSON response containing search results or None if an error occurs.
    """
    base_url = "https://utdallas.primo.exlibrisgroup.com/primaws/rest/pub/pnxs"

    params = {
        'acTriggered': 'false',
        'blendFacetsSeparately': 'false',
        'citationTrailFilterByAvailability': 'true',
        'disableCache': 'false',
        'getMore': '0',
        'inst': '01UT_DALLAS',
        'isCDSearch': 'false',
        'lang': 'en',
        'limit': str(limit),
        'newspapersActive': 'true',
        'newspapersSearch': 'false',
        'offset': str(offset),
        'otbRanking': 'false',
        'pcAvailability': 'false',
        'q': f'any,contains,{query}',
        'qExclude': '',
        'qInclude': '',
        'rapido': 'false',
        'refEntryActive': 'false',
        'rtaLinks': 'true',
        'scope': 'MyInst_and_CI',
        'searchInFulltextUserSelection': 'false',
        'skipDelivery': 'Y',
        'sort': 'rank',
        'tab': 'Everything',
        'vid': '01UT_DALLAS:UTDALMA'
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error for '{query}': {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error for '{query}': {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error for '{query}': {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception for '{query}': {req_err}")
    except ValueError:
        logger.error(f"Invalid JSON response for '{query}'.")
    return None

def check_dvd_availability(doc: dict) -> bool:
    """
    Checks if the 'lds10' attribute indicates DVD/Blu-ray availability.

    :param doc: A single document entry from the JSON response.
    :return: Boolean indicating availability.
    """
    lds10 = doc.get('pnx', {}).get('display', {}).get('lds10', [])
    if not lds10:
        return False

    for entry in lds10:
        if "DVD/Blu-ray player can be checked-out!" in entry:
            return True
    return False

def process_movie(record: dict, api_key: str) -> dict:
    """
    Processes a single movie record to determine DVD/Blu-ray availability.

    :param record: The dataset record (a dictionary) containing movie information.
    :param api_key: Your Primo API key.
    :return: Augmented record with library link if available, or None.
    """
    query = record.get('title')
    if not query:
        return None

    data = fetch_movie_availability(query, api_key)
    if not data or 'docs' not in data or not data['docs']:
        # No data found or error occurred, skip this movie
        return None

    # Assuming the first document is the most relevant
    first_doc = data['docs'][0]

    # Check availability
    if not check_dvd_availability(first_doc):
        # Movie not available for DVD/Blu-ray
        return None

    # Construct the link
    record_id = first_doc.get('pnx', {}).get('control', {}).get('recordid', ['No Record ID'])[0]
    link = f"https://utdallas.primo.exlibrisgroup.com/discovery/fulldisplay?docid={record_id}&context=L&vid=01UT_DALLAS:UTDALMA&lang=en"

    # Prepend the link to the record and retain 'plot_embedding'
    augmented_record = {'library_link': link}
    for key, value in record.items():
        # Retain all fields, including 'plot_embedding'
        augmented_record[key] = value

    return augmented_record

def main():
    # Step 1: Load the cleaned dataset
    df = load_cleaned_dataset(INPUT_CLEANED_JSON)

    # Step 2: Convert DataFrame to list of records
    records = df.to_dict(orient='records')
    logger.info(f"Total movie records to process: {len(records)}")

    results = []
    missing_plot_embeddings = []

    # Step 3: Use ThreadPoolExecutor for concurrent API requests with progress bar
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_movie, record, PRIMO_API_KEY): record for record in records}

        # Initialize tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing movies"):
            record = futures[future]
            try:
                result = future.result()
                if result:
                    # Check if 'plot_embedding' exists and is non-empty
                    plot_embedding = result.get('plot_embedding', [])
                    if plot_embedding:
                        results.append(result)
                    else:
                        missing_plot_embeddings.append(result.get('title', 'Unknown Title'))
                else:
                    # If result is None, check if 'plot_embedding' was missing
                    if not record.get('plot_embedding'):
                        missing_plot_embeddings.append(record.get('title', 'Unknown Title'))
            except Exception as exc:
                logger.error(f"Exception processing record '{record.get('title', 'Unknown')}': {exc}")

    # Step 4: Process the results
    if results:
        # Since NaNs are already handled, we can skip replace_nan
        records_list = results  # Directly assign results

        # Optionally, log missing plot embeddings
        if missing_plot_embeddings:
            logger.warning(f"{len(missing_plot_embeddings)} records have missing or empty 'plot_embedding'.")
            try:
                with open(OUTPUT_MISSING_EMBEDDINGS_LOG, 'w', encoding='utf-8') as log_file:
                    for title in missing_plot_embeddings:
                        log_file.write(f"{title}\n")
                logger.info(f"Missing 'plot_embedding' records logged to '{OUTPUT_MISSING_EMBEDDINGS_LOG}'.")
            except Exception as e:
                logger.error(f"Failed to log missing 'plot_embedding' records: {e}")

        logger.info("All records have valid 'plot_embedding'.")

        logger.info(f"Total movies available: {len(records_list)}")

        # Step 5: Display the first few results
        logger.info("\n--- Available Movies Report ---\n")
        print(json.dumps(records_list[:5], ensure_ascii=False, indent=4))

        # Step 6: Save the filtered results to a formatted JSON file without escaped slashes
        try:
            logger.info(f"Saving available movies to '{OUTPUT_AVAILABLE_JSON}'...")
            with open(OUTPUT_AVAILABLE_JSON, 'w', encoding='utf-8') as f:
                json.dump(records_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Available movies saved to '{OUTPUT_AVAILABLE_JSON}'.")
        except Exception as e:
            logger.error(f"Failed to save available movies to JSON: {e}")

        # Step 7: Optionally, save the filtered results to an Excel file
        try:
            logger.info(f"Saving available movies to '{OUTPUT_AVAILABLE_EXCEL}'...")
            df_available = pd.DataFrame(records_list)
            df_available.to_excel(OUTPUT_AVAILABLE_EXCEL, index=False, engine='openpyxl')
            logger.info(f"Available movies saved to '{OUTPUT_AVAILABLE_EXCEL}'.")
        except Exception as e:
            logger.error(f"Failed to save available movies to Excel: {e}")
    else:
        logger.info("No movies available for DVD/Blu-ray found.")

if __name__ == "__main__":
    main()

