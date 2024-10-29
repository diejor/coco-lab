import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import json
import logging
import math

# ================== Configuration ==================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Input Configuration
INPUT_AVAILABLE_JSON = 'movies_available.json'

# Visualization Configuration
DIMENSIONALITY_REDUCTION = 'UMAP'  # Options: 'PCA', 'tSNE', 'UMAP'
OUTPUT_STATIC_PLOT = 'movies_static_plot.png'
OUTPUT_INTERACTIVE_HTML = 'movies_interactive_plot.html'
OUTPUT_MISSING_EMBEDDINGS_LOG = 'visualization_missing_plot_embeddings.log'  # Log file for missing embeddings

# =====================================================

def load_available_movies(file_path: str) -> pd.DataFrame:
    """
    Loads the available movies dataset from a JSON file into a pandas DataFrame.

    :param file_path: The local path to the JSON file.
    :return: A pandas DataFrame containing the available movies.
    """
    try:
        logger.info(f"Loading available movies from '{file_path}'...")
        df = pd.read_json(file_path)
        logger.info(f"Available movies loaded with {len(df)} records.")
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

def parse_plot_embedding(embedding):
    """
    Parses the plot_embedding list, ensuring it's a list of numerical values.

    :param embedding: The plot_embedding data, expected to be a list.
    :return: A list of numerical values representing the embedding.
    """
    if isinstance(embedding, list):
        # Ensure all elements are floats; replace non-floats with 0.0 or handle appropriately
        return [float(x) if isinstance(x, (int, float)) else 0.0 for x in embedding]
    else:
        return []

def perform_dimensionality_reduction(embeddings: np.ndarray, method: str = 'UMAP') -> pd.DataFrame:
    """
    Performs dimensionality reduction on the embeddings.

    :param embeddings: Numpy array of plot embeddings.
    :param method: Dimensionality reduction technique ('PCA', 'tSNE', 'UMAP').
    :return: DataFrame with reduced dimensions.
    """
    if method == 'PCA':
        logger.info("Performing PCA for dimensionality reduction...")
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embeddings)
        logger.info("PCA completed.")
        return pd.DataFrame(reduced, columns=['PCA_1', 'PCA_2'])
    elif method == 'tSNE':
        logger.info("Performing t-SNE for dimensionality reduction...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
        reduced = tsne.fit_transform(embeddings)
        logger.info("t-SNE completed.")
        return pd.DataFrame(reduced, columns=['tSNE_1', 'tSNE_2'])
    elif method == 'UMAP':
        logger.info("Performing UMAP for dimensionality reduction...")
        umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        reduced = umap_reducer.fit_transform(embeddings)
        logger.info("UMAP completed.")
        return pd.DataFrame(reduced, columns=['UMAP_1', 'UMAP_2'])
    else:
        logger.error("Invalid dimensionality reduction method. Choose from 'PCA', 'tSNE', 'UMAP'.")
        exit(1)

def main():
    # Step 1: Load the available movies dataset
    df = load_available_movies(INPUT_AVAILABLE_JSON)

    # Step 2: Parse 'plot_embedding' ensuring it's a list of floats
    logger.info("Parsing 'plot_embedding' to ensure correct format...")
    df['plot_embedding'] = df['plot_embedding'].apply(parse_plot_embedding)

    # Debugging Step: Print first few 'plot_embedding' entries
    logger.info("Sample 'plot_embedding' entries:")
    print(json.dumps(df['plot_embedding'].head(5).tolist(), indent=4))

    # Step 3: Extract embeddings
    embeddings = df['plot_embedding'].tolist()
    embeddings_array = np.array(embeddings)
    logger.info(f"Embeddings array shape: {embeddings_array.shape}")

    # Step 4: Handle missing or empty embeddings
    missing_embeddings = df['plot_embedding'].isna().sum()
    empty_embeddings = sum(len(embed) == 0 for embed in embeddings)
    invalid_embeddings = missing_embeddings + empty_embeddings

    if invalid_embeddings > 0:
        logger.warning(f"There are {invalid_embeddings} records with missing or empty 'plot_embedding'. They will be excluded from visualization.")
        # Identify and log the titles of movies with missing embeddings
        missing_titles = df[df['plot_embedding'].apply(lambda x: not x)].get('title', 'Unknown Title').tolist()
        try:
            with open(OUTPUT_MISSING_EMBEDDINGS_LOG, 'w', encoding='utf-8') as log_file:
                for title in missing_titles:
                    log_file.write(f"{title}\n")
            logger.info(f"Missing 'plot_embedding' records logged to '{OUTPUT_MISSING_EMBEDDINGS_LOG}'.")
        except Exception as e:
            logger.error(f"Failed to log missing 'plot_embedding' records: {e}")

        # Exclude these records
        df = df[df['plot_embedding'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        embeddings = df['plot_embedding'].tolist()
        embeddings_array = np.array(embeddings)
        logger.info(f"Embeddings array shape after excluding missing and empty: {embeddings_array.shape}")
    else:
        logger.info("All records have valid 'plot_embedding'.")

    if embeddings_array.size == 0:
        logger.error("No valid 'plot_embedding' data available for visualization. Exiting.")
        exit(1)

    # Step 5: Perform dimensionality reduction
    reduced_df = perform_dimensionality_reduction(embeddings_array, method=DIMENSIONALITY_REDUCTION)

    # Step 6: Combine reduced dimensions with the original DataFrame
    df_combined = pd.concat([df.reset_index(drop=True), reduced_df], axis=1)

    # Step 7: Visualization using Matplotlib and Seaborn
    plt.figure(figsize=(12, 10))
    if 'cluster' in df_combined.columns:
        sns.scatterplot(
            x=reduced_df.columns[0],
            y=reduced_df.columns[1],
            hue='cluster',
            palette='tab10',
            data=df_combined,
            alpha=0.6
        )
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(
            x=reduced_df.columns[0],
            y=reduced_df.columns[1],
            data=df_combined,
            alpha=0.6
        )
    plt.title(f'Movies Visualization using {DIMENSIONALITY_REDUCTION}')
    plt.xlabel(reduced_df.columns[0])
    plt.ylabel(reduced_df.columns[1])
    plt.tight_layout()
    plt.savefig(OUTPUT_STATIC_PLOT)
    logger.info(f"Static plot saved to '{OUTPUT_STATIC_PLOT}'.")
    plt.show()

    # Step 8: Interactive Visualization using Plotly
    fig = px.scatter(
        df_combined,
        x=reduced_df.columns[0],
        y=reduced_df.columns[1],
        color='cluster' if 'cluster' in df_combined.columns else None,
        hover_data=['title', 'genres', 'directors', 'cast'],
        title=f'Movies Visualization using {DIMENSIONALITY_REDUCTION}',
        opacity=0.7,
        width=1200,
        height=800
    )

    fig.update_layout(legend_title_text='Cluster' if 'cluster' in df_combined.columns else '')
    fig.show()

    # Save interactive plot as HTML
    try:
        fig.write_html(OUTPUT_INTERACTIVE_HTML)
        logger.info(f"Interactive plot saved to '{OUTPUT_INTERACTIVE_HTML}'.")
    except Exception as e:
        logger.error(f"Failed to save interactive plot: {e}")

if __name__ == "__main__":
    main()

