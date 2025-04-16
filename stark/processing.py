# stark/processing.py

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup, Comment
import spacy
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter, defaultdict
from spacy.lang.en.stop_words import STOP_WORDS

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model (ensure you have downloaded it: python -m spacy download en_core_web_sm)
# Using a smaller model for efficiency, consider larger models for higher accuracy.
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please download it: python -m spacy download en_core_web_sm")
    # Fallback or exit strategy might be needed here in a full application
    nlp = None # Indicate model loading failure

# --- Constants ---
DEFAULT_REQUEST_TIMEOUT = 15  # seconds
DEFAULT_USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
MIN_TEXT_LENGTH = 100 # Minimum characters for extracted text to be considered valid
MAX_SUMMARY_SENTENCES = 5 # Max sentences for extractive summary
LDA_N_COMPONENTS = 5 # Number of topics for LDA
LDA_MAX_ITER = 10
LDA_LEARNING_OFFSET = 50.
LDA_RANDOM_STATE = 0

# --- Helper Functions ---

def _is_visible(element):
    """Checks if a BeautifulSoup element is likely visible text content."""
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    if re.match(r"<!--.*-->", str(element.string)): # Redundant with Comment check, but safe
        return False
    if element.name in ['footer', 'nav', 'aside', 'form', 'header']: # Common non-content tags
         # Be careful, sometimes headers contain useful titles
         # Could add more heuristics here (e.g., check class names)
         return False
    # Check if the element itself or its parents have 'display: none' or 'visibility: hidden'
    # Note: This requires parsing CSS, which BeautifulSoup doesn't do.
    # This is a basic heuristic.
    # style = element.get('style')
    # if style and ('display: none' in style or 'visibility: hidden' in style):
    #     return False
    # parent_style = element.find_parent(style=re.compile(r'display:\s*none|visibility:\s*hidden'))
    # if parent_style:
    #     return False

    return True

def _clean_text(text: str) -> str:
    """Basic text cleaning: remove extra whitespace."""
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace chars with single space
    text = text.strip()
    return text

# --- Core Components ---

def scrape_sources(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrapes web pages from a list of URLs to extract main textual content.

    Args:
        urls: A list of URLs to scrape.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - 'url': The original URL.
        - 'text': The extracted text content (or None if failed).
        - 'error': An error message if scraping failed for this URL (or None).
    """
    scraped_data = []
    headers = {'User-Agent': DEFAULT_USER_AGENT}

    for url in urls:
        logger.info(f"Scraping URL: {url}")
        result = {"url": url, "text": None, "error": None}
        try:
            response = requests.get(url, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Check content type - basic check for HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type:
                result['error'] = f"Skipped: Content-Type is not HTML ({content_type})"
                logger.warning(f"Skipping {url}: Content-Type is not HTML ({content_type})")
                scraped_data.append(result)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Attempt to find main content area (common tags/ids/classes)
            # This is heuristic and might need adjustment per site type
            main_content = soup.find('main') or \
                           soup.find('article') or \
                           soup.find(id='main-content') or \
                           soup.find(class_='main-content') or \
                           soup.find(id='content') or \
                           soup.find(class_='content')

            if main_content:
                texts = main_content.find_all(string=True)
            else:
                # Fallback: get all text from the body if main content not found
                logger.warning(f"Main content tag not found for {url}. Extracting from body.")
                texts = soup.body.find_all(string=True) if soup.body else []

            # Filter out invisible text and clean
            visible_texts = filter(_is_visible, texts)
            extracted_text = " ".join(_clean_text(t) for t in visible_texts)

            if len(extracted_text) < MIN_TEXT_LENGTH:
                 # If very little text extracted, try getting all paragraph tags as a fallback
                 logger.warning(f"Extracted text very short ({len(extracted_text)} chars) for {url}. Trying <p> tags.")
                 paragraphs = soup.find_all('p')
                 extracted_text = " ".join(_clean_text(p.get_text()) for p in paragraphs)

            if len(extracted_text) >= MIN_TEXT_LENGTH:
                result['text'] = extracted_text
                logger.info(f"Successfully extracted ~{len(extracted_text)} characters from {url}")
            else:
                 result['error'] = "Failed to extract sufficient text content."
                 logger.warning(f"Failed to extract sufficient text from {url}. Final length: {len(extracted_text)}")

        except requests.exceptions.RequestException as e:
            result['error'] = f"Request failed: {e}"
            logger.error(f"Error scraping {url}: {e}")
        except Exception as e:
            result['error'] = f"Parsing failed: {e}"
            logger.error(f"Error processing {url}: {e}")

        scraped_data.append(result)

    return scraped_data


def process_text(text: str) -> Optional[spacy.tokens.Doc]:
    """
    Processes raw text using spaCy for NLP tasks like tokenization, lemmatization,
    POS tagging, NER, and sentence segmentation.

    Args:
        text: The raw text string to process.

    Returns:
        A spaCy Doc object containing the processed text, or None if processing fails
        (e.g., if spaCy model failed to load or text is empty).
    """
    if not nlp:
        logger.error("spaCy model not loaded. Cannot process text.")
        return None
    if not text or not isinstance(text, str):
        logger.warning("Received empty or invalid text for processing.")
        return None

    try:
        # Increase max_length if needed, but be mindful of memory usage
        # nlp.max_length = len(text) + 10
        doc = nlp(text)
        # Example: Accessing processed data (can be done later as needed)
        # sentences = list(doc.sents)
        # tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        # entities = [(ent.text, ent.label_) for ent in doc.ents]
        # logger.info(f"Processed text: {len(sentences)} sentences, {len(doc)} tokens, {len(entities)} entities.")
        return doc
    except Exception as e:
        logger.error(f"Error processing text with spaCy: {e}")
        return None


def synthesize_information(processed_docs: List[spacy.tokens.Doc]) -> str:
    """
    Creates an extractive summary from a list of processed spaCy documents.
    Uses TF-IDF scores of sentences to rank their importance.

    Args:
        processed_docs: A list of spaCy Doc objects.

    Returns:
        A string containing the synthesized summary, or an empty string if input is invalid.
    """
    if not processed_docs:
        logger.warning("No processed documents provided for synthesis.")
        return ""

    sentences = []
    original_texts = [] # Keep track of original sentence text
    for doc in processed_docs:
        if doc and hasattr(doc, 'sents'):
             for sent in doc.sents:
                 # Basic filtering: ignore very short sentences
                 if len(sent.text.split()) > 5:
                     sentences.append(" ".join([token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct and token.is_alpha]))
                     original_texts.append(sent.text)

    if not sentences:
        logger.warning("No suitable sentences found for synthesis after filtering.")
        return ""

    try:
        # Calculate TF-IDF scores for sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Score sentences by summing TF-IDF scores of their words (simple approach)
        sentence_scores = tfidf_matrix.sum(axis=1)
        sentence_scores = [score[0, 0] for score in sentence_scores] # Flatten the matrix

        # Rank sentences
        ranked_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)

        # Select top N sentences based on original order
        num_summary_sentences = min(MAX_SUMMARY_SENTENCES, len(sentences))
        top_indices = sorted(ranked_indices[:num_summary_sentences]) # Sort by appearance order

        summary = " ".join([original_texts[i] for i in top_indices])
        logger.info(f"Generated summary with {len(top_indices)} sentences.")
        return summary

    except Exception as e:
        logger.error(f"Error during information synthesis: {e}")
        return "Error: Could not generate summary."


def build_knowledge_graph(processed_docs: List[spacy.tokens.Doc]) -> nx.Graph:
    """
    Builds a knowledge graph from processed text data using named entities
    and their co-occurrence within sentences.

    Args:
        processed_docs: A list of spaCy Doc objects.

    Returns:
        A networkx Graph object representing the knowledge graph.
        Nodes are named entities, edges represent co-occurrence in a sentence.
        Edge weights indicate frequency of co-occurrence.
    """
    graph = nx.Graph()
    if not processed_docs:
        logger.warning("No processed documents provided for knowledge graph construction.")
        return graph

    # Use defaultdict to easily increment edge weights
    edge_weights = defaultdict(int)
    entity_types = {} # Store entity types

    logger.info("Building knowledge graph...")
    for doc in processed_docs:
        if not doc or not hasattr(doc, 'ents') or not hasattr(doc, 'sents'):
            continue

        # Extract entities and store their types
        for ent in doc.ents:
             # Use lemma or root for potentially better consolidation, but text is simpler
             entity_text = ent.text.strip()
             if entity_text: # Avoid empty strings
                 entity_types[entity_text] = ent.label_
                 if entity_text not in graph:
                     graph.add_node(entity_text, type=ent.label_) # Add node with type attribute

        # Identify co-occurring entities within the same sentence
        for sent in doc.sents:
            sentence_entities = list({ent.text.strip() for ent in sent.ents if ent.text.strip()}) # Unique entities in sentence

            # Create edges between all pairs of co-occurring entities in the sentence
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    ent1 = sentence_entities[i]
                    ent2 = sentence_entities[j]
                    # Ensure nodes exist (should be added above, but safe check)
                    if ent1 not in graph: graph.add_node(ent1, type=entity_types.get(ent1, 'UNKNOWN'))
                    if ent2 not in graph: graph.add_node(ent2, type=entity_types.get(ent2, 'UNKNOWN'))

                    # Store edge pairs consistently (sorted tuple)
                    edge = tuple(sorted((ent1, ent2)))
                    edge_weights[edge] += 1

    # Add edges to the graph with weights
    for (u, v), weight in edge_weights.items():
        graph.add_edge(u, v, weight=weight)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    logger.info(f"Knowledge graph built with {node_count} nodes and {edge_count} edges.")
    if node_count == 0 and edge_count == 0:
        logger.warning("Knowledge graph is empty. No entities or co-occurrences found.")

    return graph


def identify_gaps(data: List[Dict[str, Any]], n_topics: int = LDA_N_COMPONENTS) -> Dict[str, Any]:
    """
    Identifies potential research gaps using Latent Dirichlet Allocation (LDA)
    topic modeling on the collected text data. Looks for less dominant topics.

    Args:
        data: List of dictionaries, where each dictionary must have a 'text' key
              containing the document text (e.g., from scrape_sources).
        n_topics: The number of topics to identify.

    Returns:
        A dictionary containing:
        - 'topics': A list of tuples, each with (topic_id, top_words).
        - 'topic_distribution': A pandas DataFrame showing document-topic distribution.
        - 'potential_gaps': A list of topic IDs considered less dominant/potential gaps.
        - 'error': An error message if topic modeling failed.
    """
    result = {"topics": [], "topic_distribution": None, "potential_gaps": [], "error": None}
    texts = [item['text'] for item in data if item.get('text')]

    if not texts or len(texts) < n_topics: # Need enough documents for LDA
        result['error'] = "Not enough text documents available for topic modeling."
        logger.warning(f"Topic modeling skipped: Found {len(texts)} documents, need at least {n_topics}.")
        return result

    logger.info(f"Performing LDA topic modeling with {n_topics} topics on {len(texts)} documents...")

    try:
        # Preprocessing for LDA: TF-IDF Vectorization
        # Using stop_words='english' from sklearn, could also use spaCy's list
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=LDA_MAX_ITER,
            learning_method='online',
            learning_offset=LDA_LEARNING_OFFSET,
            random_state=LDA_RANDOM_STATE
        )
        lda.fit(tfidf_matrix)

        # Get Topics and Top Words
        topics = []
        n_top_words = 10
        for topic_idx, topic in enumerate(lda.components_):
            top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices]
            topics.append((topic_idx, ", ".join(top_words)))
        result['topics'] = topics
        logger.info(f"Identified topics: {topics}")

        # Get Document-Topic Distribution
        doc_topic_dist = lda.transform(tfidf_matrix)
        result['topic_distribution'] = pd.DataFrame(doc_topic_dist, columns=[f"Topic_{i}" for i in range(n_topics)])

        # Identify Potential Gaps (e.g., topics with lowest overall probability mass)
        # This is a simple heuristic - 'gaps' could mean many things.
        # Here, we interpret it as less represented topics across the corpus.
        topic_sums = doc_topic_dist.sum(axis=0)
        # Normalize sums (optional, but helps compare)
        # topic_proportions = topic_sums / topic_sums.sum()
        # Identify topics below a certain threshold or simply the least dominant ones
        # Let's identify the topic(s) with the lowest sum(s)
        min_topic_sum = topic_sums.min()
        potential_gap_indices = [i for i, s in enumerate(topic_sums) if s <= min_topic_sum * 1.1] # Allow slight tolerance
        result['potential_gaps'] = potential_gap_indices
        logger.info(f"Potential gap topics (least dominant): {potential_gap_indices}")


    except ValueError as ve:
         # Common issue: empty vocabulary after stop word removal or min_df/max_df filtering
         result['error'] = f"Topic modeling failed: {ve}. Check text preprocessing/vectorizer settings."
         logger.error(f"LDA ValueError: {ve}. This might be due to an empty vocabulary after filtering.")
    except Exception as e:
        result['error'] = f"Topic modeling failed: {e}"
        logger.error(f"Error during topic modeling: {e}")

    return result


# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    # Example URLs (replace with actual test URLs)
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.nature.com/articles/d41586-023-02361-7", # Example Nature article
        "https://nonexistent.url/page", # Example of a failing URL
        "https://example.com" # Example with little content
    ]

    print("--- Scraping ---")
    scraped_content = scrape_sources(test_urls)
    print(f"Scraped {len(scraped_content)} URLs.")
    for item in scraped_content:
        status = "Success" if item['text'] else f"Failed ({item['error']})"
        text_preview = (item['text'][:100] + '...') if item['text'] else 'N/A'
        print(f"  URL: {item['url']} - Status: {status}")
        # print(f"    Preview: {text_preview}") # Uncomment for text preview

    # Filter successful scrapes for further processing
    valid_scraped_data = [item for item in scraped_content if item['text']]

    if valid_scraped_data and nlp: # Proceed only if scraping yielded results and spaCy loaded
        print("\n--- Processing Text ---")
        processed_docs_list = []
        for item in valid_scraped_data:
            doc = process_text(item['text'])
            if doc:
                processed_docs_list.append(doc)
                print(f"  Processed text from: {item['url']} ({len(doc)} tokens)")
            else:
                print(f"  Failed to process text from: {item['url']}")

        if processed_docs_list:
            print("\n--- Synthesizing Information ---")
            summary = synthesize_information(processed_docs_list)
            print("Generated Summary:")
            print(summary if summary else "Could not generate summary.")

            print("\n--- Building Knowledge Graph ---")
            kg = build_knowledge_graph(processed_docs_list)
            print(f"Graph built with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.")
            if kg.number_of_nodes() > 0:
                # Print some example nodes and edges
                print("  Example Nodes:", list(kg.nodes(data=True))[:5])
                print("  Example Edges:", list(kg.edges(data=True))[:5])

            print("\n--- Identifying Gaps (Topic Modeling) ---")
            gap_analysis = identify_gaps(valid_scraped_data)
            if gap_analysis['error']:
                print(f"  Error: {gap_analysis['error']}")
            else:
                print("  Identified Topics:")
                for topic_id, words in gap_analysis['topics']:
                    print(f"    Topic {topic_id}: {words}")
                print(f"  Potential Gap Topic IDs: {gap_analysis['potential_gaps']}")
                # print("  Document-Topic Distribution (Head):")
                # print(gap_analysis['topic_distribution'].head()) # Uncomment to see distribution
        else:
             print("\nNo documents successfully processed, skipping further analysis.")
    elif not nlp:
         print("\nspaCy model not loaded, skipping text processing and analysis.")
    else:
        print("\nNo text successfully scraped, skipping processing and analysis.")