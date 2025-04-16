# stark/storage.py

"""
Manages session persistence, intelligent caching, and citation tracking for STARK.
"""

import pickle
import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from filelock import FileLock # For safe concurrent file access, especially citations

# --- Configuration ---

# Define base directory for storage (e.g., ~/.stark)
# Use environment variable if set, otherwise default to user's home directory
DEFAULT_STORAGE_DIR = Path.home() / ".stark"
STORAGE_DIR = Path(os.environ.get("STARK_STORAGE_DIR", DEFAULT_STORAGE_DIR))

SESSION_FILE = STORAGE_DIR / "stark_session.pkl"
CACHE_DIR = STORAGE_DIR / "cache"
CITATION_FILE = STORAGE_DIR / "citations.json"
CITATION_LOCK_FILE = STORAGE_DIR / "citations.lock"

# Default cache expiry time (e.g., 7 days)
DEFAULT_CACHE_EXPIRY = timedelta(days=7)

# --- Setup Logging ---
# Configure logging (can be refined in a central logging setup if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ensure Directories Exist ---
try:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage directory initialized at: {STORAGE_DIR}")
except OSError as e:
    logger.error(f"Failed to create storage directories: {e}", exc_info=True)
    # Depending on the application's needs, you might want to raise this error
    # or handle it gracefully (e.g., disable persistence/caching).
    # For now, we log the error and proceed; functions will likely fail later.

# --- Session Management ---

def save_session(state: Any) -> bool:
    """
    Saves the current application state to a file using pickle.

    Args:
        state: The application state object (can be any pickleable object,
               typically a dictionary or a custom class instance).

    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Session state successfully saved to {SESSION_FILE}")
        return True
    except (pickle.PicklingError, OSError, IOError) as e:
        logger.error(f"Error saving session state to {SESSION_FILE}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during session save: {e}", exc_info=True)
        return False

def load_session() -> Optional[Any]:
    """
    Loads the application state from the session file.

    Returns:
        The loaded application state object, or None if the file doesn't exist
        or an error occurs during loading.
    """
    if not SESSION_FILE.exists():
        logger.info("No existing session file found.")
        return None

    try:
        with open(SESSION_FILE, "rb") as f:
            state = pickle.load(f)
        logger.info(f"Session state successfully loaded from {SESSION_FILE}")
        return state
    except (pickle.UnpicklingError, EOFError, OSError, IOError) as e:
        logger.error(f"Error loading session state from {SESSION_FILE}: {e}", exc_info=True)
        # Optionally, attempt to delete the corrupted file
        # try:
        #     SESSION_FILE.unlink()
        #     logger.warning(f"Corrupted session file {SESSION_FILE} removed.")
        # except OSError as del_e:
        #     logger.error(f"Failed to remove corrupted session file {SESSION_FILE}: {del_e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during session load: {e}", exc_info=True)
        return None

# --- Caching ---

def _generate_cache_key(key_string: str) -> str:
    """Generates a safe filename hash for a given cache key string."""
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def cache_data(key: str, data: Any, expiry: Optional[timedelta] = DEFAULT_CACHE_EXPIRY) -> bool:
    """
    Caches data associated with a specific key.

    Args:
        key: A unique string identifier for the data (e.g., URL, query).
        data: The Python object to cache (must be pickleable).
        expiry: A timedelta object indicating how long the cache is valid.
                If None, the cache does not expire based on time.

    Returns:
        True if caching was successful, False otherwise.
    """
    if not CACHE_DIR.exists():
        logger.warning(f"Cache directory {CACHE_DIR} does not exist. Caching skipped.")
        return False

    cache_filename = _generate_cache_key(key)
    cache_filepath = CACHE_DIR / cache_filename

    timestamp = datetime.now()
    cache_entry = {"timestamp": timestamp, "expiry": expiry, "data": data}

    try:
        with open(cache_filepath, "wb") as f:
            pickle.dump(cache_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Data cached successfully for key '{key}' at {cache_filepath}")
        return True
    except (pickle.PicklingError, OSError, IOError) as e:
        logger.error(f"Error caching data for key '{key}' to {cache_filepath}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during data caching for key '{key}': {e}", exc_info=True)
        return False

def get_cached_data(key: str) -> Optional[Any]:
    """
    Retrieves cached data associated with a specific key, checking for expiry.

    Args:
        key: The unique string identifier used when caching the data.

    Returns:
        The cached data object if found and not expired, otherwise None.
    """
    if not CACHE_DIR.exists():
        logger.warning(f"Cache directory {CACHE_DIR} does not exist. Cannot retrieve cache.")
        return None

    cache_filename = _generate_cache_key(key)
    cache_filepath = CACHE_DIR / cache_filename

    if not cache_filepath.exists():
        logger.debug(f"Cache miss for key '{key}' (file not found: {cache_filepath})")
        return None

    try:
        with open(cache_filepath, "rb") as f:
            cache_entry = pickle.load(f)

        if not isinstance(cache_entry, dict) or "timestamp" not in cache_entry or "data" not in cache_entry:
             logger.warning(f"Invalid cache entry format for key '{key}' in file {cache_filepath}. Discarding.")
             # Optionally remove the invalid file
             # cache_filepath.unlink(missing_ok=True)
             return None

        timestamp = cache_entry["timestamp"]
        expiry = cache_entry.get("expiry", DEFAULT_CACHE_EXPIRY) # Use default if expiry not stored

        if expiry is not None:
            if not isinstance(timestamp, datetime) or not isinstance(expiry, timedelta):
                 logger.warning(f"Invalid timestamp or expiry type in cache for key '{key}'. Discarding.")
                 return None

            if datetime.now() > timestamp + expiry:
                logger.info(f"Cache expired for key '{key}'. Stale data ignored.")
                # Optionally remove the expired file
                # cache_filepath.unlink(missing_ok=True)
                return None

        logger.debug(f"Cache hit for key '{key}'.")
        return cache_entry["data"]

    except (pickle.UnpicklingError, EOFError, OSError, IOError) as e:
        logger.error(f"Error reading cache file {cache_filepath} for key '{key}': {e}", exc_info=True)
        # Optionally remove the corrupted file
        # cache_filepath.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred retrieving cache for key '{key}': {e}", exc_info=True)
        return None

def clear_cache(expired_only: bool = False) -> Tuple[int, int]:
    """
    Clears the cache directory.

    Args:
        expired_only: If True, only removes expired cache files. Otherwise, removes all.

    Returns:
        A tuple containing (number_of_files_removed, number_of_files_failed_to_remove).
    """
    removed_count = 0
    failed_count = 0

    if not CACHE_DIR.exists():
        logger.info("Cache directory does not exist. Nothing to clear.")
        return 0, 0

    logger.info(f"Starting cache clearing process (expired_only={expired_only})...")
    for item in CACHE_DIR.iterdir():
        if item.is_file():
            remove_file = False
            if not expired_only:
                remove_file = True
            else:
                # Need to load the file to check expiry
                try:
                    with open(item, "rb") as f:
                        cache_entry = pickle.load(f)
                    if isinstance(cache_entry, dict) and "timestamp" in cache_entry:
                        timestamp = cache_entry["timestamp"]
                        expiry = cache_entry.get("expiry", DEFAULT_CACHE_EXPIRY)
                        if expiry is not None and isinstance(timestamp, datetime) and isinstance(expiry, timedelta):
                            if datetime.now() > timestamp + expiry:
                                remove_file = True
                                logger.debug(f"Identified expired cache file: {item.name}")
                        # else: handle cases without expiry or invalid types if needed
                    # else: handle invalid cache entry format if needed
                except Exception as e:
                    logger.warning(f"Could not read cache file {item.name} to check expiry: {e}. Skipping.")
                    # Decide if corrupted files should be removed anyway
                    # remove_file = True # Uncomment to remove corrupted files during expired check

            if remove_file:
                try:
                    item.unlink()
                    logger.debug(f"Removed cache file: {item.name}")
                    removed_count += 1
                except OSError as e:
                    logger.error(f"Failed to remove cache file {item.name}: {e}")
                    failed_count += 1

    logger.info(f"Cache clearing finished. Removed: {removed_count}, Failed: {failed_count}")
    return removed_count, failed_count


# --- Citation Tracking ---

def _load_citations() -> List[Dict[str, str]]:
    """Loads citations from the JSON file."""
    if not CITATION_FILE.exists():
        return []
    try:
        # Use FileLock for reading as well, ensuring consistency if writes are happening
        lock = FileLock(CITATION_LOCK_FILE, timeout=5)
        with lock:
            with open(CITATION_FILE, "r", encoding="utf-8") as f:
                citations = json.load(f)
            if not isinstance(citations, list):
                logger.warning(f"Citation file {CITATION_FILE} does not contain a list. Initializing empty list.")
                return []
            return citations
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from citation file {CITATION_FILE}. Returning empty list.", exc_info=True)
        # Consider backing up the corrupted file
        return []
    except (OSError, IOError) as e:
        logger.error(f"Error reading citation file {CITATION_FILE}: {e}", exc_info=True)
        return []
    except Exception as e:
         logger.error(f"An unexpected error occurred loading citations: {e}", exc_info=True)
         return []


def _save_citations(citations: List[Dict[str, str]]) -> bool:
    """Saves the list of citations to the JSON file."""
    try:
        # Use FileLock to prevent race conditions during writes
        lock = FileLock(CITATION_LOCK_FILE, timeout=10) # Increased timeout for write
        with lock:
            with open(CITATION_FILE, "w", encoding="utf-8") as f:
                json.dump(citations, f, indent=4, ensure_ascii=False)
        return True
    except (OSError, IOError) as e:
        logger.error(f"Error writing citation file {CITATION_FILE}: {e}", exc_info=True)
        return False
    except Exception as e:
         logger.error(f"An unexpected error occurred saving citations: {e}", exc_info=True)
         return False

def add_citation(source: str, content_snippet: str) -> bool:
    """
    Adds a citation record (source URL/identifier and content snippet).

    Args:
        source: The identifier of the source (e.g., URL, document title, DOI).
        content_snippet: A relevant snippet of text from the source.

    Returns:
        True if the citation was added successfully, False otherwise.
    """
    if not source or not content_snippet:
        logger.warning("Attempted to add citation with empty source or content.")
        return False

    citations = _load_citations()

    new_citation = {
        "source": source,
        "content_snippet": content_snippet,
        "timestamp": datetime.now().isoformat()
    }

    # Optional: Check for duplicates before adding
    # is_duplicate = any(c.get("source") == source and c.get("content_snippet") == content_snippet for c in citations)
    # if is_duplicate:
    #     logger.info(f"Duplicate citation detected for source '{source}'. Skipping.")
    #     return True # Or False depending on desired behavior

    citations.append(new_citation)

    if _save_citations(citations):
        logger.info(f"Citation added for source: {source}")
        return True
    else:
        logger.error(f"Failed to save updated citations after adding source: {source}")
        return False

def get_citations() -> List[Dict[str, str]]:
    """
    Retrieves all recorded citations.

    Returns:
        A list of citation dictionaries, each containing 'source',
        'content_snippet', and 'timestamp'. Returns an empty list on error.
    """
    return _load_citations()

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    print(f"STARK Storage Module - Test Run")
    print(f"Storage Directory: {STORAGE_DIR}")
    print("-" * 30)

    # --- Test Session ---
    print("Testing Session Management...")
    initial_state = {"user": "test_user", "theme": "dark", "last_query": "example query"}
    print(f"Saving initial state: {initial_state}")
    save_success = save_session(initial_state)
    print(f"Save successful: {save_success}")

    loaded_state = load_session()
    print(f"Loaded state: {loaded_state}")
    if loaded_state == initial_state:
        print("Session Load/Save Test: PASSED")
    else:
        print("Session Load/Save Test: FAILED")
    print("-" * 30)

    # --- Test Caching ---
    print("Testing Caching...")
    cache_key1 = "https://example.com/page1"
    cache_data1 = {"title": "Example Page 1", "content": "Some text content."}
    print(f"Caching data for key: {cache_key1}")
    cache_success1 = cache_data(cache_key1, cache_data1, expiry=timedelta(seconds=10)) # Short expiry for testing
    print(f"Cache successful: {cache_success1}")

    retrieved_data1 = get_cached_data(cache_key1)
    print(f"Retrieved data (before expiry): {retrieved_data1}")
    if retrieved_data1 == cache_data1:
        print("Cache Retrieve Test (Before Expiry): PASSED")
    else:
        print("Cache Retrieve Test (Before Expiry): FAILED")

    print("Waiting for cache to expire (12 seconds)...")
    import time
    time.sleep(12)

    retrieved_data1_expired = get_cached_data(cache_key1)
    print(f"Retrieved data (after expiry): {retrieved_data1_expired}")
    if retrieved_data1_expired is None:
        print("Cache Expiry Test: PASSED")
    else:
        print("Cache Expiry Test: FAILED")

    # Test non-expiring cache
    cache_key2 = "permanent_data"
    cache_data2 = [1, 2, 3]
    cache_data(cache_key2, cache_data2, expiry=None)
    retrieved_data2 = get_cached_data(cache_key2)
    print(f"Retrieved non-expiring data: {retrieved_data2}")
    if retrieved_data2 == cache_data2:
        print("Cache Non-Expiring Test: PASSED")
    else:
        print("Cache Non-Expiring Test: FAILED")

    print("Clearing only expired cache...")
    removed, failed = clear_cache(expired_only=True)
    print(f"Expired cache clear result: Removed={removed}, Failed={failed}")
    retrieved_data1_after_clear = get_cached_data(cache_key1)
    retrieved_data2_after_clear = get_cached_data(cache_key2)
    if retrieved_data1_after_clear is None and retrieved_data2_after_clear == cache_data2:
         print("Clear Expired Cache Test: PASSED")
    else:
         print("Clear Expired Cache Test: FAILED")


    print("Clearing all cache...")
    removed_all, failed_all = clear_cache(expired_only=False)
    print(f"Full cache clear result: Removed={removed_all}, Failed={failed_all}")
    if get_cached_data(cache_key1) is None and get_cached_data(cache_key2) is None:
        print("Clear All Cache Test: PASSED")
    else:
        print("Clear All Cache Test: FAILED")

    print("-" * 30)

    # --- Test Citations ---
    print("Testing Citation Tracking...")
    # Clear existing citations for clean test
    _save_citations([])

    cite_success1 = add_citation("https://example.com/source1", "This is the first important snippet.")
    cite_success2 = add_citation("https://anothersite.org/doc.pdf", "Another key finding mentioned here.")
    cite_success3 = add_citation("https://example.com/source1", "A second snippet from the first source.") # Add another from same source

    print(f"Citation add success: {cite_success1}, {cite_success2}, {cite_success3}")

    all_citations = get_citations()
    print(f"Retrieved citations ({len(all_citations)}):")
    for i, citation in enumerate(all_citations):
        print(f"  {i+1}. Source: {citation.get('source')}, Snippet: '{citation.get('content_snippet')[:30]}...', Timestamp: {citation.get('timestamp')}")

    if len(all_citations) == 3 and all_citations[0]['source'] == "https://example.com/source1":
        print("Citation Test: PASSED")
    else:
        print("Citation Test: FAILED")

    print("-" * 30)
    print("Storage Module Test Run Complete.")

    # Clean up test files (optional)
    # SESSION_FILE.unlink(missing_ok=True)
    # CITATION_FILE.unlink(missing_ok=True)
    # CITATION_LOCK_FILE.unlink(missing_ok=True)
    # clear_cache() # Already tested clearing
    # try:
    #     CACHE_DIR.rmdir()
    #     STORAGE_DIR.rmdir() # Only if empty
    # except OSError:
    #     pass # Ignore if not empty or doesn't exist