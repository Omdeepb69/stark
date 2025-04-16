# main.py
# Entry point for the STARK application.

import argparse
import sys
import logging

# --- Project Library Imports (as specified in project details) ---
# These imports ensure necessary libraries are acknowledged, even if not directly used in main.py
try:
    import prompt_toolkit
    import requests
    import bs4 as beautifulsoup4 # Alias common practice
    import spacy
    # import nltk # Alternative NLP library
    # import transformers # Optional advanced NLP
    import sklearn as scikit_learn # Alias common practice
    import networkx
    import pandas
    # import rich # Optional TUI styling
    # import gensim # Optional topic modeling
except ImportError as e:
    print(f"Error: Missing required library: {e.name}", file=sys.stderr)
    print("Please install all required libraries listed in requirements.txt", file=sys.stderr)
    sys.exit(1)

# --- STARK Core Imports ---
try:
    from stark.app import StarkApp
except ImportError:
    print("Error: Could not import StarkApp from stark.app.", file=sys.stderr)
    print("Ensure the stark package is correctly installed or accessible in the Python path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}", file=sys.stderr)
    sys.exit(1)


# --- Logging Configuration ---
# Basic logging setup, can be expanded in StarkApp or a dedicated config module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """
    Sets up the command-line argument parser for STARK.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="STARK: Systematic Text Analysis & Research Kit - An autonomous AI research assistant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Initial research query to start the session with.",
        default=None
    )

    parser.add_argument(
        "-s", "--session",
        type=str,
        help="Path to a session file to load.",
        default=None
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging output."
    )

    # Add more arguments as needed for specific features:
    # e.g., --config path/to/config.yaml
    # e.g., --api-key YOUR_API_KEY
    # e.g., --offline # Run in offline mode using cached data

    return parser


def main() -> None:
    """
    Main function to parse arguments and run the STARK application.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    logger.info("Initializing STARK...")

    try:
        # Pass relevant arguments to the StarkApp constructor or run method
        app = StarkApp(initial_query=args.query, session_file=args.session)
        app.run()
    except KeyboardInterrupt:
        logger.info("STARK interrupted by user. Exiting.")
        sys.exit(0)
    except ImportError as e:
         # Catch potential import errors within StarkApp initialization if not caught earlier
        logger.error(f"Import error during app initialization: {e}. Ensure all dependencies are installed.")
        print(f"Error: Missing library required by STARK components: {e.name}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}", exc_info=True) # Log full traceback
        print(f"\nAn critical error occurred: {e}", file=sys.stderr)
        print("Please check the logs for more details.", file=sys.stderr)
        sys.exit(1)

    logger.info("STARK finished.")


if __name__ == "__main__":
    main()