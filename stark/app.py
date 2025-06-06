```python
# -*- coding: utf-8 -*-
"""
stark/app.py

Core application class for STARK. Manages the TUI, application state,
user interaction loop, and orchestrates calls to processing and storage modules.
"""

import argparse
import logging
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# --- Core Libraries ---
import pandas as pd
import networkx as nx

# --- TUI Libraries ---
from prompt_toolkit import Application, HTML
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign, FloatContainer, Float
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.application.current import get_app
from prompt_toolkit.widgets import SearchToolbar, TextArea, Frame, Box, Label

# --- Optional Rich Integration ---
try:
    from rich.logging import RichHandler
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes if rich is not available
    class Syntax:
        def __init__(self, code, lexer, theme, line_numbers): pass
    class Panel:
        def __init__(self, content, title): pass
    class Text:
        def __init__(self, content): pass


# --- Project Imports ---
# Use placeholder classes if actual modules are not yet available
try:
    from .processing import ProcessingEngine
except ImportError:
    print("Warning: stark.processing not found. Using dummy ProcessingEngine.", file=sys.stderr)
    class ProcessingEngine:
        """Dummy Processing Engine for STARK."""
        def __init__(self, logger):
            self.logger = logger
            self.logger.info("Initialized Dummy ProcessingEngine")
            self.nlp = None # Placeholder for spaCy/nltk/transformers model if loaded

        async def scrape_and_process(self, query: str, sources: list = None):
            self.logger.info(f"Dummy scrape/process started for query: '{query}' with sources: {sources}")
            await asyncio.sleep(1) # Simulate async work
            # Simulate some data processing
            summary = f"Synthesized information about '{query}' from {sources or ['web']}.\n"
            summary += "Key findings: A, B, C.\n"
            summary += f"Timestamp: {datetime.now()}"
            # Simulate graph generation (simple example)
            graph = nx.DiGraph()
            graph.add_node(query, type='query')
            graph.add_node("Source1", type='source')
            graph.add_node("Finding A", type='finding')
            graph.add_edge(query, "Source1")
            graph.add_edge("Source1", "Finding A")
            self.logger.info(f"Dummy scrape/process finished for query: '{query}'")
            return {"summary": summary, "graph": graph, "citations": {"Source1": "http://example.com"}}

        async def query_data(self, query: str, current_data: dict):
            self.logger.info(f"Dummy querying internal data for: '{query}'")
            await asyncio.sleep(0.5)
            # Simulate searching within previously gathered data
            results = f"Found relevant information for '{query}' within stored data:\n"
            if current_data and 'summary' in current_data:
                 if query.lower() in current_data['summary'].lower():
                     results += f"- Mention found in summary.\n"
            if current_data and 'graph' in current_data and current_data['graph']:
                 nodes_match = [n for n, data in current_data['graph'].nodes(data=True) if query.lower() in str(n).lower()]
                 if nodes_match:
                     results += f"- Related nodes in graph: {', '.join(nodes_match)}\n"

            return results if results else f"No specific information found for '{query}' in current session data."

        async def generate_graph_view(self, graph: nx.Graph | None):
            self.logger.info("Generating graph view")
            if graph is None or not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
                return "No graph data available to display."

            # Basic text representation of the graph
            view = f"--- Knowledge Graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges) ---\n"
            try:
                # Simple node list
                view += "Nodes:\n"
                for node, data in graph.nodes(data=True):
                    view += f"- {node} ({data.get('type', 'N/A')})\n"
                # Simple edge list
                view += "\nEdges:\n"
                for u, v, data in graph.edges(data=True):
                    view += f"- {u} -> {v} ({data.get('relation', 'related')})\n"

                # Potential: Use NetworkX drawing capabilities with matplotlib to save an image
                # Or, use libraries like `textual` or advanced `prompt_toolkit` widgets for interactive graphs
            except Exception as e:
                self.logger.error(f"Error generating graph view: {e}")
                view += f"\nError generating graph view: {e}"
            return view

        async def identify_gaps(self, current_data: dict):
            self.logger.info("Dummy identifying research gaps")
            await asyncio.sleep(1)
            # Simulate topic modeling or analysis
            gaps = "Potential Research Gaps:\n"
            gaps += "- Further investigation needed on topic X.\n"
            gaps += "- Relationship between Y and Z is unclear.\n"
            if not current_data or not current_data.get('graph'):
                gaps += "- More data required for meaningful gap analysis.\n"
            return gaps

try:
    from .storage import StorageManager
except ImportError:
    print("Warning: stark.storage not found. Using dummy StorageManager.", file=sys.stderr)
    class StorageManager:
        """Dummy Storage Manager for STARK."""
        def __init__(self, logger, session_file='stark_session.pkl'):
            self.logger = logger
            self.session_file = Path(session_file)
            self.logger.info(f"Initialized Dummy StorageManager with file: {self.session_file}")

        def save_session(self, data: dict):
            self.logger.info(f"Dummy saving session to {self.session_file}")
            try:
                # In a real scenario, use pickle, json, or a database
                # For this dummy version, just log it
                print(f"Session Data to Save (to {self.session_file}): {list(data.keys())}")
                # import pickle
                # with open(self.session_file, 'wb') as f:
                #     pickle.dump(data, f)
                self.logger.info("Dummy session save complete.")
                return True
            except Exception as e:
                self.logger.error(f"Dummy failed to save session: {e}")
                return False

        def load_session(self):
            self.logger.info(f"Dummy loading session from {self.session_file}")
            if self.session_file.exists():
                try:
                    # In a real scenario, load the data
                    # import pickle
                    # with open(self.session_file, 'rb') as f:
                    #     data = pickle.load(f)
                    # self.logger.info("Dummy session load complete.")
                    # return data
                    # For dummy, return some basic structure
                    self.logger.info("Dummy session file found, returning mock data.")
                    return {"history": ["Session loaded from previous state."], "citations": {}, "graph": None, "summary": "Loaded previous session."}
                except Exception as e:
                    self.logger.error(f"Dummy failed to load session: {e}")
                    return None
            else:
                self.logger.warning(f"Session file {self.session_file} not found.")
                return None

        def add_citation(self, session_data: dict, source: str, url: str):
            self.logger.info(f"Dummy adding citation: {source} - {url}")
            if 'citations' not in session_data:
                session_data['citations'] = {}
            session_data['citations'][source] = url
            return session_data

        def get_cache(self, key: str):
            self.logger.info(f"Dummy cache get for key: {key}")
            # No actual caching in dummy version
            return None

        def set_cache(self, key: str, value: any):
            self.logger.info(f"Dummy cache set for key: {key}")
            # No actual caching in dummy version
            pass

# --- Constants ---
APP_NAME = "STARK"
VERSION = "0.1.0"
DEFAULT_SESSION_FILE = "stark_session.pkl"
LOG_FILE = "stark_app.log"

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log_handler = logging.StreamHandler() # Default handler
if RICH_AVAILABLE:
    log_handler = RichHandler(rich_tracebacks=True, show_path=False)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s", # Simpler format for Rich
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode='a'), log_handler]
)
# Suppress noisy libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


# --- Custom Completer for Commands ---
class StarkCompleter(Completer):
    """Autocompleter for STARK commands."""
    def __init__(self, commands):
        self.commands = commands

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor.lstrip()
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        # Base commands
        if not text or ' ' not in text:
            for cmd in self.commands:
                if cmd.startswith(word_before_cursor):
                    yield Completion(cmd, start_position=-len(word_before_cursor), display=HTML(f'<skyblue>{cmd}</skyblue>'), display_meta=self.commands[cmd]['help'])

        # Command arguments (basic example)
        parts = text.split()
        if len(parts) > 0 and parts[0] in self.commands:
            cmd_info = self.commands[parts[0]]
            if 'subcommands' in cmd_info:
                 if len(parts) == 1 and text.endswith(' '): # Suggest subcommands after space
                     for sub_cmd in cmd_info['subcommands']:
                         yield Completion(sub_cmd, start_position=0, display=HTML(f'<violet>{sub_cmd}</violet>'))
                 elif len(parts) > 1: # Complete subcommand
                     current_sub_cmd = parts[1]
                     for sub_cmd in cmd_info['subcommands']:
                         if sub_cmd.startswith(current_sub_cmd):
                             yield Completion(sub_cmd, start_position=-len(current_sub_cmd), display=HTML(f'<violet>{sub_cmd}</violet>'))
            # Add more logic here for specific argument types (files, URLs, etc.)


# --- STARK Application Class ---
class StarkApp:
    """
    Main application class for the Systematic Text Analysis & Research Kit (STARK).
    Manages the TUI, state, and orchestrates processing and storage.
    """
    def __init__(self, session_file=DEFAULT_SESSION_FILE, initial_query=None, initial_sources=None):
        self.logger = logging.getLogger(APP_NAME)
        self.logger.info(f"Initializing {APP_NAME} v{VERSION}")

        self.session_file = session_file
        self.running = True
        self.show_help = False # Flag to control help popup

        # Core components
        self.storage = StorageManager(self.logger, self.session_file)
        self.processor = ProcessingEngine(self.logger)

        # Application state
        self.session_data = self.storage.load_session() or {"history": [], "citations": {}, "graph": None, "summary": None}
        if not self.session_data.get("history"): self.session_data["history"] = []
        self._log_to_output("Welcome to STARK! Type 'help' for commands, 'exit' to quit.")

        # TUI Elements
        self.output_buffer = Buffer(read_only=True)
        self.graph_buffer = Buffer(read_only=True) # For text-based graph view
        self.input_buffer = Buffer(
            completer=StarkCompleter(self.get_commands()),
            multiline=False,
            accept_handler=self._accept_input_handler,
            name="INPUT_BUFFER" # Name for keybindings
        )
        self.search_field = SearchToolbar() # Search for the output buffer

        # Layout
        self.layout = self._create_layout()

        # Keybindings
        self.key_bindings = self._create_key_bindings()

        # The prompt_toolkit Application instance
        self.pt_app = Application(
            layout=self.layout,
            key_bindings=self.key_bindings,
            style=self._get_style(),
            full_screen=True,
            mouse_support=True, # Enable mouse support (e.g., for scrolling)
            enable_page_navigation_bindings=True # Use PageUp/PageDown in buffers
        )

        # Handle initial command-line query
        if initial_query:
            # Schedule the initial processing after the event loop starts
            asyncio.ensure_future(self.handle_user_input(f"research {initial_query}"))


    def get_commands(self):
        """Returns a dictionary of available commands and their descriptions."""
        return {
            "research": {"help": "Start research: research <query> [--sources src1,src2]", "handler": self.command_research},
            "query": {"help": "Query collected data: query <term>", "handler": self.command_query},
            "graph": {"help": "Show knowledge graph", "handler": self.command_graph},
            "gaps": {"help": "Identify research gaps", "handler": self.command_gaps},
            "summary": {"help": "Show current research summary", "handler": self.command_summary},
            "citations": {"help": "Show collected citations", "handler": self.command_citations},
            "save": {"help": "Save current session", "handler": self.command_save},
            "load": {"help": "Load session (restarts)", "handler": self.command_load},
            "log": {"help": "Show application log file contents", "handler": self.command_log},
            "clear": {"help": "Clear the output screen", "handler": self.command_clear},
            "help": {"help": "Show this help message", "handler": self.command_help},
            "exit": {"help": "Exit STARK", "handler": self.command_exit},
        }

    def _create_layout(self):
        """Creates the prompt_toolkit layout."""

        # --- Output Window ---
        output_window = Window(
            content=BufferControl(buffer=self.output_buffer, search_buffer_control=self.search_field.control),
            wrap_lines=True,
            # style="class:output_window" # Apply style if needed
        )

        # --- Graph/Status Window ---
        graph_window = Window(
            content=BufferControl(buffer=self.graph_buffer),
            wrap_lines=False,
            align=WindowAlign.LEFT,
            # style="class:graph_window" # Apply style if needed
        )

        # --- Input Window ---
        input_window = Window(
            content=BufferControl(buffer=self.input_buffer),
            height=1,
            prompt=HTML("<bold><seagreen>STARK> </seagreen></bold>"),
            # style="class:input_window" # Apply style if needed
        )

        # --- Main Body Split ---
        # VSplit: Output on left (70%), Graph/Status on right (30%)
        main_body = VSplit([
            Frame(title="Output / Log", body=output_window),
            Window(width=1, char='│', style='class:separator'), # Vertical separator
            Frame(title="Graph / Status", body=graph_window),
        ], padding=0)

        # --- Help Popup (Float) ---
        help_content = "\n".join([f"<skyblue>{cmd}</skyblue>: {info['help']}" for cmd, info in self.get_commands().items()])
        help_popup = Float(
            content=Frame(
                title="Help - Commands (Ctrl+H to close)",
                body=Window(content=FormattedTextControl(HTML(help_content)), wrap_lines=False),
                style="class:dialog"
            ),
            top=2, bottom=2, left=5, right=5, # Position and size
        )

        # --- Root Container ---
        # HSplit: Main body on top, separator, search bar, input at bottom
        root_container = HSplit([
            main_body,
            Window(height=1, char='─', style='class:separator'), # Horizontal separator
            self.search_field, # Add search toolbar
            input_window,
        ])

        # --- Float Container for Help ---
        # Show help popup only when self.show_help is True
        float_container = FloatContainer(
            content=root_container,
            floats=[
                Float(content=help_popup, is_visible=lambda: self.show_help)
            ]
        )

        return Layout(float_container, focused_element=self.input_buffer)


    def _create_key_bindings(self):
        """Creates the key bindings."""
        kb = KeyBindings()

        @kb.add('c-c', eager=True)
        @kb.add('c-q', eager=True)
        def _(event):
            """ Exit application. """
            self._log_to_output("Exiting STARK...")
            self.command_exit() # Ensure save prompt if needed, etc.

        @kb.add('enter', filter=True) # Filter ensures it applies when input buffer is focused
        def _(event):
            """ Process command input when Enter is pressed. """
            # This handler is now primarily managed by `_accept_input_handler`
            # via the buffer's accept_handler. This binding can be a fallback
            # or removed if accept_handler is sufficient.
            # self._accept_input_handler(event.app.current_buffer)
            pass # Let accept_handler do the work

        @kb.add('c-l') # Clear screen
        def _(event):
            """ Clear output screen """
            self.command_clear()

        @kb.add('c-s') # Save session
        def _(event):
            """ Save session """
            asyncio.ensure_future(self.command_save())

        @kb.add('c-f') # Focus search
        def _(event):
            """ Focus search bar """
            get_app().layout.focus(self.search_field.control)

        @kb.add('c-h') # Toggle help
        def _(event):
            """ Toggle help popup """
            self.show_help = not self.show_help
            # Force redraw if layout doesn't update automatically
            event.app.invalidate()

        # Add PageUp/PageDown for scrolling main output if default isn't enough
        # (prompt_toolkit usually handles this with enable_page_navigation_bindings=True)

        return kb

    def _get_style(self):
        """Defines the application style."""
        return Style.from_dict({
            # Base styles
            '': 'bg:#1e1e1e #ffffff', # Default background and foreground
            'separator': 'fg:#444444',

            # Window styles
            'frame.border': 'fg:#888888',
            'frame.title': 'fg:#ffffff bold',
            # 'output_window': 'bg:#2a2a2a',
            # 'graph_window': 'bg:#252525',
            # 'input_window': 'bg:#333333',

            # Input field
            'input-field': 'bg:#333333 #ffffff',
            'prompt': 'fg:ansigreen bold',

            # Autocompletion menu
            'completion-menu.completion.current': 'bg:ansiblue #ffffff',
            'completion-menu.completion': 'bg:#444444 #cccccc',
            'completion-menu.meta.current': 'bg:ansiblue #eeeeee',
            'completion-menu.meta': 'bg:#555555 #dddddd',

            # Search toolbar
            'search-toolbar': 'bg:#333333',
            'search-toolbar.text': '#ffffff',

            # Dialogs / Popups
            'dialog': 'bg:#3a3a3a',
            'dialog frame.border': 'fg:ansiyellow',
            'dialog frame.title': 'fg:ansiyellow bold',

            # Custom semantic styles
            'info': 'fg:ansicyan',
            'warning': 'fg:ansiyellow',
            'error': 'fg:ansired bold',
            'success': 'fg:ansigreen',
            'query': 'fg:ansimagenta',
            'command': 'fg:ansibrightblue',
            'timestamp': 'fg:#888888',
        })

    def _log_to_output(self, message: str, style_class: str = ''):
        """Appends a message to the output buffer with optional styling."""
        if not message: return
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = HTML(f"<timestamp>[{timestamp}]</timestamp> ")
        
        # Basic HTML escaping for user content to prevent TUI errors
        # A more robust sanitizer might be needed for complex inputs
        escaped_message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        full_message = prefix + HTML(f"<{style_class}>{escaped_message}</{style_class}>" if style_class else escaped_message)
        
        current_text = self.output_buffer.text
        new_text = current_text + "\n" + full_message if current_text else full_message
        
        # Update buffer and scroll to end
        # Update buffer and scroll to end
        self.output_buffer.document = Document(text=new_text, cursor_position=len(new_text))
        # Make sure the output buffer is updated in the TUI
        self.pt_app.invalidate() # Force refresh

    def _accept_input_handler(self, buffer):
        """Handle input when 'Enter' is pressed on the input buffer."""
        text = buffer.text.strip()
        if not text:
            return # Skip empty input

        # Clear the input buffer for the next command
        buffer.reset(append_to_history=True) # Add to buffer history for arrow-up recall

        # Echo command to output
        self._log_to_output(f"{text}", style_class="command")

        # Process input using asyncio run_until_complete
        try:
            # Run the input handling as a coroutine
            future = asyncio.ensure_future(self.handle_user_input(text))
            # Return True to indicate the input was accepted
            return True
        except Exception as e:
            self._log_to_output(f"Error processing input: {e}", style_class="error")
            self.logger.exception(f"Error in input handler: {e}")
            return True

    async def handle_user_input(self, text: str):
        """
        Processes user input text and runs the appropriate command.
        This is an async function to handle commands that require async operations.
        """
        text = text.strip()
        if not text:
            return

        # Add to session history
        self.session_data["history"].append(text)

        # Simple command parsing (space-delimited, first word is command)
        parts = text.split()
        command = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        # Match command to handler using the commands dictionary
        commands = self.get_commands()
        if command in commands:
            try:
                # Call the command handler with any arguments
                await commands[command]["handler"](*args)
            except Exception as e:
                self._log_to_output(f"Error executing command: {e}", style_class="error")
                self.logger.exception(f"Error in command {command}: {e}")
        else:
            self._log_to_output(f"Unknown command: '{command}'. Type 'help' for available commands.", style_class="warning")

    # --- Command Handlers ---
    # Each of these corresponds to a command from get_commands()

    async def command_research(self, *args):
        """Handle the 'research' command to start a new research query."""
        if not args:
            self._log_to_output("Please provide a query. Usage: research <query> [--sources src1,src2]", style_class="warning")
            return

        # Parse arguments
        query = ""
        sources = None
        in_sources = False
        for arg in args:
            if arg == "--sources":
                in_sources = True
                continue
            
            if in_sources:
                sources = arg.split(',')
            else:
                query += f"{arg} "
        
        query = query.strip()
        
        if not query:
            self._log_to_output("Please provide a research query.", style_class="warning")
            return

        self._log_to_output(f"Starting research on: '{query}'" + 
                            (f" with sources: {sources}" if sources else ""), style_class="info")

        # Update graph display to show "processing"
        self.graph_buffer.document = Document(text="Processing research query...\nThis may take a moment.")
        self.pt_app.invalidate()  # Force refresh

        try:
            # Call the processing engine's scrape_and_process method
            result = await self.processor.scrape_and_process(query, sources)
            
            # Update session data with results
            self.session_data['summary'] = result.get('summary', "No summary generated.")
            self.session_data['graph'] = result.get('graph', None)
            if 'citations' in result:
                # Merge new citations with existing ones
                if 'citations' not in self.session_data:
                    self.session_data['citations'] = {}
                self.session_data['citations'].update(result['citations'])

            # Display summary in output
            self._log_to_output(f"Research completed. Summary:", style_class="success")
            self._log_to_output(self.session_data['summary'])

            # Update graph display
            if self.session_data['graph']:
                graph_view = await self.processor.generate_graph_view(self.session_data['graph'])
                self.graph_buffer.document = Document(text=graph_view)
            
            # Auto-save session
            await self.command_save()

        except Exception as e:
            self._log_to_output(f"Error during research: {e}", style_class="error")
            self.logger.exception(f"Research error: {e}")

    async def command_query(self, *args):
        """Query the collected data for specific information."""
        if not args:
            self._log_to_output("Please provide a query term. Usage: query <term>", style_class="warning")
            return

        query_term = " ".join(args)
        self._log_to_output(f"Querying collected data for: '{query_term}'", style_class="query")
        
        # Check if we have any data to query
        if not self.session_data.get('summary') and not self.session_data.get('graph'):
            self._log_to_output("No research data available to query. Run a research command first.", style_class="warning")
            return

        # Use the processor to search within existing data
        results = await self.processor.query_data(query_term, self.session_data)
        self._log_to_output(results)

    async def command_graph(self):
        """Show the knowledge graph."""
        if not self.session_data.get('graph'):
            self._log_to_output("No knowledge graph available. Run a research command first.", style_class="warning")
            return

        self._log_to_output("Generating graph view...", style_class="info")
        graph_view = await self.processor.generate_graph_view(self.session_data['graph'])
        self.graph_buffer.document = Document(text=graph_view)
        self._log_to_output("Graph view updated in right panel.", style_class="success")

    async def command_gaps(self):
        """Identify research gaps."""
        if not self.session_data.get('graph') and not self.session_data.get('summary'):
            self._log_to_output("Insufficient data to identify gaps. Run a research command first.", style_class="warning")
            return

        self._log_to_output("Analyzing for research gaps...", style_class="info")
        gaps = await self.processor.identify_gaps(self.session_data)
        self._log_to_output(gaps)

    async def command_summary(self):
        """Show current research summary."""
        if not self.session_data.get('summary'):
            self._log_to_output("No research summary available. Run a research command first.", style_class="warning")
            return

        self._log_to_output("Research Summary:", style_class="info")
        self._log_to_output(self.session_data['summary'])

    async def command_citations(self):
        """Show collected citations."""
        if not self.session_data.get('citations') or not self.session_data['citations']:
            self._log_to_output("No citations available. Run a research command first.", style_class="warning")
            return

        self._log_to_output("Citations:", style_class="info")
        for source, url in sorted(self.session_data['citations'].items()):
            self._log_to_output(f"- {source}: {url}")

    async def command_save(self):
        """Save current session to file."""
        if self.storage.save_session(self.session_data):
            self._log_to_output(f"Session saved to {self.session_file}", style_class="success")
        else:
            self._log_to_output("Failed to save session", style_class="error")

    async def command_load(self, file_path=None):
        """Load session from file."""
        if file_path:
            self.session_file = file_path
            self._log_to_output(f"Setting session file to: {file_path}", style_class="info")

        loaded_data = self.storage.load_session()
        if loaded_data:
            self.session_data = loaded_data
            self._log_to_output(f"Session loaded from {self.session_file}", style_class="success")
            
            # Update display with loaded data
            if self.session_data.get('summary'):
                self._log_to_output("Loaded summary:", style_class="info")
                self._log_to_output(self.session_data['summary'])
            
            if self.session_data.get('graph'):
                # Schedule graph refresh for later to avoid blocking
                asyncio.ensure_future(self.command_graph())
        else:
            self._log_to_output(f"Could not load session from {self.session_file}", style_class="error")

    async def command_log(self):
        """Show application log file contents."""
        try:
            with open(LOG_FILE, 'r') as f:
                # Get last N lines of log file (tail)
                lines = f.readlines()[-50:] # Last 50 lines
                self._log_to_output(f"Last {len(lines)} lines of log:", style_class="info")
                self._log_to_output("".join(lines))
        except Exception as e:
            self._log_to_output(f"Failed to read log file: {e}", style_class="error")

    async def command_clear(self):
        """Clear the output screen."""
        self.output_buffer.document = Document(text="")
        self._log_to_output("Output cleared.", style_class="info")

    async def command_help(self):
        """Show help message."""
        self.show_help = True
        self._log_to_output("Showing help popup (Ctrl+H to close).", style_class="info")

    async def command_exit(self):
        """Exit the application."""
        self._log_to_output("Exiting STARK...", style_class="info")
        
        # Prompt for save if there's unsaved data from this session
        # This could be enhanced with a proper TUI dialog
        if self.session_data.get('summary') or self.session_data.get('graph'):
            # For now, just auto-save
            self._log_to_output("Saving session before exit...", style_class="info")
            await self.command_save()
        
        # Signal application to exit
        self.running = False
        # Schedule app exit
        self.pt_app.exit()

    def run(self):
        """Run the STARK application."""
        try:
            self.pt_app.run()
        except KeyboardInterrupt:
            self.logger.info("STARK interrupted by user. Exiting.")
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred: {e}")
            print(f"\nA critical error occurred: {e}", file=sys.stderr)
        finally:
            # Any cleanup needed
            self.logger.info("STARK finished.")


# When running as a standalone module
if __name__ == "__main__":
    app = StarkApp()
    app.run()
