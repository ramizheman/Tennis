#!/usr/bin/env python3

import gradio as gr
import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Import the match questions agent (uses taxonomy-based routing)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from chat_match_questions import TennisChatAgentEmbeddingQALocal
from data_collection_agent import TennisDataCollector

class TennisChatInterface:
    def __init__(self, llm_provider: str = "gemini"):
        """
        Initialize the tennis chat interface.
        
        Args:
            llm_provider: "openai", "claude", or "gemini"
        """
        # Check if API keys are available
        self.llm_provider = llm_provider.lower()
        self._check_api_keys()
        
        # Initialize the LOCAL embedding agent (uses free transformers)
        self.chat_agent = TennisChatAgentEmbeddingQALocal(llm_provider=self.llm_provider)
        self.matches = []
        self.current_match_index = None
        
        # Player names cache
        self.cache_file = "cached_player_names.json"
        self.cache_age_days = 7
        self.player_names = self._load_player_names()
        
        print(f"[TENNIS] Tennis Chat Interface initialized with LOCAL embeddings + {self.llm_provider.upper()} LLM")
        print(f"[TENNIS] Loaded {len(self.player_names)} player names for autocomplete")
    
    def _check_api_keys(self):
        """Check if required API keys are available."""
        if self.llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        elif self.llm_provider == "claude":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        elif self.llm_provider == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
        else:
            raise ValueError("llm_provider must be 'openai', 'claude', or 'gemini'")
    
    def _load_player_names(self, force_refresh: bool = False) -> List[str]:
        """
        Load player names from cache or refresh if needed.
        
        Args:
            force_refresh: If True, fetch fresh data regardless of cache age
            
        Returns:
            List of player names for autocomplete
        """
        # Check if cache exists and is valid
        cache_valid = False
        if os.path.exists(self.cache_file) and not force_refresh:
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cache_date = datetime.fromisoformat(cache_data['timestamp'])
                    cache_age = datetime.now() - cache_date
                    
                    if cache_age < timedelta(days=self.cache_age_days):
                        cache_valid = True
                        player_names = cache_data['player_names']
                        print(f"[CACHE] Loaded {len(player_names)} player names from cache (age: {cache_age.days} days)")
                        return player_names
                    else:
                        print(f"[CACHE] Cache expired (age: {cache_age.days} days > {self.cache_age_days} days)")
            except Exception as e:
                print(f"[CACHE] Error reading cache: {e}")
        
        # Cache doesn't exist, is invalid, or force refresh requested
        print("[CACHE] Fetching fresh player names...")
        collector = TennisDataCollector()
        player_names = collector.get_all_player_names()
        
        if player_names:
            # Save to cache
            self._save_player_cache(player_names)
            return player_names
        else:
            print("[CACHE] Warning: No player names fetched, using empty list")
            return []
    
    def _save_player_cache(self, player_names: List[str]):
        """Save player names to cache file."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'player_names': player_names,
                'count': len(player_names)
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            print(f"[CACHE] Saved {len(player_names)} player names to cache")
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")
    
    def refresh_player_names(self) -> Tuple[gr.Dropdown, gr.Dropdown, str]:
        """
        Manually refresh the player names cache.
        Returns updated dropdown choices for both player inputs.
        """
        print("[CACHE] Manual refresh requested...")
        self.player_names = self._load_player_names(force_refresh=True)
        
        # Return updated dropdowns and status message
        status_msg = f"✅ Player list refreshed! Loaded {len(self.player_names)} players."
        return (
            gr.Dropdown(choices=self.player_names),
            gr.Dropdown(choices=self.player_names),
            status_msg
        )
        
    def search_matches(self, player1: str, player2: str):
        """Search for matches between two players (without scraping details)"""
        if not player1.strip() or not player2.strip():
            return gr.update(choices=[], value=None)
        
        try:
            # Initialize the data collection agent to search for matches
            from data_collection_agent import TennisDataCollector
            collector = TennisDataCollector()
            
            print(f"[SEARCH] Searching for matches between {player1} and {player2}...")
            
            # Search for matches (without scraping details)
            matches = collector.search_player_matches(player1.strip(), player2.strip())
            
            if not matches:
                return gr.update(choices=[f"No matches found between {player1} and {player2}."], value=None)
            
            # Store all matches for the interface
            self.matches = matches
            
            # Create match options for radio list
            match_options = []
            for i, match in enumerate(self.matches):
                date = match.get('date', 'N/A')
                tournament = match.get('tournament', 'N/A')
                p1 = match.get('player1', 'Player 1')
                p2 = match.get('player2', 'Player 2')
                
                match_text = f"{date} - {tournament}: {p1} vs {p2}"
                match_options.append(match_text)
            
            # Return Radio update with all matches
            return gr.update(choices=match_options, value=None)
            
        except Exception as e:
            return gr.update(choices=[f"Error: {str(e)}"], value=None)
    
    def load_match(self, match_selection) -> str:
        """Load a specific match by scraping its data"""
        
        if not match_selection or match_selection == "":
            return "Please select a match from the dropdown."
        
        if not self.matches:
            return "Please search for matches first and select a valid match."
        
        # Initialize variables at the top level
        player1 = None
        player2 = None
        
        try:
            # Find the selected match index
            match_index = None
            for i, match in enumerate(self.matches):
                date = match.get('date', 'N/A')
                tournament = match.get('tournament', 'N/A')
                p1 = match.get('player1', 'Player 1')
                p2 = match.get('player2', 'Player 2')
                
                match_text = f"{date} - {tournament}: {p1} vs {p2}"
                if match_text == match_selection:
                    match_index = i
                    break
            
            if match_index is None:
                return "Selected match not found. Please search again."
            
            # Get player names and match info from the selected match
            selected_match = self.matches[match_index]
            player1 = selected_match.get('player1', 'Player 1')
            player2 = selected_match.get('player2', 'Player 2')
            match_date = selected_match.get('date', '2025-06-28')
            
            # Construct expected filename
            player1_clean = player1.replace(' ', '_').replace('.', '')
            player2_clean = player2.replace(' ', '_').replace('.', '')
            date_clean = match_date.replace('-', '')
            expected_filename = f"{player1_clean}_{player2_clean}_{date_clean}.json"
            
            # Check if match data already exists (cached from previous load)
            skip_scraping = False
            if os.path.exists(expected_filename):
                print(f"[CACHE] Match already cached: {expected_filename}")
                skip_scraping = True
                status_message = f"**Loading Match Data...**\n\n"
                status_message += f"**Selected Match:** {match_selection}\n\n"
            else:
                print(f"[SCRAPE] First time loading this match, scraping from Tennis Abstract...")
                status_message = f"**Loading Match Data...**\n\n"
                status_message += f"**Selected Match:** {match_selection}\n\n"
            
            # Only scrape if cache doesn't exist
            if not skip_scraping:
                # Call the data collection script to scrape the specific match
                import subprocess
                import sys
                
                print(f"[SCRAPE] Scraping data for match {match_index}: {match_selection}")
                
                # Run the data collection script with the match index
                result = subprocess.run([
                    sys.executable, 'run_data_collection_agent.py', 
                    player1, player2, str(match_index)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    return f"❌ **Error during data scraping:** {result.stderr}"
                
                print("[SUCCESS] Data collection completed successfully")
            
            # Load the JSON file (either just scraped or from cache)
            import json
            
            filename = expected_filename
            
            if not os.path.exists(filename):
                return f"❌ **Error:** Could not find the JSON file: {filename}"
            
            # Load the match data
            with open(filename, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Create match-specific filename prefix (without .json extension)
            match_prefix = expected_filename.replace('.json', '')
            embeddings_cache_file = f"{match_prefix}_faiss.pkl"  # Check for the FAISS index file
            
            # Always generate the natural language file in markdown format
            # Get the Tennis folder path (where this script is located)
            tennis_folder = os.path.dirname(os.path.abspath(__file__))
            
            print("[CONVERT] Converting match data to natural language format...")
            natural_language = self.chat_agent.convert_json_to_natural_language(match_data['matches'][0])
            nl_filename = os.path.join(tennis_folder, f"{match_prefix}_NL.md")
            
            with open(nl_filename, 'w', encoding='utf-8') as f:
                f.write(natural_language)
            print(f"[SAVE] Natural language file saved as '{nl_filename}'")
            
            # Extract structured point-by-point data from JSON
            match = match_data['matches'][0]
            pointlog_rows = match.get('pointlog_rows', [])
            print(f"[EXTRACT] Found {len(pointlog_rows)} points in structured data")
            
            # Check if embeddings are already cached
            if os.path.exists(embeddings_cache_file):
                print(f"[CACHE] Embeddings already cached: {embeddings_cache_file}")
                
                # Load pre-computed embeddings directly
                success = self.chat_agent.load_embeddings_from_disk(match_prefix)
                
                if not success:
                    # Cache corrupted, regenerate
                    print("[WARN] Cached embeddings corrupted, regenerating...")
                    self.chat_agent.load_exact_full_format(nl_filename)
                    # NOTE: load_exact_full_format already extracts PBP from NL file
                    # which has [Point won by:] tags - DON'T overwrite with JSON!
                    self.chat_agent.save_embeddings_to_disk(match_prefix)
                    print(f"[CACHE] Embeddings saved for future use: {embeddings_cache_file}")
                else:
                    # Even if embeddings are cached, we need to load point-by-point from NL file
                    # (not JSON) to get [Point won by:] tags
                    if not self.chat_agent.point_by_point:
                        print("[EXTRACT] Loading point-by-point from NL file (with point winners)...")
                        self.chat_agent._extract_point_by_point_from_nl_file(nl_filename)
            else:
                # No cache - generate fresh embeddings
                print("[EMBED] Generating embeddings from converted match data...")
                self.chat_agent.load_exact_full_format(nl_filename)
                # NOTE: load_exact_full_format already extracts PBP from NL file
                # which has [Point won by:] tags - DON'T overwrite with JSON!
                
                # Save embeddings for future loads
                self.chat_agent.save_embeddings_to_disk(match_prefix)
                print(f"[CACHE] Embeddings saved for future use: {embeddings_cache_file}")
            
            print("[SUCCESS] Match loaded successfully!")
            
            # Final success message
            match = match_data['matches'][0]
            date = match.get('basic', {}).get('date', 'N/A')
            tournament = match.get('basic', {}).get('tournament', 'N/A')
            p1 = match.get('basic', {}).get('player1', 'Player 1')
            p2 = match.get('basic', {}).get('player2', 'Player 2')
            
            # final_message = status_message + f"[SUCCESS] **Step 4 Complete:** Embeddings loaded successfully!\n\n"
            final_message = status_message + f"\n"
            final_message += f"✅ **Match Successfully Loaded!**\n\n"
            final_message += f"**Match:** {date} - {tournament}\n"
            final_message += f"**Players:** {p1} vs {p2}\n"
            final_message += f"**System:** Player detection is automatic based on your questions.\n\n"
            final_message += f"**Ready to Answer Questions!** 🎾\n\n"
            final_message += f"**Example Questions:**\n\n"
            final_message += f"**Simple Stats:**\n"
            final_message += f"- What was the final score?\n"
            final_message += f"- How many aces did {p1} hit?\n"
            final_message += f"- How many forehand winners did each player hit?\n"
            final_message += f"- How many double faults did each player have?\n"
            final_message += f"- What was each player's second serve win percentage?\n"
            final_message += f"- How many cross court winners did each player have?\n\n"
            final_message += f"**Set Analysis:**\n"
            final_message += f"- What happened in Set 3?\n"
            final_message += f"- How did {p2}'s tactics change in Set 1 versus Set 3?\n"
            final_message += f"- How did {p2}'s serve effectiveness change across sets?\n\n"
            final_message += f"**Tactical Comparisons:**\n"
            final_message += f"- When {p1} served to the T versus wide, which was more effective?\n"
            final_message += f"- Did {p2} play more aggressively on second serve returns compared to first serve returns?\n"
            final_message += f"- Who controlled rallies from the baseline more effectively?\n\n"
            final_message += f"**Advanced Analysis:**\n"
            final_message += f"- When one player won a break point, did they play more aggressively in the next game? Calculate the momentum carry-over effect\n"
            final_message += f"- How did {p1}'s first serves change when facing break points?\n"
            final_message += f"- Tell the parallel journey of both players through the match - how did each player's mental and tactical state evolve?\n"
            final_message += f"- Provide a strategic summary of the match\n"
            
            return final_message
            
        except Exception as e:
            return f"❌ **Error loading match:** {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the currently loaded match"""
        if not question.strip():
            return "Please enter a question."
        
        if not self.chat_agent.chunks:
            return "No match is currently loaded. Please search for and load a match first."
        
        try:
            # Always use gemini-2.5-flash
            original_model = self.chat_agent.model
            self.chat_agent.model = "gemini-2.5-flash"
            
            answer = self.chat_agent.ask_question(question.strip())
            
            # Restore original model
            self.chat_agent.model = original_model
            
            return answer
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print full traceback to console
            return f"Error answering question: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Tennis Match Q&A Assistant", theme=gr.themes.Soft(), css="""
            .two-column-radio {
                column-count: 2 !important;
                column-gap: 20px !important;
                column-fill: auto !important;
            }
            .two-column-radio label {
                display: block !important;
                width: 100% !important;
                margin-bottom: 8px !important;
                margin-top: 0 !important;
                padding-top: 0 !important;
                break-inside: avoid !important;
                vertical-align: top !important;
                white-space: normal !important;
            }
            .two-column-radio fieldset {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
        """) as interface:
            gr.Markdown("# 🎾 Tennis Match Q&A Assistant")
            gr.Markdown("Ask questions about tennis matches between any two players and get detailed analysis!")
            
            # Search and Load section - fully integrated
            gr.Markdown("### Search & Load Matches")
            gr.Markdown("**Note:** Last name searches work best (e.g., 'Federer' or 'Nadal')")
            
            with gr.Row():
                player1_input = gr.Dropdown(
                    label="Player 1 Name",
                    choices=self.player_names,
                    allow_custom_value=True,
                    filterable=True,
                    value=None,
                    info="Start typing to filter player names"
                )
                
                player2_input = gr.Dropdown(
                    label="Player 2 Name", 
                    choices=self.player_names,
                    allow_custom_value=True,
                    filterable=True,
                    value=None,
                    info="Start typing to filter player names"
                )
            
            with gr.Row():
                search_btn = gr.Button("🔍 Search Matches", variant="primary", scale=3)
                refresh_btn = gr.Button("🔄 Refresh Player List", variant="secondary", scale=1)
            
            # Refresh status output
            refresh_status = gr.Textbox(
                label="",
                value="",
                visible=False,
                lines=1
            )
            
            # Integrated match selection - single interactive list with 2 columns
            with gr.Group():
                gr.Markdown("**Search Results - Click a match to select it:**")
                
                match_dropdown = gr.Radio(
                    label="",
                    choices=[],
                    value=None,
                    interactive=True,
                    elem_classes="two-column-radio"
                )
                
                load_btn = gr.Button("📥 Load Selected Match", variant="primary", size="lg")
                
                load_output = gr.Textbox(
                    label="Status",
                    value="Search for matches above, select one from the list, then click Load.",
                    lines=6,
                    interactive=False
                )
            
            gr.Markdown("---")
            
            # Q&A section - side by side, equal width
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ask Questions")
                    
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What were the serve statistics? How many aces were hit?",
                        lines=4
                    )
                    
                    ask_btn = gr.Button("💬 Ask Question", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Answer")
                    
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="Load a match first, then ask your question!",
                        height=400
                    )
            
            gr.Markdown("---")
            gr.Markdown("*Powered by Tennis Abstract Match Charting Project (http://www.tennisabstract.com/charting/) Data*")
            
            # Connect the interface components
            search_btn.click(
                fn=self.search_matches,
                inputs=[player1_input, player2_input],
                outputs=match_dropdown
            )
            
            refresh_btn.click(
                fn=self.refresh_player_names,
                inputs=[],
                outputs=[player1_input, player2_input, refresh_status]
            )
            
            load_btn.click(
                fn=self.load_match,
                inputs=[match_dropdown],
                outputs=[load_output]
            )
            
            ask_btn.click(
                fn=self.ask_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
            
            # Allow Enter key to submit questions
            question_input.submit(
                fn=self.ask_question,
                inputs=[question_input],
                outputs=[answer_output]
            )
        
        return interface

def main():
    """Main function to run the chat interface"""
    import socket
    import subprocess
    import sys
    
    def kill_process_on_port(port):
        """Kill any process using the specified port"""
        try:
            # Find process using the port
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"Killing process {pid} on port {port}")
                        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                        return True
        except Exception as e:
            print(f"Could not kill process on port {port}: {e}")
        return False
    
    # Try to kill any existing process on common Gradio ports
    for port in [7860, 7861, 7862, 7863, 8080]:
        kill_process_on_port(port)
    
    chat_interface = TennisChatInterface()
    interface = chat_interface.create_interface()
    
    # Launch the interface
    # On Hugging Face Spaces, the correct port is provided via the PORT environment variable.
    # Locally, this will default to 7860.
    port = int(os.getenv("PORT", "7860"))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main() 