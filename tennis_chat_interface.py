#!/usr/bin/env python3

import gradio as gr
import json
import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the LOCAL embedding agent (uses free transformers + OpenAI)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from chat_agent_embedding_qa_local import TennisChatAgentEmbeddingQALocal

class TennisChatInterface:
    def __init__(self, llm_provider: str = "openai"):
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
        
        print(f"[TENNIS] Tennis Chat Interface initialized with LOCAL embeddings + {self.llm_provider.upper()} LLM")
    
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
            print(f"DEBUG: search_player_matches returned {len(matches)} matches")
            
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
            
            print(f"DEBUG: Created {len(match_options)} match options")
            print(f"DEBUG: First option: {match_options[0] if match_options else 'None'}")
            
            # Return Radio update with all matches
            return gr.update(choices=match_options, value=None)
            
        except Exception as e:
            return gr.update(choices=[f"Error: {str(e)}"], value=None)
    
    def load_match(self, match_selection) -> str:
        """Load a specific match by scraping its data"""
        print(f"DEBUG: load_match called with match_selection: {match_selection}")
        print(f"DEBUG: self.matches length: {len(self.matches) if self.matches else 0}")
        
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
            
            # Step 1: Display initial status
            status_message = f"[LOADING] **Loading Match Data...**\n\n"
            status_message += f"**Selected Match:** {match_selection}\n\n"
            status_message += f"**Step 1/4:** [SCRAPE] Scraping detailed match data from TennisAbstract.com...\n"
            
            # Call the data collection script to scrape the specific match
            import subprocess
            import sys
            
            print(f"[SCRAPE] Scraping data for match {match_index}: {match_selection}")
            
            # Get player names from the selected match
            selected_match = self.matches[match_index]
            player1 = selected_match.get('player1', 'Player 1')
            player2 = selected_match.get('player2', 'Player 2')
            
            print(f"DEBUG: player1 = {player1}, player2 = {player2}")
            
            # Run the data collection script with the match index
            result = subprocess.run([
                sys.executable, 'run_data_collection_agent.py', 
                player1, player2, str(match_index)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return f"[ERROR] **Error during data scraping:** {result.stderr}"
            
            print("[SUCCESS] Data collection completed successfully")
            print(f"DEBUG: Data collection stdout: {result.stdout}")
            print(f"DEBUG: Data collection stderr: {result.stderr}")
            
            # Step 2: Update status
            status_message += f"[SUCCESS] **Step 1 Complete:** Match data scraped successfully!\n\n"
            status_message += f"**Step 2/4:** 📁 Loading JSON data file...\n"
            
            # Load the created JSON file
            import json
            import glob
            
            # Determine the filename that was created
            print(f"DEBUG: About to create player1_clean from player1 = {player1}")
            player1_clean = player1.replace(' ', '_').replace('.', '')
            player2_clean = player2.replace(' ', '_').replace('.', '')
            
            print(f"DEBUG: player1_clean = {player1_clean}, player2_clean = {player2_clean}")
            
            # Look for the JSON file that was created (includes date in filename)
            pattern = f"{player1_clean}_{player2_clean}_*.json"
            matching_files = glob.glob(pattern)
            
            print(f"DEBUG: Looking for files matching pattern: {pattern}")
            print(f"DEBUG: Found matching files: {matching_files}")
            
            if matching_files:
                filename = matching_files[0]
            else:
                return "[ERROR] **Error:** Could not find the created JSON file."
            
            # Load the match data
            with open(filename, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Step 3: Update status
            status_message += f"[SUCCESS] **Step 2 Complete:** JSON data loaded!\n\n"
            status_message += f"**Step 3/4:** [CONVERT] Converting match data to natural language...\n"
            
            # Convert to natural language and load embeddings
            print("[CONVERT] Converting match data to natural language...")
            print(f"DEBUG: match_data type: {type(match_data)}")
            print(f"DEBUG: match_data keys: {list(match_data.keys()) if isinstance(match_data, dict) else 'Not a dict'}")
            if isinstance(match_data, dict) and 'matches' in match_data:
                print(f"DEBUG: matches length: {len(match_data['matches'])}")
                print(f"DEBUG: first match type: {type(match_data['matches'][0])}")
                print(f"DEBUG: first match keys: {list(match_data['matches'][0].keys()) if isinstance(match_data['matches'][0], dict) else 'Not a dict'}")
            
            natural_language = self.chat_agent.convert_json_to_natural_language(match_data['matches'][0])
            
            # Save natural language file
            with open('FINAL_NATURAL_LANGUAGE.md', 'w', encoding='utf-8') as f:
                f.write(natural_language)
            
            print("[SAVE] Natural language file saved as 'FINAL_NATURAL_LANGUAGE.md'")
            
            # Step 4: Update status
            status_message += f"[SUCCESS] **Step 3 Complete:** Natural language conversion done!\n\n"
            status_message += f"**Step 4/4:** [EMBED] Loading embeddings for question answering...\n"
            
            # Generate embeddings from the converted data (now with fixed match result!)
            print("[EMBED] Generating embeddings from converted match data...")
            self.chat_agent.load_exact_full_format('FINAL_NATURAL_LANGUAGE.md')
            
            print("[SUCCESS] Match loaded successfully!")
            
            # Final success message
            match = match_data['matches'][0]
            date = match.get('basic', {}).get('date', 'N/A')
            tournament = match.get('basic', {}).get('tournament', 'N/A')
            p1 = match.get('basic', {}).get('player1', 'Player 1')
            p2 = match.get('basic', {}).get('player2', 'Player 2')
            
            final_message = status_message + f"[SUCCESS] **Step 4 Complete:** Embeddings loaded successfully!\n\n"
            final_message += f"[TENNIS] **Match Successfully Loaded!**\n\n"
            final_message += f"**Match:** {date} - {tournament}\n"
            final_message += f"**Players:** {p1} vs {p2}\n"
            final_message += f"**System:** Player detection is automatic based on your questions.\n\n"
            final_message += f"**[READY] Ready to Answer Questions!**\n\n"
            final_message += f"**Example Questions:**\n"
            final_message += f"- What was the final score?\n"
            final_message += f"- How many aces did each player hit?\n"
            final_message += f"- What were the break point statistics?\n"
            final_message += f"- Give me a comprehensive analysis of the match\n"
            
            return final_message
            
        except Exception as e:
            print(f"DEBUG: Exception caught: {str(e)}")
            print(f"DEBUG: player1 = {player1}, player2 = {player2}")
            return f"[ERROR] **Error loading match:** {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the currently loaded match"""
        if not question.strip():
            return "Please enter a question."
        
        if not self.chat_agent.chunks:
            return "No match is currently loaded. Please search for and load a match first."
        
        try:
            answer = self.chat_agent.ask_question(question.strip())
            return answer
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Tennis Match Chat Assistant", theme=gr.themes.Soft(), css="""
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
            gr.Markdown("# [TENNIS] Tennis Match Chat Assistant")
            gr.Markdown("Ask questions about tennis matches between any two players!")
            
            # Search and Load section - fully integrated
            gr.Markdown("### [SEARCH] Search & Load Matches")
            
            with gr.Row():
                player1_input = gr.Textbox(
                    label="Player 1 Name",
                    placeholder="e.g., Iga Swiatek",
                    lines=1
                )
                
                player2_input = gr.Textbox(
                    label="Player 2 Name", 
                    placeholder="e.g., Jessica Pegula",
                    lines=1
                )
            
            search_btn = gr.Button("[SEARCH] Search Matches", variant="primary")
            
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
                
                load_btn = gr.Button("[LOAD] Load Selected Match", variant="primary", size="lg")
                
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
                    gr.Markdown("### [ASK] Ask Questions")
                    
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What were the serve statistics? How many aces were hit?",
                        lines=4
                    )
                    
                    ask_btn = gr.Button("[ASK] Ask Question", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### [ANSWER] Answer")
                    
                    answer_output = gr.Markdown(
                        label="Answer",
                        value="Load a match first, then ask your question!"
                    )
            
            gr.Markdown("---")
            
            gr.Markdown("### [EXAMPLES] Example Questions")
            gr.Markdown("""
            - **Basic Info:** When was this match played? Who were the players?
            - **Serve Stats:** What were the serve statistics? How many aces were hit?
            - **Rally Analysis:** What were the rally outcomes? How many winners were hit?
            - **Key Points:** What were the key points? How did they perform on break points?
            - **Net Play:** How did they perform at the net? What were the net point statistics?
            - **Shot Analysis:** What shot types were used? How many forehand/backhand winners?
            - **Tactical Analysis:** Give me a tactical analysis of the match
            - **General:** Give me a match summary
            """)
            
            # Connect the interface components
            search_btn.click(
                fn=self.search_matches,
                inputs=[player1_input, player2_input],
                outputs=match_dropdown
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
    
    # Launch the interface - let Gradio find an available port
    interface.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main() 