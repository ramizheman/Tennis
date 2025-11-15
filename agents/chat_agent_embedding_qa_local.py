#!/usr/bin/env python3

import json
import os
import numpy as np
import faiss
import pickle
from typing import Dict, List, Any, Optional
import tiktoken
import time
import pickle

class TennisChatAgentEmbeddingQALocal:
    """
    Tennis Chat Agent using LOCAL embedding model (sentence-transformers) and LLM for answering.
    Uses FREE local embeddings with sophisticated retrieval logic.
    Supports Claude, Gemini, and OpenAI for answering questions.
    """
    
    def __init__(self, llm_provider: str = "gemini", api_key: str = None, model: str = None):
        """
        Initialize with specified LLM provider and LOCAL embeddings.
        
        Args:
            llm_provider: "openai", "claude", or "gemini" (for answering questions, default: gemini)
            api_key: API key for the provider (optional, will use env vars)
        """
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        self.custom_model = model  # Allow custom model override
        self.chunks = []
        self.index = None
        self.metadata_store = []
        self.match_id = "match_data"  # Generic default, overwritten when loading match
        self.player1 = None
        self.player2 = None
        
        # Rate limiting
        self.last_api_call = 0
        self.min_delay = 2.0  # Minimum 2 seconds between API calls
        
        # Initialize local embedding model (FREE!)
        print("Loading LOCAL embedding model (sentence-transformers)...")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            print("Local embedding model loaded (FREE)!")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Initialize the appropriate LLM client
        self._init_llm_client()
        
    def _init_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "claude":
            try:
                import anthropic
                api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable required for Claude")
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
                
        elif self.llm_provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable required for Gemini")
                genai.configure(api_key=api_key)
                self.client = genai
                self.model = "gemini-2.5-flash"  # Latest model with enhanced reasoning (June 2025)
            except ImportError:
                raise ImportError("Please install google-generativeai: pip install google-generativeai")
                
        elif self.llm_provider == "openai":
            try:
                from openai import OpenAI
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable required for OpenAI")
                self.client = OpenAI(api_key=api_key)
                self.model = self.custom_model or "gpt-4o-mini"  # Allow custom model
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError("llm_provider must be 'claude', 'gemini', or 'openai'")
        
    def _parse_match_score(self, content: str) -> None:
        """
        Parse the match score from the markdown content and build set mapping.
        Example: "Carlos Alcaraz d. Jannik Sinner 4-6 6-7(4) 6-4 7-6(3)"
        Format: "Winner d. Loser winner_score-loser_score ..."
        """
        import re
        
        # Initialize
        self.match_score = None
        self.set_mapping = {}  # Maps set number to set score (e.g., {1: "0-0", 2: "0-1", 3: "0-2", 4: "1-2"})
        self.total_sets = 0
        
        # Find the Final Score line
        score_match = re.search(r'Final Score:\s*(.+?)(?:\n|$)', content)
        if not score_match:
            print("[WARN] Could not find Final Score in markdown")
            return
        
        score_line = score_match.group(1).strip()
        self.match_score = score_line
        
        # Parse the format: "Winner d. Loser score score score"
        match_result = re.match(r'(.+?)\s+d\.\s+(.+?)\s+([\d\-\(\)\s]+)$', score_line)
        if not match_result:
            print("[WARN] Could not parse match result format:", score_line)
            return
        
        winner_name = match_result.group(1).strip()
        loser_name = match_result.group(2).strip()
        scores_str = match_result.group(3).strip()
        
        # Extract the set scores
        set_scores = re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', scores_str)
        
        if not set_scores:
            print("[WARN] Could not parse set scores from:", score_line)
            return
        
        self.total_sets = len(set_scores)
        
        # Determine which player is player1 and which is player2
        # (based on who is mentioned first in self.player1/player2)
        winner_is_player1 = (hasattr(self, 'player1') and 
                            self.player1 and 
                            winner_name.lower().replace(' ', '') == self.player1.lower().replace(' ', ''))
        
        # Build cumulative set score
        # Scores in the format are: winner_score-loser_score for each set
        player1_sets = 0
        player2_sets = 0
        
        for i, (score1, score2) in enumerate(set_scores, 1):
            # Record the set score BEFORE this set starts
            self.set_mapping[i] = f"{player1_sets}-{player2_sets}"
            
            # Determine who won this set
            # score1 is winner's score, score2 is loser's score
            if int(score1) > int(score2):
                # Winner won this set
                if winner_is_player1:
                    player1_sets += 1
                else:
                    player2_sets += 1
            else:
                # Loser won this set
                if winner_is_player1:
                    player2_sets += 1
                else:
                    player1_sets += 1
        
        print(f"[MATCH] Final Score: {self.match_score}")
        print(f"[MATCH] Winner: {winner_name} | Loser: {loser_name}")
        print(f"[MATCH] Winner is Player1: {winner_is_player1}")
        print(f"[MATCH] Set Mapping: {self.set_mapping}")
    
    def _detect_set_reference(self, query: str) -> int:
        """
        Detect if the query references a specific set.
        Returns the set number (1, 2, 3, 4, 5) or None.
        Returns None if MULTIPLE sets are mentioned (e.g., "set 3 and set 4").
        """
        import re
        
        query_lower = query.lower()
        
        # Check for multiple set references (e.g., "set 3 and set 4")
        # If found, return None to avoid filtering
        set_count = len(re.findall(r'set\s+\d+', query_lower))
        if set_count > 1:
            return None
        
        # Direct number references
        # Match "set 3", "in set 3", "3rd set", "the 3rd set", etc.
        set_match = re.search(r'set\s+(\d+)', query_lower)  # "set 3"
        if not set_match:
            set_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+set', query_lower)  # "3rd set"
        if set_match:
            return int(set_match.group(1))
        
        # Word-based references
        set_words = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5
        }
        
        for word, num in set_words.items():
            if f"{word} set" in query_lower:
                return num
        
        # Special references
        if any(word in query_lower for word in ['final set', 'deciding set', 'last set']):
            return self.total_sets if self.total_sets else None
        
        return None
    
    def _detect_multiple_set_references(self, query: str) -> List[int]:
        """
        Detect if the query references MULTIPLE specific sets (for comparison questions).
        Returns a list of set numbers [1, 5] or empty list if not a multi-set comparison.
        Examples: "Set 1 vs Set 5", "Set 1 versus Set 5", "compare Set 1 and Set 5"
        """
        import re
        
        query_lower = query.lower()
        
        # Find all set number references
        set_numbers = []
        
        # Match "set 1", "set 5", etc.
        for match in re.finditer(r'set\s+(\d+)', query_lower):
            set_num = int(match.group(1))
            if 1 <= set_num <= 5 and set_num not in set_numbers:
                set_numbers.append(set_num)
        
        # Also check for word-based references
        set_words = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5
        }
        
        for word, num in set_words.items():
            if f"{word} set" in query_lower and num not in set_numbers:
                set_numbers.append(num)
        
        # Only return if we found multiple sets AND it's a comparison question
        if len(set_numbers) >= 2:
            comparison_indicators = ['vs', 'versus', 'compared to', 'compare', 'vs.', 'v.']
            if any(indicator in query_lower for indicator in comparison_indicators):
                set_numbers.sort()  # Return in order [1, 5]
                return set_numbers
        
        return []
    
    def _detect_game_reference(self, query: str) -> str:
        """
        Detect if the query references a specific game or game score.
        Returns a game score pattern like "3-2" or None.
        """
        import re
        
        query_lower = query.lower()
        
        # Game score patterns (e.g., "at 4-3", "when it was 5-5", "3-2")
        game_score_match = re.search(r'(?:at\s+|when\s+it\s+was\s+|score\s+was\s+)?(\d+)-(\d+)', query_lower)
        if game_score_match:
            return f"{game_score_match.group(1)}-{game_score_match.group(2)}"
        
        # Game number references (e.g., "game 5", "5th game")
        game_num_match = re.search(r'(?:game\s+|the\s+)?(\d+)(?:st|nd|rd|th)?\s+game', query_lower)
        if game_num_match:
            # This would need more complex logic to determine the exact game score
            # For now, just return a marker that we need game-specific data
            return f"game_{game_num_match.group(1)}"
        
        return None
        
    def load_exact_full_format(self, file_path: str = "EXACT_FULL_FORMAT.md") -> None:
        """
        Load and process the specified natural language file into chunks with embeddings.
        """
        print(f"Loading {file_path}...")
        
        # Extract player names from filename (e.g., "Jannik_Sinner_Carlos_Alcaraz_20250608_NL.md")
        import os
        filename = os.path.basename(file_path)
        parts = filename.replace('_NL.md', '').split('_')
        if len(parts) >= 4:
            # Reconstruct player names (handle names with spaces)
            # Find the date (8 digits) to split the names
            date_idx = next((i for i, part in enumerate(parts) if part.isdigit() and len(part) == 8), None)
            if date_idx and date_idx >= 2:
                player1_parts = parts[:date_idx//2] 
                player2_parts = parts[date_idx//2:date_idx]
                self.player1 = ' '.join(player1_parts)
                self.player2 = ' '.join(player2_parts)
                print(f"Extracted player names: {self.player1} vs {self.player2}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract match score for set mapping (needed for transformation)
        self._parse_match_score(content)
        
        # Transform point-by-point data to add shot attribution (NEW!)
        content = self._transform_point_by_point_in_content(content)
        
        # Split content into sections (tables vs point-by-point)
        sections = self._split_into_sections(content)
        
        # Process each section
        all_chunks = []
        for section_name, section_text in sections.items():
            print(f"Processing section: {section_name}")
            
            # Determine chunk type based on section content
            chunk_type = self._determine_chunk_type(section_name, section_text)
            
            chunks = self._create_chunks_with_metadata(
                section_text, 
                chunk_type=chunk_type,
                section=section_name,
                match_id=self.match_id
            )
            all_chunks.extend(chunks)
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        self.chunks = self._embed_chunks(all_chunks)
        
        # Create FAISS index
        print("Creating vector index...")
        self._create_vector_index()
        
        print(f"Loaded {len(self.chunks)} chunks with embeddings")
        
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """
        Split the natural language content using improved semantic + size-aware chunking.
        Creates optimal chunks based on content type and size for better searchability.
        """
        sections = {}
        lines = content.split('\n')
        
        # Define section patterns and chunking strategies
        section_config = {
            'MATCH OVERVIEW:': 'small',
            'RALLY OUTCOMES STATISTICS:': 'medium', 
            'OVERVIEW STATISTICS:': 'small',
            'SERVE1 STATISTICS (SUMMARY):': 'medium',
            'SERVE1 STATISTICS (DETAILED):': 'large',
            'SERVE2 STATISTICS (SUMMARY):': 'medium',
            'SERVE2 STATISTICS (DETAILED):': 'large',
            'RETURN1 STATISTICS (DETAILED):': 'large',
            'RETURN2 STATISTICS (DETAILED):': 'large',
            'KEY POINTS STATISTICS (SERVES):': 'medium',
            'KEY POINTS STATISTICS (RETURNS):': 'medium',
            'SHOTS1 STATISTICS:': 'large',
            'SHOTS2 STATISTICS:': 'large',
            'SHOTDIR1 STATISTICS:': 'large',
            'SHOTDIR2 STATISTICS:': 'large',
            'EXPLICIT TOTALS FOR SHOT DIRECTION + OUTCOME COMBINATIONS:': 'small',
            'NETPTS1 STATISTICS:': 'small',
            'NETPTS2 STATISTICS:': 'small',
            'POINT-BY-POINT NARRATIVE:': 'narrative'
        }
        
        current_section = None
        current_text = []
        current_strategy = None
        
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line starts a new section
            section_found = None
            strategy_found = None
            
            for pattern, strategy in section_config.items():
                if line_stripped.startswith(pattern):
                    section_found = pattern.rstrip(':').lower().replace(' ', '_').replace('(', '').replace(')', '')
                    strategy_found = strategy
                    break
            
            if section_found:
                # Process previous section with its strategy
                if current_section and current_text:
                    self._process_section_with_strategy(sections, current_section, current_text, current_strategy)
                
                # Start new section
                current_section = section_found
                current_strategy = strategy_found
                current_text = [line]
            else:
                current_text.append(line)
        
        # Process the last section
        if current_section and current_text:
            self._process_section_with_strategy(sections, current_section, current_text, current_strategy)
        
        print(f"Created {len(sections)} optimized chunks using semantic boundaries")
        return sections
    
    def _process_section_with_strategy(self, sections: Dict[str, str], section_name: str, 
                                     lines: List[str], strategy: str):
        """Process a section according to its chunking strategy."""
        content = '\n'.join(lines)
        
        if strategy == 'small':
            # Keep small sections intact (MATCH OVERVIEW, NETPTS, etc.)
            sections[section_name] = content
            
        elif strategy == 'medium':
            # Split medium sections by player if beneficial
            player_chunks = self._split_by_player_if_beneficial(content, section_name)
            if len(player_chunks) > 1:
                for i, chunk in enumerate(player_chunks):
                    sections[f"{section_name}_player_{i+1}"] = chunk
            else:
                sections[section_name] = content
                
        elif strategy == 'large':
            # Split large detailed sections by logical subsections
            subsection_chunks = self._split_large_section_intelligently(content, section_name)
            if len(subsection_chunks) > 1:
                for i, chunk in enumerate(subsection_chunks):
                    sections[f"{section_name}_part_{i+1}"] = chunk
            else:
                sections[section_name] = content
                
        elif strategy == 'narrative':
            # Split point-by-point by game groups (every 12-15 points)
            point_chunks = self._split_narrative_by_games(content)
            for i, chunk in enumerate(point_chunks):
                sections[f"{section_name}_games_{i+1}"] = chunk
            
        elif strategy == 'return_split':
            # Custom strategy for return statistics: split by outcomes and depth
            return_chunks = self._split_return_statistics_by_outcomes_and_depth(content, section_name)
            if len(return_chunks) > 1:
                for i, chunk in enumerate(return_chunks):
                    if i == 0:
                        sections[f"{section_name}_outcomes"] = chunk
                    else:
                        sections[f"{section_name}_depth"] = chunk
            else:
                sections[section_name] = content
    
    def _split_return_statistics_by_outcomes_and_depth(self, content: str, section_name: str) -> List[str]:
        """Split return statistics into outcomes and depth sections."""
        lines = content.split('\n')
        outcomes_section = []
        depth_section = []
        current_section = None
        
        for line in lines:
            # Check for depth section indicators
            if any(phrase in line.lower() for phrase in [
                'shallow returns', 'deep returns', 'very deep returns', 
                'unforced errors when returning', 'net approaches when returning'
            ]):
                current_section = 'depth'
            
            # Add line to appropriate section
            if current_section == 'depth':
                depth_section.append(line)
            else:
                outcomes_section.append(line)
        
        chunks = []
        if outcomes_section:
            chunks.append('\n'.join(outcomes_section))
        if depth_section:
            chunks.append('\n'.join(depth_section))
        
        return chunks if len(chunks) > 1 else [content]
    
    def _split_by_player_if_beneficial(self, content: str, section_name: str, player1: str = None, player2: str = None) -> List[str]:
        """Split content by player if it creates meaningful chunks."""
        lines = content.split('\n')
        player1_lines = []
        player2_lines = []
        header_lines = []
        
        # For key points sections, we need to include the authoritative totals
        authoritative_totals = []
        if 'key_points_statistics' in section_name:
            # Look for authoritative totals in the content
            content_lines = content.split('\n')
            in_authoritative_section = False
            for line in content_lines:
                if 'AUTHORITATIVE TOTALS FOR KEY POINTS:' in line:
                    in_authoritative_section = True
                    authoritative_totals.append(line)
                elif in_authoritative_section and line.strip() == '':
                    # End of authoritative section
                    break
                elif in_authoritative_section:
                    authoritative_totals.append(line)
        
        for line in lines:
            if player1 and player1 in line and not line.strip().endswith(':'):
                player1_lines.append(line)
            elif player2 and player2 in line and not line.strip().endswith(':'):
                player2_lines.append(line)
            elif line.strip().endswith(':') or line.startswith('---') or line.startswith('='):
                header_lines.append(line)
            else:
                # Neutral lines go to both if we're splitting, otherwise to header
                if player1_lines or player2_lines:
                    continue
                header_lines.append(line)
        
        # Only split if both players have significant content
        if len(player1_lines) >= 3 and len(player2_lines) >= 3:
            chunks = []
            if player1_lines:
                # Include authoritative totals at the beginning for key points sections
                if authoritative_totals:
                    chunks.append('\n'.join(authoritative_totals + [''] + header_lines + player1_lines))
                else:
                    chunks.append('\n'.join(header_lines + player1_lines))
            if player2_lines:
                # Include authoritative totals at the beginning for key points sections
                if authoritative_totals:
                    chunks.append('\n'.join(authoritative_totals + [''] + header_lines + player2_lines))
                else:
                    chunks.append('\n'.join(header_lines + player2_lines))
            return chunks
            
        return [content]  # Don't split if not beneficial
    
    def _split_large_section_intelligently(self, content: str, section_name: str) -> List[str]:
        """Split large sections by natural semantic boundaries."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        
        # Identify natural break points
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Look for natural breaks in detailed statistics
            is_break_point = (
                len(current_chunk) > 40 and  # Minimum chunk size
                (line.strip().endswith('court.') or 
                 line.strip().endswith('serves.') or
                 line.strip().endswith('returns.') or
                 (i < len(lines) - 1 and lines[i+1].strip() == '') or  # Empty line follows
                 ('served to the' in line and 'court' in line and 
                  i > 0 and 'served to the' not in lines[i-1]))
            )
            
            if is_break_point:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        # Handle remaining content
        if current_chunk:
            if chunks and len(current_chunk) < 20:
                # Append small remainder to last chunk
                chunks[-1] += '\n' + '\n'.join(current_chunk)
            else:
                chunks.append('\n'.join(current_chunk))
        
        return chunks if len(chunks) > 1 else [content]
    
    def _split_narrative_by_games(self, content: str) -> List[str]:
        """Split point-by-point narrative by game groups for optimal chunk size."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        point_count = 0
        
        for line in lines:
            current_chunk.append(line)
            
            # Count actual points
            if line.startswith('Point '):
                point_count += 1
                
                # Create new chunk every 20 points (more comprehensive coverage)
                # This ensures longer rallies don't get split across chunks
                if point_count % 20 == 0 and len(current_chunk) > 10:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
        
        # Handle remaining points
        if current_chunk:
            if chunks and len(current_chunk) < 8:
                # Append small remainder to last chunk
                chunks[-1] += '\n' + '\n'.join(current_chunk)
            else:
                chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [content]
    
    def _extract_set_numbers_from_chunk(self, chunk_text: str) -> List[int]:
        """
        Extract which set(s) this chunk contains based on score patterns.
        Returns list of set numbers (e.g., [2, 3] for a mixed chunk).
        """
        import re
        
        if not hasattr(self, 'set_mapping') or not self.set_mapping:
            return []
        
        set_numbers = set()
        
        # Find all "Score: X-Y" patterns in chunk
        score_matches = re.findall(r'Score: (\d+)-(\d+)', chunk_text)
        
        for server_sets, returner_sets in score_matches:
            set_score = f"{server_sets}-{returner_sets}"
            
            # Reverse lookup: which set has this score pattern?
            for set_num, expected_score in self.set_mapping.items():
                # Check both the expected score and its reverse (server perspective flip)
                score_parts = expected_score.split('-')
                reversed_score = f"{score_parts[1]}-{score_parts[0]}" if len(score_parts) == 2 else expected_score
                
                if set_score == expected_score or set_score == reversed_score:
                    set_numbers.add(set_num)
        
        return sorted(list(set_numbers))
    
    def _extract_game_scores_from_chunk(self, chunk_text: str) -> List[str]:
        """
        Extract which game scores appear in this chunk.
        Returns list of game scores (e.g., ['0-0', '1-0', '2-1']).
        """
        import re
        
        game_scores = set()
        
        # Find all "Score: X-Y G1-G2" patterns in chunk
        # Format: "Score: 2-0 4-5 0-15" where 4-5 is the game score
        score_matches = re.findall(r'Score: \d+-\d+ (\d+-\d+)', chunk_text)
        
        for game_score in score_matches:
            game_scores.add(game_score)
        
        return sorted(list(game_scores))
    
    def _transform_point_by_point_in_content(self, content: str) -> str:
        """
        Find and transform the point-by-point section in the full content.
        Adds player attribution to each shot in rallies.
        """
        import re
        
        # Find the point-by-point section
        pbp_start = content.find('POINT-BY-POINT NARRATIVE:')
        if pbp_start == -1:
            return content  # No PBP section found
        
        # Find where it ends (look for next major section or end of file)
        # PBP is typically the last section, but check anyway
        pbp_end = len(content)
        for next_section in ['RALLY OUTCOMES STATISTICS:', 'OTHER DATA:']:
            section_pos = content.find(next_section, pbp_start + 1)
            if section_pos != -1 and section_pos < pbp_end:
                pbp_end = section_pos
        
        # Extract, transform, and replace
        raw_pbp = content[pbp_start:pbp_end]
        print("[TRANSFORM] Adding shot attribution to point-by-point data...")
        transformed_pbp = self._transform_point_by_point_data(raw_pbp)
        
        return content[:pbp_start] + transformed_pbp + content[pbp_end:]
    
    def _transform_point_by_point_data(self, raw_pbp_text: str) -> str:
        """
        Transform raw PBP notation into explicit, LLM-friendly format with shot attribution.
        """
        import re
        
        # Keep the header
        result = "POINT-BY-POINT NARRATIVE:\n"
        result += "-" * 30 + "\n\n"
        
        # Find all points
        points = re.findall(r'(Point \d+.*?)(?=Point \d+|$)', raw_pbp_text, re.DOTALL)
        
        transformed_count = 0
        for point_text in points:
            if not point_text.strip():
                continue
            transformed = self._transform_single_point(point_text.strip())
            result += transformed + "\n\n"
            transformed_count += 1
        
        return result
    
    def _transform_single_point(self, point_text: str) -> str:
        """Transform a single point into explicit format with shot attribution."""
        import re
        
        # Parse point header
        header_match = re.search(
            r'Point (\d+) \[Server: (.*?) \| Returner: (.*?) \| Score: ([\d-]+) ([\d-]+) ([\d-]+)\]:',
            point_text
        )
        
        if not header_match:
            return point_text  # Fallback to original
        
        point_num = header_match.group(1)
        server = header_match.group(2).strip()
        returner = header_match.group(3).strip()
        set_score = header_match.group(4)
        game_score = header_match.group(5)
        point_score = header_match.group(6)
        
        # CRITICAL: Validate and fix server == returner bug using rally structure
        # SEMICOLONS INDICATE A CHANGE IN HITTER
        rally_text = point_text[header_match.end():].strip()
        
        # Check if server == returner (this is the bug!)
        if server.strip().lower() == returner.strip().lower() and self.player1 and self.player2:
            # Use rally structure (semicolons = hitter alternation) to determine correct returner
            shots = [s.strip() for s in rally_text.split(';') if s.strip()]
            
            if len(shots) >= 2:
                # Rally structure: Shot 0 = server, Shot 1 = returner, Shot 2 = server, etc.
                # First shot should be serve (by server), second shot is return (by returner)
                first_shot = shots[0].lower()
                is_serve = '1st serve' in first_shot or '2nd serve' in first_shot or 'serve' in first_shot
                
                if is_serve:
                    # Determine which player the server is by matching server name
                    server_lower = server.lower().strip()
                    p1_lower = self.player1.lower().strip()
                    p2_lower = self.player2.lower().strip()
                    
                    # Use fuzzy matching to determine server
                    if server_lower == p1_lower or (p1_lower in server_lower or server_lower in p1_lower):
                        # Server is player1, returner is player2
                        returner = self.player2
                    elif server_lower == p2_lower or (p2_lower in server_lower or server_lower in p2_lower):
                        # Server is player2, returner is player1
                        returner = self.player1
                    else:
                        # Can't match - use context from player names
                        # If server name is closer to player1, returner is player2
                        sim1 = 1.0 if server_lower == p1_lower else (0.9 if p1_lower in server_lower or server_lower in p1_lower else 0.0)
                        sim2 = 1.0 if server_lower == p2_lower else (0.9 if p2_lower in server_lower or server_lower in p2_lower else 0.0)
                        returner = self.player2 if sim1 > sim2 else self.player1
                    
                else:
                    # First shot is not a serve - can't infer from structure, use best guess
                    server_lower = server.lower().strip()
                    p1_lower = self.player1.lower().strip()
                    p2_lower = self.player2.lower().strip()
                    sim1 = 1.0 if server_lower == p1_lower else (0.9 if p1_lower in server_lower or server_lower in p1_lower else 0.0)
                    sim2 = 1.0 if server_lower == p2_lower else (0.9 if p2_lower in server_lower or server_lower in p2_lower else 0.0)
                    returner = self.player2 if sim1 > sim2 else self.player1
            else:
                # Single shot (ace/service winner) - CRITICAL: Server ALWAYS hits the shot
                # Since server is correct, returner MUST be the other player
                server_lower = server.lower().strip()
                p1_lower = self.player1.lower().strip()
                p2_lower = self.player2.lower().strip()
                
                # Match server to determine which player it is, then returner is the other
                if server_lower == p1_lower or (p1_lower in server_lower or server_lower in p1_lower):
                    # Server is player1, returner is player2
                    returner = self.player2
                elif server_lower == p2_lower or (p2_lower in server_lower or server_lower in p2_lower):
                    # Server is player2, returner is player1
                    returner = self.player1
                else:
                    # Can't match - use similarity as fallback
                    sim1 = 1.0 if server_lower == p1_lower else (0.9 if p1_lower in server_lower or server_lower in p1_lower else 0.0)
                    sim2 = 1.0 if server_lower == p2_lower else (0.9 if p2_lower in server_lower or server_lower in p2_lower else 0.0)
                    returner = self.player2 if sim1 > sim2 else self.player1
                
        
        # Parse rally with corrected returner
        rally_shots = self._parse_rally_sequence(rally_text, server, returner)
        
        # Determine point winner
        point_winner = None
        if rally_shots:
            # Check for double fault (2nd serve fault)
            for shot in rally_shots:
                if '2nd serve' in shot['description'] and shot.get('outcome') == 'FAULT':
                    # Double fault - returner wins
                    point_winner = returner
                    break
            
            # If not double fault, find the last shot with an outcome
            if not point_winner:
                for shot in reversed(rally_shots):
                    outcome = shot.get('outcome')
                    if outcome and outcome != 'FAULT':
                        # Winner or error determines point winner
                        if outcome == 'WINNER':
                            point_winner = shot['player']
                        elif outcome in ['UNFORCED ERROR', 'FORCED ERROR']:
                            # The player who made the error LOST
                            point_winner = returner if shot['player'] == server else server
                        else:
                            # Normal shot
                            point_winner = shot['player']
                        break
        
        # Build the transformed point text
        result = f"Point {point_num} [Server: {server} | Returner: {returner} | Score: {set_score} {game_score} {point_score}]:\n"
        
        # Add rally with attribution
        rally_description = []
        for shot in rally_shots:
            shot_desc = f"{shot['description']} [{shot['player']}]"
            rally_description.append(shot_desc)
        
        result += "; ".join(rally_description) + "."
        
        # Add point winner
        if point_winner:
            result += f" [Point won by: {point_winner}]"
        
        return result
    
    def _parse_rally_sequence(self, rally_text: str, server: str, returner: str) -> List[Dict]:
        """
        Parse rally into alternating sequence of shots with player attribution.
        Returns list of {player, description, outcome} dicts.
        """
        import re
        
        shots = []
        current_player = server  # Server always starts
        in_rally = False  # Track if we're in rally (after successful serve)
        
        # CRITICAL: Handle "1st serve fault. 2nd serve" pattern
        # Replace period-space between serves with marker for proper splitting
        rally_text = re.sub(r'fault \([^)]+\)\.\s+(2nd serve)', r'fault. SERVE_SEPARATOR \1', rally_text)
        
        # Split by semicolons (each shot)
        shot_parts = rally_text.split(';')
        
        # Also split by SERVE_SEPARATOR marker
        expanded_parts = []
        for part in shot_parts:
            if 'SERVE_SEPARATOR' in part:
                sub_parts = part.split('SERVE_SEPARATOR')
                expanded_parts.extend(sub_parts)
            else:
                expanded_parts.append(part)
        shot_parts = expanded_parts
        
        for shot_part in shot_parts:
            shot_part = shot_part.strip()
            if not shot_part:
                continue
            
            # Clean up
            shot_clean = shot_part.strip()
            
            # Check for outcomes
            outcome = None
            if 'winner' in shot_clean.lower():
                outcome = 'WINNER'
            elif 'unforced error' in shot_clean.lower():
                outcome = 'UNFORCED ERROR'
            elif 'forced error' in shot_clean.lower():
                outcome = 'FORCED ERROR'
            elif 'fault' in shot_clean.lower():
                outcome = 'FAULT'
            
            # Handle serves
            if '1st serve' in shot_clean or '2nd serve' in shot_clean:
                # Serve is always by the server
                shots.append({
                    'player': server,  # Always server, not current_player
                    'description': shot_clean,
                    'outcome': outcome
                })
                # If successful serve, next shot is returner's
                if outcome != 'FAULT':
                    current_player = returner
                    in_rally = True
                else:
                    # Fault - next serve attempt, stay on server
                    current_player = server
                    in_rally = False
            else:
                # Regular rally shot
                shots.append({
                    'player': current_player,
                    'description': shot_clean,
                    'outcome': outcome
                })
                # Alternate for next shot
                current_player = returner if current_player == server else server
        
        return shots
    
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[str]:
        """
        Splits a long text into chunks small enough to feed an LLM.
        Uses simple token count approximation (1 token ≈ 4 chars).
        """
        approx_chunk_size = max_tokens * 4  # 4 chars per token
        chunks = []
        start = 0
        while start < len(text):
            end = start + approx_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        return chunks
    
    def _create_chunks_with_metadata(self, text: str, chunk_type: str, section: str, match_id: str) -> List[Dict]:
        """
        Returns a list of dicts with chunk text and enhanced metadata for better searchability.
        """
        # For tennis data, we want to keep sections intact rather than splitting arbitrarily
        # Only split if the section exceeds embedding model limits (8k tokens)
        estimated_tokens = len(text) // 4  # Rough approximation
        
        if estimated_tokens > 6000:  # More conservative buffer for 8192 token limit
            # Split large sections intelligently
            chunks = self._smart_chunk_large_section(text, section)
        else:
            # Keep section intact
            chunks = [text]
        
        # Determine player focus and statistics type from section
        player_focus = self._determine_player_focus(section)
        stat_category = self._determine_stat_category(section)
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract set and game metadata for point-by-point chunks
            set_numbers = []
            game_scores = []
            if 'point-by-point' in section.lower():
                set_numbers = self._extract_set_numbers_from_chunk(chunk)
                game_scores = self._extract_game_scores_from_chunk(chunk)
            
            result_chunks.append({
                "text": chunk,
                "metadata": {
                    "type": chunk_type,
                    "section": section,
                    "match_id": match_id,
                    "player_focus": player_focus,
                    "stat_category": stat_category,
                    "set_numbers": set_numbers,  # NEW: Which sets are in this chunk (e.g., [2, 3])
                    "game_scores": game_scores,  # NEW: Which game scores appear (e.g., ['4-5', '5-5', '6-5'])
                    "contains_percentages": "%" in chunk,
                    "contains_point_details": "Point " in chunk,
                    "contains_long_rallies": any("shot rally" in line or "stroke rally" in line for line in chunk.split('\n') if "rally" in line),
                    "match_info": {
                        "date": "2025-06-28",
                        "tournament": "Bad Homburg F",
                        "players": [self.player1 if self.player1 else "Player 1", self.player2 if self.player2 else "Player 2"],
                        "match_type": "WTA"
                    }
                }
            })
        
        return result_chunks
    
    def _smart_chunk_large_section(self, text: str, section: str) -> List[str]:
        """
        Intelligently chunk large sections while preserving semantic meaning.
        """
        lines = text.split('\n')
        
        if "point-by-point" in section.lower():
            # For point-by-point, group by games (every 6-8 points)
            chunks = []
            current_chunk = []
            point_count = 0
            
            for line in lines:
                current_chunk.append(line)
                if line.startswith("Point "):
                    point_count += 1
                    if point_count >= 6:  # Start new chunk every 6 points
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        point_count = 0
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
        
        else:
            # For statistics sections, try to split by player if possible
            player1_lines = []
            player2_lines = []
            header_lines = []
            other_lines = []
            
            for line in lines:
                if self.player1 and self.player1 in line:
                    player1_lines.append(line)
                elif self.player2 and self.player2 in line:
                    player2_lines.append(line)
                elif line.strip().endswith(':') or line.startswith('---'):
                    header_lines.append(line)
                else:
                    other_lines.append(line)
            
            # If we can separate by players, create player-specific chunks
            if player1_lines or player2_lines:
                chunks = []
                if player1_lines:
                    chunks.append('\n'.join(header_lines + player1_lines))
                if player2_lines:
                    chunks.append('\n'.join(header_lines + player2_lines))
                if other_lines and not (player1_lines or player2_lines):
                    chunks.append('\n'.join(header_lines + other_lines))
                return chunks
            
            # Fallback: use precise token-based chunking
            return self._precise_token_chunk(text, max_tokens=6000)
    
    def _precise_token_chunk(self, text: str, max_tokens: int) -> List[str]:
        """Precisely chunk text using actual token counting."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            total_tokens = len(encoding.encode(text))
            if total_tokens <= max_tokens:
                return [text]
            
            
            lines = text.split('\n')
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for line in lines:
                line_tokens = len(encoding.encode(line + '\n'))
                
                if current_tokens + line_tokens > max_tokens and current_chunk:
                    # Start new chunk
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_tokens = line_tokens
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            # Fallback to character-based chunking
            max_chars = max_tokens * 4  # Rough estimate
            chunks = []
            for i in range(0, len(text), max_chars):
                chunks.append(text[i:i + max_chars])
            return chunks
    
    def _apply_rate_limit(self):
        """Apply rate limiting to avoid hitting API limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_delay:
            sleep_time = self.min_delay - time_since_last_call
            print(f"[WAIT] Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def _determine_player_focus(self, section: str) -> str:
        """Determine which player this section focuses on."""
        if "serve1" in section.lower() or "return1" in section.lower() or "shots1" in section.lower() or "shotdir1" in section.lower() or "netpts1" in section.lower():
            return self.player1 if self.player1 else "Player 1"
        elif "serve2" in section.lower() or "return2" in section.lower() or "shots2" in section.lower() or "shotdir2" in section.lower() or "netpts2" in section.lower():
            return self.player2 if self.player2 else "Player 2"
        else:
            return "Both"
    
    def _determine_stat_category(self, section: str) -> str:
        """Determine the category of statistics."""
        section_lower = section.lower()
        if "serve" in section_lower:
            return "serving"
        elif "return" in section_lower:
            return "returning"
        elif "shot" in section_lower:
            return "shots"
        elif "net" in section_lower:
            return "net_play"
        elif "key_points" in section_lower:
            return "key_points"
        elif "rally" in section_lower:
            return "rally_outcomes"
        elif "point-by-point" in section_lower:
            return "narrative"
        elif "overview" in section_lower:
            return "overview"
        else:
            return "general"
    
    def _determine_chunk_type(self, section_name: str, section_text: str) -> str:
        """Determine the type of chunk based on section name and content."""
        section_lower = section_name.lower()
        
        if "point-by-point" in section_lower or "Point " in section_text:
            return "narrative"
        elif any(keyword in section_lower for keyword in ["statistics", "serve", "return", "shots", "net"]):
            return "statistics"
        elif "overview" in section_lower:
            return "overview"
        else:
            return "general"
    
    def _detect_long_rallies(self, text: str) -> List[str]:
        """Detect and extract information about long rallies in the text."""
        long_rallies = []
        lines = text.split('\n')
        
        for line in lines:
            if "shot rally" in line or "stroke rally" in line:
                # Extract rally length and details
                if "16-shot rally" in line:
                    long_rallies.append(f"LONGEST RALLY: {line}")
                elif "10+ shot rally" in line or "11-shot rally" in line or "12-shot rally" in line or "13-shot rally" in line or "14-shot rally" in line or "15-shot rally" in line:
                    long_rallies.append(f"LONG RALLY: {line}")
        
        return long_rallies
    
    def _is_match_insight_question(self, query: str) -> bool:
        """Determine if a query is asking for match insights vs. statistics."""
        query_lower = query.lower()
        
        # Match insight keywords
        insight_keywords = [
            "key moments", "decided", "outcome", "strategy", "momentum", 
            "critical", "turning point", "what happened", "how did", 
            "why did", "analyze", "explain", "describe the match",
            "tactical", "pattern", "trend", "shift", "flow"
        ]
        
        # Statistical keywords (strong indicators)
        stat_keywords = [
            "how many", "percentage", "total", "count", "statistics",
            "breakdown", "compare", "numbers", "figures", "aces", 
            "double faults", "winners", "errors", "first serve", "second serve"
        ]
        
        insight_score = sum(1 for keyword in insight_keywords if keyword in query_lower)
        stat_score = sum(1 for keyword in stat_keywords if keyword in query_lower)
        
        # If it's clearly asking for numbers, it's stats
        if stat_score > 0:
            return False
        
        return insight_score > 0
    
    def _embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for chunks using LOCAL sentence-transformers (100% FREE).
        No API calls, runs entirely on your computer!
        """
        print("[EMBED] Generating embeddings with LOCAL model (FREE - no API calls!)...")
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings locally
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
        
        print(f"Successfully generated {len(chunks)} embeddings locally (384 dimensions)")
        
        return chunks
    
    def _create_vector_index(self) -> None:
        """
        Create FAISS index and store metadata.
        """
        if not self.chunks:
            raise ValueError("No chunks available. Call load_exact_full_format() first.")
        
        # Get embedding dimension
        dim = len(self.chunks[0]["embedding"])
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(dim)
        
        # Convert embeddings to numpy
        vectors = np.array([c["embedding"] for c in self.chunks]).astype('float32')
        self.index.add(vectors)
        
        # Keep a parallel list of metadata for retrieval
        self.metadata_store = [c["metadata"] for c in self.chunks]
        
        print(f"Created FAISS index with {len(self.chunks)} vectors of dimension {dim}")
    
    def save_embeddings_to_disk(self, filename_prefix: str = "tennis_embeddings") -> None:
        """
        Save embeddings, FAISS index, and metadata to disk for persistence.
        """
        if not self.chunks or not self.index:
            raise ValueError("No embeddings available. Call load_exact_full_format() first.")
        
        # Save FAISS index
        faiss_filename = f"{filename_prefix}_faiss.pkl"
        with open(faiss_filename, 'wb') as f:
            pickle.dump(self.index, f)
        
        # Save metadata store (including match score and set mapping)
        metadata_filename = f"{filename_prefix}_metadata.pkl"
        metadata_bundle = {
            'metadata_store': self.metadata_store,
            'match_score': getattr(self, 'match_score', None),
            'set_mapping': getattr(self, 'set_mapping', {}),
            'total_sets': getattr(self, 'total_sets', 0)
        }
        with open(metadata_filename, 'wb') as f:
            pickle.dump(metadata_bundle, f)
        
        # Save chunks (for debugging/inspection)
        chunks_filename = f"{filename_prefix}_chunks.pkl"
        with open(chunks_filename, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved embeddings to disk:")
        print(f"   FAISS index: {faiss_filename}")
        print(f"   Metadata: {metadata_filename}")
        print(f"   Chunks: {chunks_filename}")
    
    def load_embeddings_from_disk(self, filename_prefix: str = "tennis_embeddings") -> bool:
        """
        Load embeddings, FAISS index, and metadata from disk.
        Returns True if successful, False if files don't exist.
        """
        try:
            # Extract player names from filename_prefix (e.g., "Jannik_Sinner_Carlos_Alcaraz_20250608")
            parts = filename_prefix.split('_')
            if len(parts) >= 4:
                # Find the date (8 digits at the end)
                date_idx = -1
                for i in range(len(parts) - 1, -1, -1):
                    if parts[i].isdigit() and len(parts[i]) == 8:
                        date_idx = i
                        break
                
                if date_idx >= 2:
                    # Everything before the date is player names
                    # Find the split point (usually middle, but handle multi-word names)
                    player_parts = parts[:date_idx]
                    # Simple heuristic: split roughly in middle
                    mid = len(player_parts) // 2
                    self.player1 = ' '.join(player_parts[:mid])
                    self.player2 = ' '.join(player_parts[mid:])
                    print(f"[CACHE] Extracted player names: {self.player1} vs {self.player2}")
            
            # Load FAISS index
            faiss_filename = f"{filename_prefix}_faiss.pkl"
            with open(faiss_filename, 'rb') as f:
                self.index = pickle.load(f)
            
            # Load metadata store and match info
            metadata_filename = f"{filename_prefix}_metadata.pkl"
            with open(metadata_filename, 'rb') as f:
                metadata_bundle = pickle.load(f)
                
                # Handle both old format (just metadata_store) and new format (bundle)
                if isinstance(metadata_bundle, dict) and 'metadata_store' in metadata_bundle:
                    # New format with match score and set mapping
                    self.metadata_store = metadata_bundle['metadata_store']
                    self.match_score = metadata_bundle.get('match_score', None)
                    self.set_mapping = metadata_bundle.get('set_mapping', {})
                    self.total_sets = metadata_bundle.get('total_sets', 0)
                    if self.match_score:
                        print(f"[CACHE] Loaded match score: {self.match_score}")
                    if self.set_mapping:
                        print(f"[CACHE] Loaded set mapping with {len(self.set_mapping)} entries")
                else:
                    # Old format - just metadata_store
                    self.metadata_store = metadata_bundle
                    self.match_score = None
                    self.set_mapping = {}
                    self.total_sets = 0
            
            # Load chunks (optional, for debugging)
            chunks_filename = f"{filename_prefix}_chunks.pkl"
            if os.path.exists(chunks_filename):
                with open(chunks_filename, 'rb') as f:
                    self.chunks = pickle.load(f)
            else:
                # Reconstruct chunks from metadata if needed
                self.chunks = [{"metadata": meta} for meta in self.metadata_store]
            
            print(f"Loaded embeddings from disk:")
            print(f"   FAISS index: {faiss_filename}")
            print(f"   Metadata: {metadata_filename}")
            print(f"   Total chunks: {len(self.metadata_store)}")
            
            return True
            
        except FileNotFoundError:
            print(f"[ERROR] Embedding files not found. Run load_exact_full_format() first.")
            return False
        except Exception as e:
            print(f"[ERROR] Error loading embeddings: {e}")
            return False
    
    def _detect_player_mentioned(self, query: str) -> str:
        """
        Detect which player is mentioned in the query.
        Returns the player name or None if no specific player is mentioned.
        """
        if not self.player1 or not self.player2:
            return None
            
        query_lower = query.lower()
        player1_lower = self.player1.lower()
        player2_lower = self.player2.lower()
        
        # Check for player1 (first player)
        if any(word in query_lower for word in player1_lower.split()):
            return self.player1
        
        # Check for player2 (second player)  
        if any(word in query_lower for word in player2_lower.split()):
            return self.player2
        
        return None

    def _get_player_suffix(self, player_mentioned: str) -> str:
        """
        Get the player suffix (_player_1 or _player_2) based on which player is mentioned.
        """
        if not player_mentioned or not self.player1 or not self.player2:
            return "_player_1"  # Default to player 1
            
        if player_mentioned.lower() == self.player1.lower():
            return "_player_1"
        elif player_mentioned.lower() == self.player2.lower():
            return "_player_2"
        else:
            return "_player_1"  # Default fallback

    def _determine_optimal_chunk_count(self, query: str) -> int:
        """
        Determine optimal number of chunks based on question complexity.
        Returns the number of chunks needed for comprehensive analysis.
        
        4-tier system:
        - Ultra-high (18): Multi-factor analysis needing ALL point-by-point data
        - High (15): Complex tactical/pattern questions
        - Detailed (8): Specific context questions
        - Simple (5): Direct factual questions
        """
        query_lower = query.lower()
        
        # Ultra-high complexity (need ALL point-by-point chunks - ~18-19 total)
        # These are RARE, genuinely complex multi-factor questions
        ultra_high_indicators = [
            # Multi-factor match analysis
            "match flow", "match progression", "match dynamics",
            "how the match unfolded", "match evolution",
            "throughout the match", "over time", "entire match",
            # Complex shot sequences and patterns
            "shot sequences", "shot sequence", "rally patterns",
            "when hit a", "when they hit", "when he hit", "when she hit",
            "response to", "respond to", "responded to", "in response",
            # Adaptation and change detection (need before/after comparison)
            "adapt", "adapted", "adaptation",
            "adjust", "adjusted", "adjustment",
            "changed over",
            "counter", "countered", "neutralize", "neutralized",
            # Behavioral transitions
            "switched to", "started hitting more", "began favoring",
            "moved away from", "increased reliance on",
            "started doing", "stopped doing",
            "changed shot direction",
            # Sequential momentum analysis (specific patterns, not common words)
            "consecutive", "streak", "run of", "string of",
            # Rally length and patterns
            "long rallies", "short rallies", "rally length",
            "extended rallies", "quick points",
            "8+ shot", "5+ shot", "rallies longer", "rallies shorter",
            # Physical and performance variance
            "fatigue", "tired", "wore down",
            "variance", "highs and lows",
            "defensive to offensive", "pushed back", "stepped in",
            # Counterfactual analysis
            "what if", "if he had", "if she had",
            "could have", "should have", "would have", "prevented"
        ]
        
        # High complexity (need most point-by-point data)
        high_complexity_indicators = [
            # Match narrative and strategy
            "match pattern", "match analysis", "match strategy",
            "what was the pattern", "analyze the match",
            "turning point", "momentum", "match narrative", "match story",
            "match breakdown", "match summary", "match overview", "match recap",
            "progression", "progressed", "developed",
            "early vs late", "first set vs", "as the match", "match went on",
            # Evolution and shifts (moved from ultra_high - common but still complex)
            "evolve", "evolved", "evolution",
            "momentum shift", "momentum swing",
            "shift", "shifted",
            # Situational and conditional
            "after losing", "after winning", "after missing", "after break",
            "after", "following", "then", "led to", "triggered",
            "when rallies got", "as rallies got",
            "on important points", "on key points", "in crucial moments", "in tight moments",
            "when facing break", "when trailing", "when ahead",
            # Shot responses and effectiveness
            "most common response", "typical response", "usual response",
            "how did", "what did", "how would", "what would",
            "drop shot", "most effective", "least effective", "best response", "worst response",
            # Tactical patterns
            "serve placement", "shot selection", "shot placement",
            "approach shot", "net approach", "serve and volley", "chip and charge",
            "rally control", "dictate", "dictating", "control the point",
            "tactical", "tactics", "strategy", "strategic", "game plan",
            # Psychology and momentum
            "clutch", "pressure", "under pressure", "mental", "mentality",
            "champion", "fighting spirit", "resilience", "comeback",
            "statement", "deflate", "demoralize", "confidence",
            # Sequential patterns
            "next", "subsequent", "immediately after",
            "in response to", "as a result",
            # Comparative time periods
            "first half", "second half", "early", "late", "beginning", "end",
            "set 1 vs", "set 2 vs", "first two sets", "last three sets",
            "tiebreak vs", "regular games", "important games vs",
            # Pattern and tendency
            "pattern", "patterns", "tendency", "tendencies", "prefer", "preferred",
            "favor", "favored", "typically", "usually", "commonly", "often",
            "most of the time", "majority", "predominant", "characteristic",
            # Context-specific situations
            "set point", "game point", "match point",
            "deuce", "advantage", "30-30", "40-40",
            "must hold", "must break",
            "tiebreak", "tie-break", "deciding", "crucial game",
            # Outcome and effectiveness
            "effectiveness", "effective", "ineffective",
            "worked", "didn't work", "successful", "unsuccessful",
            "paid off", "backfired",
            # Quality and level
            "quality", "level", "standard", "performance level",
            "raised their game", "elevated", "dropped off", "declined",
            "peak", "best", "worst", "high point", "low point"
        ]
        
        # Specific but detailed questions
        detailed_indicators = [
            "break points", "game points", "key points", "pressure points",
            "shot direction", "shot types", "serve types", "return types",
            "net play", "baseline play", "point construction",
            "serve performance", "return performance", "overall performance"
        ]
        
        # Check complexity in order (most specific first)
        if any(indicator in query_lower for indicator in ultra_high_indicators):
            return 18  # Get ALL point-by-point chunks
        
        if any(indicator in query_lower for indicator in high_complexity_indicators):
            return 15  # Most point-by-point chunks
        
        if any(indicator in query_lower for indicator in detailed_indicators):
            return 8  # More context for detailed analysis
        
        # Simple specific questions
        return 5  # Default for simple questions

    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a given query with enhanced filtering.
        Uses adaptive chunk count based on question complexity.
        """
        # Determine optimal chunk count based on question complexity
        if top_k is None:
            top_k = self._determine_optimal_chunk_count(query)
        
        # BOOST chunk count for multi-set comparisons (need data from multiple sets)
        multiple_sets = self._detect_multiple_set_references(query)
        if multiple_sets and len(multiple_sets) >= 2:
            original_top_k = top_k
            top_k = 20  # Ensure we get enough chunks from all requested sets
        
        
        if not self.index:
            raise ValueError("Vector index not created. Call load_exact_full_format() first.")
        
        # Generate query embedding with LOCAL model (100% FREE!)
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search for similar chunks (get more than needed for filtering)
        search_k = min(top_k * 3, len(self.chunks))  # Get 3x more for filtering
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, search_k)
        
        # Collect candidates with metadata
        candidates = []
        for i in indices[0]:
            if i < len(self.chunks):
                chunk_info = {
                    "text": self.chunks[i]["text"],
                    "metadata": self.chunks[i]["metadata"],
                    "distance": float(distances[0][list(indices[0]).index(i)])
                }
                candidates.append(chunk_info)
        
        # Apply intelligent filtering based on query content
        filtered_chunks = self._filter_chunks_by_query(query, candidates)
        
        # Initialize fix flags FIRST (must be before any code path that uses them!)
        direction_outcome_fix_applied = False
        bp_gp_fix_applied = False
        universal_fix_applied = False
        net_points_fix_applied = False

        
        # MATCH OVERVIEW PRIORITY: For score/winner questions, always include match_overview
        query_lower = query.lower()
        if any(word in query_lower for word in ["score", "won", "winner", "result", "defeated", "beat", "lost", "final"]):
            match_overview_chunk = None
            for chunk in self.chunks:
                if 'match_overview' in chunk['metadata']['section']:
                    match_overview_chunk = {
                        "text": chunk['text'],
                        "metadata": chunk['metadata'],
                        "distance": 0.0,
                        "relevance_score": 10.0
                    }
                    break
            if match_overview_chunk:
                filtered_chunks = [chunk for chunk in filtered_chunks if 'match_overview' not in chunk['metadata']['section']]
                filtered_chunks.insert(0, match_overview_chunk)
        
        # OVERVIEW PRIORITY: For basic statistical questions, prioritize overview_statistics
        if not self._is_match_insight_question(query) and any(word in query.lower() for word in [
            # Winners
            "winner", "winners", "winner forehand", "winner backhand",
            # Errors
            "unforced error", "unforced errors", "unforced error forehand", "unforced error backhand",
            "forced error", "forced errors",
            # Serve statistics
            "ace", "aces", "ace percent", "ace percentage",
            "double fault", "double faults", "double fault percent", "double fault percentage",
            "first serve", "first serve in", "first serve won", "first serve percentage",
            "second serve", "second serve won", "second serve percentage",
            # Break points
            "break point", "break points", "break points saved", "break points converted",
            # Return points
            "return points", "return points won", "rpw", "rpw%",
            # General
            "total", "how many", "percentage", "percent"
        ]):
            # Find overview_statistics chunk
            overview_chunk = None
            for chunk in self.chunks:
                if 'overview_statistics' in chunk['metadata']['section']:
                    overview_chunk = {
                        "text": chunk['text'],
                        "metadata": chunk['metadata'],
                        "distance": 0.0,
                        "relevance_score": 10.0
                    }
                    break
            
            # Force overview to be the first chunk for basic stats questions
            if overview_chunk:
                # Remove any existing overview chunk
                filtered_chunks = [chunk for chunk in filtered_chunks if 'overview_statistics' not in chunk['metadata']['section']]
                # Add overview at the very top
                filtered_chunks.insert(0, overview_chunk)
        
        # CRITICAL FIX: Always include overview_statistics for statistical questions
        if not self._is_match_insight_question(query):
            # Find overview_statistics chunk
            overview_chunk = None
            for chunk in self.chunks:
                if 'overview_statistics' in chunk['metadata']['section']:
                    overview_chunk = {
                        "text": chunk['text'],
                        "metadata": chunk['metadata'],
                        "distance": 0.0,  # Perfect match
                        "relevance_score": 10.0  # Maximum score
                    }
                    break
            
            # If overview chunk exists and not already in results, add it at the top
            if overview_chunk and not any('overview_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                filtered_chunks.insert(0, overview_chunk)
            
            # DIRECTION + OUTCOME FIX: For questions about shots by direction AND outcome, force include all shot types for that direction/outcome combo
            direction_keywords = {
                "crosscourt": ["crosscourt", "cross court"],
                "down the line": ["down the line", "downline", "dtl"],
                "down the middle": ["down the middle", "middle", "center"],
                "inside-out": ["inside-out", "inside out"],
                "inside-in": ["inside-in", "inside in"]
            }
            
            outcome_keywords = {
                "winner": ["winner", "winners"],
                "forced error": ["forced error", "forced errors", "induced"],
                "unforced error": ["unforced error", "unforced errors"],
                "pt ending": ["pt ending", "point ending", "ended", "end points"],
                "pts won": ["pts won", "points won", "ptswon"],
                "pts lost": ["pts lost", "points lost", "ptslost"]
            }
            
            detected_direction = None
            detected_outcome = None
            
            # Check for direction + outcome combinations
            for direction, dir_keywords in direction_keywords.items():
                if any(keyword in query.lower() for keyword in dir_keywords):
                    detected_direction = direction
                    break
            
            for outcome, outcome_keywords_list in outcome_keywords.items():
                if any(keyword in query.lower() for keyword in outcome_keywords_list):
                    detected_outcome = outcome
                    break
            
            if detected_direction and detected_outcome:
                # Determine which player is being asked about
                player_mentioned = None
                if self.player1 and (self.player1.lower() in query.lower() or any(word.lower() in query.lower() for word in self.player1.lower().split())):
                    player_mentioned = self.player1
                    target_chunk_name = "shotdir1_statistics"
                elif self.player2 and (self.player2.lower() in query.lower() or any(word.lower() in query.lower() for word in self.player2.lower().split())):
                    player_mentioned = self.player2
                    target_chunk_name = "shotdir2_statistics"
                else:
                    # If no specific player mentioned, include both (for "each player" questions)
                    player_mentioned = "both"
                    target_chunk_name = None
                
                if player_mentioned == "both":
                    # Force include both players' shot direction chunks for "each player" questions
                    chunk_names = ["shotdir1_statistics", "shotdir2_statistics"]
                    chunk1_found = None
                    chunk2_found = None
                    explicit_totals_found = None
                    
                    # Find both chunks and explicit totals chunk
                    for chunk in self.chunks:
                        if "shotdir1_statistics" in chunk['metadata']['section']:
                            chunk1_found = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 9.5  # Very high score for specific direction/outcome queries
                            }
                        elif "shotdir2_statistics" in chunk['metadata']['section']:
                            chunk2_found = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 9.5  # Very high score for specific direction/outcome queries
                            }
                        elif "explicit_totals_for_shot_direction_+_outcome_combinations" in chunk['metadata']['section']:
                            explicit_totals_found = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 10.0  # Highest score for explicit totals
                            }
                    
                    # Add explicit totals chunk first (highest priority)
                    if explicit_totals_found:
                        # Remove any existing explicit totals chunk first
                        filtered_chunks = [chunk for chunk in filtered_chunks if "explicit_totals_for_shot_direction_+_outcome_combinations" not in chunk['metadata']['section']]
                        # Insert at the very beginning
                        filtered_chunks.insert(0, explicit_totals_found)
                    
                    # Add chunks if not already in results
                    if chunk1_found and not any("shotdir1_statistics" in chunk['metadata']['section'] for chunk in filtered_chunks):
                        filtered_chunks.insert(0, chunk1_found)
                    
                    if chunk2_found and not any("shotdir2_statistics" in chunk['metadata']['section'] for chunk in filtered_chunks):
                        filtered_chunks.insert(0, chunk2_found)
                else:
                    # Force include only the specific player's shot direction chunk and explicit totals
                    target_chunk = None
                    explicit_totals_found = None
                    
                    for chunk in self.chunks:
                        if target_chunk_name in chunk['metadata']['section']:
                            target_chunk = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 9.5  # Very high score for specific direction/outcome queries
                            }
                        elif "explicit_totals_for_shot_direction_+_outcome_combinations" in chunk['metadata']['section']:
                            explicit_totals_found = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 10.0  # Highest score for explicit totals
                            }
                    
                    # Add explicit totals chunk first (highest priority)
                    if explicit_totals_found:
                        # Remove any existing explicit totals chunk first
                        filtered_chunks = [chunk for chunk in filtered_chunks if "explicit_totals_for_shot_direction_+_outcome_combinations" not in chunk['metadata']['section']]
                        # Insert at the very beginning
                        filtered_chunks.insert(0, explicit_totals_found)
                    
                    # Add chunk if not already in results
                    if target_chunk and not any(target_chunk_name in chunk['metadata']['section'] for chunk in filtered_chunks):
                        filtered_chunks.insert(0, target_chunk)
                
                direction_outcome_fix_applied = True
            
            # BP/GP ROUTING FIX: For break point and game point questions, route to correct data sources
            if any(phrase in query.lower() for phrase in ["break point", "bp", "game point", "gp", "deuce"]) and "net points" not in query.lower():
                # Determine which player is being asked about
                player_mentioned = self._detect_player_mentioned(query)
                
                if player_mentioned:
                    # Route the question to get the correct section
                    target_section = self._route_bp_gp_question(query, player_mentioned)
                    
                    if target_section != "unknown" and target_section != "both":
                        # First, check if overview statistics has the data we need
                        overview_chunk = None
                        for chunk in self.chunks:
                            if 'overview_statistics' in chunk['metadata']['section']:
                                overview_chunk = {
                                    "text": chunk['text'],
                                    "metadata": chunk['metadata'],
                                    "distance": 0.0,  # Perfect match
                                    "relevance_score": 10.0  # Highest score for overview data
                                }
                                break
                        
                        # Add overview chunk if it has BP/GP data and not already in results
                        if overview_chunk and not any('overview_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                            filtered_chunks.insert(0, overview_chunk)
                            bp_gp_fix_applied = True
                        
                        # Also add the specific section chunk as backup
                        target_chunk = None
                        
                        # For break point, game point, and deuce questions, prioritize key points statistics
                        if "break point" in query.lower() or "bp" in query.lower() or "game point" in query.lower() or "gp" in query.lower() or "deuce" in query.lower():
                            # Look for key points statistics first
                            # Determine which key points section to look for based on question type
                            if "break point" in query.lower() or "bp" in query.lower():
                                # For break points, use returns data for conversions, serves data for faced/saved
                                if "convert" in query.lower() or "win" in query.lower() or "won" in query.lower():
                                    # Look for returning key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_returns' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                                else:
                                    # For other break point questions, look for serving key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_serves' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                            elif "game point" in query.lower() or "gp" in query.lower():
                                # For game points, use returns data for faced, serves data for won
                                if "face" in query.lower():
                                    # Look for returning key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_returns' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                                else:
                                    # For other game point questions, look for serving key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_serves' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                            elif "deuce" in query.lower():
                                # For deuce points, use returns data for return deuce, serves data for serve deuce
                                if "return" in query.lower():
                                    # Look for returning key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_returns' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                                else:
                                    # For other deuce questions, look for serving key points statistics
                                    for chunk in self.chunks:
                                        # Get player suffix dynamically
                                        player_suffix = self._get_player_suffix(player_mentioned)
                                        if 'key_points_statistics_serves' in chunk['metadata']['section'] and player_suffix in chunk['metadata']['section']:
                                            target_chunk = {
                                                "text": chunk['text'],
                                                "metadata": chunk['metadata'],
                                                "distance": 0.0,  # Perfect match
                                                "relevance_score": 9.5  # Higher score for key points data
                                            }
                                            break
                        
                        # If no key points chunk found, fall back to regular statistics
                        if not target_chunk:
                            for chunk in self.chunks:
                                # Handle split return statistics chunks (e.g., return2_statistics_detailed_part_1)
                                if target_section in chunk['metadata']['section'] or (target_section.replace('_statistics', '') in chunk['metadata']['section'] and 'return' in chunk['metadata']['section']):
                                    target_chunk = {
                                        "text": chunk['text'],
                                        "metadata": chunk['metadata'],
                                        "distance": 0.0,  # Perfect match
                                        "relevance_score": 9.0  # High score for BP/GP questions
                                    }
                                    break
                        
                        # Add the chunk if not already in results
                        if target_chunk:
                            # For key points chunks, use the actual chunk section name
                            if 'key_points_statistics' in target_chunk['metadata']['section']:
                                key_points_section = target_chunk['metadata']['section']
                                # Remove any existing instance first
                                filtered_chunks = [chunk for chunk in filtered_chunks if key_points_section not in chunk['metadata']['section']]
                                
                                # Set maximum priority and force to top
                                target_chunk['relevance_score'] = 10.0
                                filtered_chunks.insert(0, target_chunk)
                                bp_gp_fix_applied = True
                            else:
                                # For regular statistics chunks, use target_section
                                if not any(target_section in chunk['metadata']['section'] for chunk in filtered_chunks):
                                    filtered_chunks.insert(0, target_chunk)
                                    bp_gp_fix_applied = True
                    
                    elif target_section == "both":
                        # Need both serve and return sections for BP/GP/deuce totals or ambiguous questions
                        serve_section = "serve1_statistics_summary" if (self.player1 and (self.player1.lower() in query.lower() or any(word.lower() in query.lower() for word in self.player1.lower().split()))) else "serve2_statistics_summary"
                        return_section = "return1_statistics_detailed" if (self.player1 and (self.player1.lower() in query.lower() or any(word.lower() in query.lower() for word in self.player1.lower().split()))) else "return2_statistics_detailed"
                        
                        # Add both chunks
                        for section in [serve_section, return_section]:
                            target_chunk = None
                            for chunk in self.chunks:
                                if section in chunk['metadata']['section']:
                                    target_chunk = {
                                        "text": chunk['text'],
                                        "metadata": chunk['metadata'],
                                        "distance": 0.0,
                                        "relevance_score": 9.0
                                    }
                                    break
                            
                            if target_chunk and not any(section in chunk['metadata']['section'] for chunk in filtered_chunks):
                                filtered_chunks.insert(0, target_chunk)
                                bp_gp_fix_applied = True
            
            # DIRECTION TOTALS FIX: For questions asking about shot direction totals (e.g., "how many crosscourt shots")
            if any(direction in query.lower() for direction in ["crosscourt", "down the line", "down the middle", "inside-out", "inside-in"]) and not any(outcome in query.lower() for outcome in ["winner", "winners", "unforced error", "unforced errors", "forced error", "forced errors", "error", "errors"]):
                # Force include both shotdir chunks for comprehensive direction totals
                shotdir1_chunk = None
                shotdir2_chunk = None
                
                # Find shotdir1_statistics chunk
                for chunk in self.chunks:
                    if 'shotdir1_statistics' in chunk['metadata']['section']:
                        shotdir1_chunk = {
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,  # Perfect match
                            "relevance_score": 8.0  # High score
                        }
                        break
                
                # Find shotdir2_statistics chunk
                for chunk in self.chunks:
                    if 'shotdir2_statistics' in chunk['metadata']['section']:
                        shotdir2_chunk = {
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,  # Perfect match
                            "relevance_score": 8.0  # High score
                        }
                        break
                
                # Add shotdir1 chunk if not already in results
                if shotdir1_chunk and not any('shotdir1_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                    filtered_chunks.insert(0, shotdir1_chunk)
                
                # Add shotdir2 chunk if not already in results
                if shotdir2_chunk and not any('shotdir2_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                    filtered_chunks.insert(0, shotdir2_chunk)
            
            # UNIVERSAL FIX: For "each player" or "both players" questions, force include both players' chunks
            detected_stat_type = None  # Initialize outside the if block
            
            if any(phrase in query.lower() for phrase in ["each player", "both players", "both player"]):
                # Define all the split player statistics sections
                split_sections = {
                    "serve": ["serve1_statistics_summary", "serve2_statistics_summary", "serve1_statistics_detailed", "serve2_statistics_detailed"],
                    "shots": ["shots1_statistics", "shots2_statistics"],
                    "shotdir": ["shotdir1_statistics", "shotdir2_statistics"],
                    "return": ["return1_statistics_detailed", "return2_statistics_detailed"],
                    "netpts": ["netpts1_statistics", "netpts2_statistics"],
                    "keypoints": ["key_points_statistics_serves", "key_points_statistics_returns"]
                }
                
                # Check which statistic type the question is about (in order of specificity)
                
                # Check most specific keywords first to avoid conflicts
                if any(phrase in query.lower() for phrase in ["break point", "break points", "game point", "game points"]):
                    detected_stat_type = "keypoints"
                elif any(phrase in query.lower() for phrase in ["net points", "net approaches"]):
                    detected_stat_type = "netpts"
                elif any(phrase in query.lower() for phrase in ["volley", "volleys"]):
                    detected_stat_type = "shots"  # Volley is a shot type, not just net points
                elif any(phrase in query.lower() for phrase in ["overhead", "smash", "smashes"]):
                    detected_stat_type = "shots"  # Overhead/smash is a shot type
                elif any(phrase in query.lower() for phrase in ["crosscourt", "down the line", "down the middle", "inside-out", "inside-in"]):
                    detected_stat_type = "shotdir"
                elif any(phrase in query.lower() for phrase in ["winner", "winners", "unforced error", "unforced errors", "forehand", "backhand"]):
                    # Only set to shots if we haven't already detected a more specific type
                    if not detected_stat_type:
                        detected_stat_type = "shots"
                elif any(phrase in query.lower() for phrase in ["return", "returns", "returnable"]):
                    detected_stat_type = "return"
                elif any(phrase in query.lower() for phrase in ["ace", "aces", "serve", "serving", "double fault", "double faults", "first serve", "second serve"]):
                    detected_stat_type = "serve"
                
                # Apply the fix for the detected statistic type
                if detected_stat_type and detected_stat_type in split_sections:
                    chunk_names = split_sections[detected_stat_type]
                    
                    # Find all matching chunks (handle variable-length lists)
                    found_chunks = []
                    for chunk_name in chunk_names:
                        for chunk in self.chunks:
                            if chunk_name in chunk['metadata']['section']:
                                found_chunk = {
                                    "text": chunk['text'],
                                    "metadata": chunk['metadata'],
                                    "distance": 0.0,
                                    "relevance_score": 8.0
                                }
                                found_chunks.append(found_chunk)
                                break
                    
                    # Remove existing instances and add with high priority
                    for found_chunk in found_chunks:
                        chunk_section = found_chunk['metadata']['section']
                        # Remove any existing instance
                        filtered_chunks = [chunk for chunk in filtered_chunks if chunk['metadata']['section'] != chunk_section]
                        # Set high priority and add to top
                        found_chunk['relevance_score'] = 9.0
                        filtered_chunks.insert(0, found_chunk)
                    
                    universal_fix_applied = True
                    
                    # If we added netpts chunks, set the flag to prevent FINAL NEGATIVE FILTER from removing them
                    if detected_stat_type == "netpts":
                        net_points_fix_applied = True
            
                        # CRITICAL FIX: For ace/serve questions (only if other fixes didn't apply)
            if not universal_fix_applied and not direction_outcome_fix_applied and not bp_gp_fix_applied and any(word in query.lower() for word in ["serve", "serves", "ace", "aces", "serving", "double fault", "first serve", "second serve"]) and not any(word in query.lower() for word in ["return", "returning"]):
                serve1_summary_chunks = []
                serve2_summary_chunks = []
                serve1_detailed_chunks = []
                serve2_detailed_chunks = []
                
                # Find ALL serve1_statistics_summary chunks
                for chunk in self.chunks:
                    if 'serve1_statistics_summary' in chunk['metadata'].get('section', ''):
                        serve1_summary_chunks.append({
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,
                            "relevance_score": 8.0
                        })
                
                # Find ALL serve2_statistics_summary chunks
                for chunk in self.chunks:
                    if 'serve2_statistics_summary' in chunk['metadata'].get('section', ''):
                        serve2_summary_chunks.append({
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,
                            "relevance_score": 8.0
                        })
                
                # Find ALL serve1_statistics_detailed chunks
                for chunk in self.chunks:
                    if 'serve1_statistics_detailed' in chunk['metadata'].get('section', ''):
                        serve1_detailed_chunks.append({
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,
                            "relevance_score": 7.5  # Slightly lower than summary
                        })
                
                # Find ALL serve2_statistics_detailed chunks
                for chunk in self.chunks:
                    if 'serve2_statistics_detailed' in chunk['metadata'].get('section', ''):
                        serve2_detailed_chunks.append({
                            "text": chunk['text'],
                            "metadata": chunk['metadata'],
                            "distance": 0.0,
                            "relevance_score": 7.5
                        })
                
                # Add serve1 summary chunks to position 0 (remove if exists first)
                for chunk in serve1_summary_chunks:
                    filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk['metadata']['section']]
                    filtered_chunks.insert(0, chunk)
                
                # Add serve2 summary chunks to position 0 (remove if exists first)
                for chunk in serve2_summary_chunks:
                    filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk['metadata']['section']]
                    filtered_chunks.insert(0, chunk)
                
                # Add serve1 detailed chunks to position 0 (for detailed questions)
                for chunk in serve1_detailed_chunks:
                    filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk['metadata']['section']]
                    filtered_chunks.insert(0, chunk)
                
                # Add serve2 detailed chunks to position 0 (for detailed questions)
                for chunk in serve2_detailed_chunks:
                    filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk['metadata']['section']]
                    filtered_chunks.insert(0, chunk)
            
            # CRITICAL FIX: For return questions (return depth, return direction, etc.)
            if any(word in query.lower() for word in ["return", "returning", "deep return", "shallow return", "return depth"]):
                
                # Detect which player is mentioned
                player_mentioned = self._detect_player_mentioned(query)
                
                # Check if this is a serve-return question (e.g., "When X served, Y's return rate")
                is_serve_return_question = any(word in query.lower() for word in ["served", "serving", "serve to", "serve wide", "serve down"])
                
                # Only force the relevant player's chunks
                if player_mentioned:
                    # FLIP logic for serve-return questions: if asking about X's serve → need OTHER player's returns
                    if is_serve_return_question:
                        # If server is mentioned, get the OTHER player's return stats
                        if player_mentioned.lower() == self.player1.lower():
                            target_sections = "return2_statistics_detailed"  # FLIPPED: Get player2's returns
                        else:
                            target_sections = "return1_statistics_detailed"  # FLIPPED: Get player1's returns
                    else:
                        # Normal return question: player mentioned is the returner
                        if player_mentioned.lower() == self.player1.lower():
                            target_sections = "return1_statistics_detailed"
                        else:
                            target_sections = "return2_statistics_detailed"
                    
                    # Find and force the relevant player's chunks to position 0
                    chunks_to_add = []
                    for chunk in self.chunks:
                        if target_sections in chunk['metadata'].get('section', ''):
                            chunks_to_add.append({
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,
                                "relevance_score": 8.0
                            })
                    
                    # Remove these chunks if they already exist, then add them at position 0
                    for chunk_to_add in chunks_to_add:
                        filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk_to_add['metadata']['section']]
                        filtered_chunks.insert(0, chunk_to_add)
                else:
                    # If no player mentioned, force both players' chunks to position 0
                    chunks_to_add = []
                    for chunk in self.chunks:
                        if 'return1_statistics_detailed' in chunk['metadata'].get('section', '') or 'return2_statistics_detailed' in chunk['metadata'].get('section', ''):
                            chunks_to_add.append({
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,
                                "relevance_score": 8.0
                            })
                    
                    # Remove these chunks if they already exist, then add them at position 0
                    for chunk_to_add in chunks_to_add:
                        filtered_chunks = [c for c in filtered_chunks if c['metadata']['section'] != chunk_to_add['metadata']['section']]
                        filtered_chunks.insert(0, chunk_to_add)
            
            # CRITICAL FIX: For net points questions (always apply for net points questions)
            if any(word in query.lower() for word in ["net points", "net approaches", "approach shot", "approach shots", "net play", "at the net", "net points won", "net points lost", "net percentage", "net points won percentage", "net points percentage", "net"]):
                # Determine which player is being asked about
                player_mentioned = self._detect_player_mentioned(query)
                if player_mentioned:
                    if player_mentioned.lower() == self.player1.lower():
                        target_chunk_name = "netpts1_statistics"
                    else:
                        target_chunk_name = "netpts2_statistics"
                else:
                    # If no specific player mentioned, pull both chunks
                    player_mentioned = "both players"
                    target_chunk_name = None
                
                if target_chunk_name:
                    # Find the specific player's net points chunk
                    target_chunk = None
                    for chunk in self.chunks:
                        if target_chunk_name in chunk['metadata']['section']:
                            target_chunk = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 10.0  # Maximum score for net points
                            }
                            break
                    
                    # Add or move the specific player's chunk to position 0
                    if target_chunk:
                        # Remove it if it already exists (we'll re-insert at position 0)
                        filtered_chunks = [c for c in filtered_chunks if target_chunk_name not in c['metadata']['section']]
                        # Insert at position 0
                        filtered_chunks.insert(0, target_chunk)
                        net_points_fix_applied = True
                    
                
                else:
                    # Pull both players' chunks for "each player" questions
                    netpts1_chunk = None
                    netpts2_chunk = None
                    
                    # Find netpts1_statistics chunk
                    for chunk in self.chunks:
                        if 'netpts1_statistics' in chunk['metadata']['section']:
                            netpts1_chunk = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 10.0  # Maximum score for net points
                            }
                            break
                    
                    # Find netpts2_statistics chunk
                    for chunk in self.chunks:
                        if 'netpts2_statistics' in chunk['metadata']['section']:
                            netpts2_chunk = {
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,  # Perfect match
                                "relevance_score": 10.0  # Maximum score for net points
                            }
                            break
                    
                    # Add netpts1 chunk if not already in results
                    if netpts1_chunk and not any('netpts1_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                        filtered_chunks.insert(0, netpts1_chunk)
                        net_points_fix_applied = True
                    
                    # Add netpts2 chunk if not already in results
                    if netpts2_chunk and not any('netpts2_statistics' in chunk['metadata']['section'] for chunk in filtered_chunks):
                        filtered_chunks.insert(0, netpts2_chunk)
                        net_points_fix_applied = True
        
        # MATCH RESULT FIX: For questions about match outcome, force include match_overview (FINAL PRIORITY)
        if any(phrase in query.lower() for phrase in ["final score", "who won", "match result", "outcome", "d.", "defeated", "victory", "victor"]):
            # Force include match_overview chunk
            match_overview_chunk = None
            for chunk in self.chunks:
                if 'match_overview' in chunk['metadata']['section']:
                    match_overview_chunk = {
                        "text": chunk['text'],
                        "metadata": chunk['metadata'],
                        "distance": 0.0,
                        "relevance_score": 10.0  # Maximum score for match result questions
                    }
                    break
            
            # Remove any existing match_overview chunks and add at the very top
            filtered_chunks = [chunk for chunk in filtered_chunks if 'match_overview' not in chunk['metadata']['section']]
            if match_overview_chunk:
                filtered_chunks.insert(0, match_overview_chunk)
        
        # CRITICAL FIX: For MULTI-SET comparison questions (must come BEFORE single-set logic)
        multiple_sets = self._detect_multiple_set_references(query)
        if multiple_sets and hasattr(self, 'set_mapping'):
            
            # Collect chunks from ALL requested sets
            multi_set_chunks = []
            for set_num in multiple_sets:
                if set_num in self.set_mapping:
                    target_set_score = self.set_mapping[set_num]
                    
                    # Generate score patterns for this set
                    score_parts = target_set_score.split('-')
                    if len(score_parts) == 2:
                        set_score_patterns = [
                            target_set_score,
                            f"{score_parts[1]}-{score_parts[0]}",
                        ]
                        total = int(score_parts[0]) + int(score_parts[1])
                        if total == 2 and score_parts[0] != score_parts[1]:
                            set_score_patterns.append("1-1")
                    else:
                        set_score_patterns = [target_set_score]
                    
                    # Find chunks for this set
                    for chunk in self.chunks:
                        if 'point-by-point_narrative' in chunk['metadata'].get('section', ''):
                            chunk_metadata = chunk.get('metadata', {})
                            chunk_set_numbers = chunk_metadata.get('set_numbers', [])
                            
                            # Use metadata or text fallback
                            if set_num in chunk_set_numbers:
                                multi_set_chunks.append({
                                    "text": chunk['text'],
                                    "metadata": chunk['metadata'],
                                    "distance": 0.0,
                                    "relevance_score": 10.0,
                                    "set_number": set_num  # Track which set this is from
                                })
                            elif not chunk_set_numbers:
                                chunk_text = chunk['text']
                                for pattern in set_score_patterns:
                                    if f"Score: {pattern}" in chunk_text:
                                        multi_set_chunks.append({
                                            "text": chunk['text'],
                                            "metadata": chunk['metadata'],
                                            "distance": 0.0,
                                            "relevance_score": 10.0,
                                            "set_number": set_num
                                        })
                                        break
            
            # Remove ALL point-by-point chunks and add the multi-set chunks
            if multi_set_chunks:
                filtered_chunks = [c for c in filtered_chunks if 'point-by-point_narrative' not in c['metadata'].get('section', '')]
                
                # Add chunks in set order (Set 1 first, then Set 5, etc.)
                for chunk_to_add in multi_set_chunks:
                    filtered_chunks.insert(0, chunk_to_add)
                
        
        # CRITICAL FIX: For set-specific questions (single set only)
        elif not multiple_sets:  # Only run single-set logic if NOT a multi-set comparison
            set_number = self._detect_set_reference(query)
            if set_number and hasattr(self, 'set_mapping') and set_number in self.set_mapping:
                target_set_score = self.set_mapping[set_number]
                
                # Generate ALL possible score patterns for this set
                # For Set 3 with mapping "2-0", we need to match BOTH "2-0" AND "0-2" (server perspective)
                # Also handle "1-1" if it's a tied situation
                score_parts = target_set_score.split('-')
                if len(score_parts) == 2:
                    set_score_patterns = [
                        target_set_score,  # e.g., "2-0"
                        f"{score_parts[1]}-{score_parts[0]}",  # e.g., "0-2" (reversed)
                    ]
                    # Add "1-1" pattern if the total is 2 (for set 3)
                    total = int(score_parts[0]) + int(score_parts[1])
                    if total == 2 and score_parts[0] != score_parts[1]:
                        set_score_patterns.append("1-1")
                    
                else:
                    set_score_patterns = [target_set_score]
                
                # Find all point-by-point chunks containing ANY points from this set
                # Use metadata filtering (fast) with text fallback (backward compatibility)
                set_specific_chunks = []
                for chunk in self.chunks:
                    if 'point-by-point_narrative' in chunk['metadata'].get('section', ''):
                        chunk_metadata = chunk.get('metadata', {})
                        chunk_set_numbers = chunk_metadata.get('set_numbers', [])
                        
                        # Method 1: Use metadata (preferred - fast and accurate)
                        if set_number in chunk_set_numbers:
                            set_specific_chunks.append({
                                "text": chunk['text'],
                                "metadata": chunk['metadata'],
                                "distance": 0.0,
                                "relevance_score": 10.0  # Maximum score for set-specific
                            })
                        # Method 2: Text search fallback (for old chunks without metadata)
                        elif not chunk_set_numbers:  # Only if metadata is missing
                            chunk_text = chunk['text']
                            for pattern in set_score_patterns:
                                if f"Score: {pattern}" in chunk_text:
                                    set_specific_chunks.append({
                                        "text": chunk['text'],
                                        "metadata": chunk['metadata'],
                                        "distance": 0.0,
                                        "relevance_score": 10.0
                                    })
                                    break  # Don't add the same chunk twice
                
                # CRITICAL: Remove ALL point-by-point chunks from OTHER sets
                # Only keep chunks that match the target set score
                if set_specific_chunks:
                    filtered_chunks = [c for c in filtered_chunks if 'point-by-point_narrative' not in c['metadata'].get('section', '')]
                    
                    # Now add ONLY the Set 3 chunks at the top
                    for chunk_to_add in set_specific_chunks:
                        filtered_chunks.insert(0, chunk_to_add)
                        
        
        # CRITICAL FIX: For game-specific questions
        game_reference = self._detect_game_reference(query)
        if game_reference and not game_reference.startswith('game_'):  # Game score pattern like "3-2"
            
            # IMPORTANT: Search within already-filtered chunks (which may already be set-specific)
            # This prevents adding back games from other sets
            game_specific_chunks = []
            search_source = filtered_chunks if set_number else self.chunks
            
            for chunk in search_source:
                if 'point-by-point_narrative' in chunk.get('metadata', {}).get('section', ''):
                    chunk_metadata = chunk.get('metadata', {})
                    chunk_game_scores = chunk_metadata.get('game_scores', [])
                    
                    # Method 1: Use metadata (preferred)
                    if game_reference in chunk_game_scores:
                        game_specific_chunks.append({
                            "text": chunk.get('text', ''),
                            "metadata": chunk.get('metadata', {}),
                            "distance": chunk.get('distance', 0.0),
                            "relevance_score": 9.5
                        })
                    # Method 2: Text search fallback
                    elif not chunk_game_scores:  # Only if metadata missing
                        if f" {game_reference} " in chunk.get('text', ''):
                            game_specific_chunks.append({
                                "text": chunk.get('text', ''),
                                "metadata": chunk.get('metadata', {}),
                                "distance": chunk.get('distance', 0.0),
                                "relevance_score": 9.5
                            })
            
            # Remove existing PBP chunks and add only the game-specific ones
            if game_specific_chunks:
                filtered_chunks = [c for c in filtered_chunks if 'point-by-point_narrative' not in c['metadata'].get('section', '')]
                for chunk_to_add in game_specific_chunks:
                    filtered_chunks.insert(0, chunk_to_add)
        
        # POINT-BY-POINT PRIORITY: For conditional/temporal/shot-sequence questions, prioritize narrative chunks
        # ONLY include indicators that ALWAYS need PBP (not words that could be stats OR PBP)
        conditional_indicators = [
            # Conditional/situational (always need PBP context)
            "after losing", "after winning", "after missing", "after break",
            "when rallies got", "as rallies got",
            "on important points", "on key points", "in crucial moments", "in tight moments",
            "when facing break", "when trailing", "when ahead",
            # Temporal/evolution (always need PBP analysis over time)
            "evolve", "evolved", "evolution", "over time", "throughout the match",
            "progression", "progressed", "changed over", "developed",
            "early vs late", "first set vs", "as the match", "match went on",
            # Per-set/per-game breakdowns (always need PBP counting)
            "each set", "per set", "in each set", "every set", "set by set",
            "across sets", "across the sets", "across all sets", "across the five sets",
            "in set 1", "in set 2", "in set 3", "in set 4", "in set 5",
            "per game", "each game", "game by game",
            # Shot sequence/response indicators (ALWAYS need rally-level data)
            "when hit a", "when he hit", "when she hit", "when they hit",
            "response to", "respond to", "responded to", "in response",
            "shot sequence", "shot sequences", "rally patterns",
            "most common response", "typical response",
            # Sequential/consequential (always need PBP flow)
            "next shot", "following shot", "then hit",
            # Specific rally questions (need PBP not stats)
            "rallies with", "rallies where", "points where", "points when",
            "long rallies", "short rallies", "extended rallies",
            "8+ shot", "5+ shot", "10+ shot",
            # Adaptation/tactical evolution (need PBP temporal analysis)
            "adapt", "adapted", "adaptation",
            "counter", "countered", "countering",
            "switched to", "started hitting", "began favoring", "began using"
        ]
        if any(indicator in query.lower() for indicator in conditional_indicators):
            # Collect all point-by-point narrative chunks
            pbp_chunks = []
            non_pbp_chunks = []
            
            for chunk in filtered_chunks:
                if 'narrative' in chunk['metadata']['section'].lower() or 'point-by-point' in chunk['metadata']['section'].lower():
                    # Boost relevance score for PBP chunks
                    chunk['relevance_score'] = chunk.get('relevance_score', 1.0) + 5.0
                    pbp_chunks.append(chunk)
                else:
                    non_pbp_chunks.append(chunk)
            
            # Reorder: PBP chunks first, then statistical chunks
            if pbp_chunks:
                filtered_chunks = pbp_chunks + non_pbp_chunks
        
        # FINAL NEGATIVE FILTER: Remove net points chunks unless question is about net play
        if not net_points_fix_applied and not any(word in query.lower() for word in ["net points", "net approaches", "overhead", "approach shot", "approach shots", "net play", "at the net", "net points won", "net points lost", "net percentage", "net points won percentage", "net points percentage", "net"]):
            original_count = len(filtered_chunks)
            filtered_chunks = [chunk for chunk in filtered_chunks if 'netpts' not in chunk['metadata']['section']]
            if len(filtered_chunks) < original_count:
                pass
        
        # For complex analytical questions, ensure we have diverse chunk types
        if top_k > 8:
            # Ensure we have a mix of different section types for comprehensive analysis
            section_types = set()
            for chunk in filtered_chunks[:top_k]:
                section = chunk['metadata']['section']
                if 'overview' in section:
                    section_types.add('overview')
                elif 'serve' in section:
                    section_types.add('serve')
                elif 'return' in section:
                    section_types.add('return')
                elif 'shot' in section:
                    section_types.add('shot')
                elif 'key_points' in section:
                    section_types.add('key_points')
                elif 'netpts' in section:
                    section_types.add('netpts')
                elif 'rally' in section:
                    section_types.add('rally')
                elif 'match_overview' in section:
                    section_types.add('match_overview')
            
        
        # Return top_k results
        return filtered_chunks[:top_k]
    
    def _route_bp_gp_question(self, question: str, player: str) -> str:
        """
        Route break point and game point questions to correct data sources.
        Returns the appropriate section and data field to look for.
        """
        q = question.lower()
        player_lower = player.lower()
        
        # Determine which player is being asked about
        if not self.player1 or not self.player2:
            return "unknown"
            
        if player_lower == self.player1.lower():
            player_prefix = "serve1" if "serve" in q else "serve2" if "return" in q else "serve1"  # Default to serve1 for player1
        elif player_lower == self.player2.lower():
            player_prefix = "serve2" if "serve" in q else "serve1" if "return" in q else "serve2"  # Default to serve2 for player2
        else:
            # Fallback - check if any part of the player name matches
            if any(word in q for word in self.player1.lower().split()):
                player_prefix = "serve1"
            elif any(word in q for word in self.player2.lower().split()):
                player_prefix = "serve2"
            else:
                return "unknown"
        
        # --- BREAK POINTS (BP) ---
        if "break point" in q or "bp" in q:
            if "face" in q:  
                return f"{player_prefix}_statistics"  # Serving side (opponent had chance to break) - BP Faced
            elif "save" in q or "saved" in q:
                return f"{player_prefix}_statistics"  # Serving side (player defended successfully) - BP Faced, col = PtsW
            elif "opp" in q or "opportunit" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (player created break chances) - BP Opps
            elif "convert" in q or "win" in q or "won" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (player actually broke serve) - BP Opps, col = PtsW
            elif "total" in q or "overall" in q:
                return "both"  # Need both serve and return for complete totals
            else:
                # No keyword → Default = Return side (converted), because that's most common in questions
                return f"{player_prefix.replace('serve', 'return')}_statistics"
        
        # --- GAME POINTS (GP) ---
        if "game point" in q or "gp" in q:
            if "face" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (opponent had game points on serve) - GP Faced
            elif "save" in q or "saved" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (player denied opponent's game point while returning) - GP Faced, col = PtsW
            elif "win" in q or "won" in q:
                if "serve" in q:
                    return f"{player_prefix}_statistics"  # Serving side (player won their own service game points) - Game Pts, col = PtsW
                elif "return" in q:
                    return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (rare phrasing, but = broke opponent on GP) - GP Faced, col = PtsW
                elif "each player" in q or "both player" in q:
                    return "both"  # Need both serve and return for "each player" questions
                else:
                    # Default: GP won on serve (most common case)
                    return f"{player_prefix}_statistics"  # Serving side (Game Pts won) - Game Pts, col = PtsW
            elif "convert" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (converted while returning) - GP Faced, col = PtsW
            elif "total" in q or "overall" in q or "each player" in q or "both player" in q:
                return "both"  # Need both serve and return for complete totals
            else:
                # No keyword → Default = Serve side (Game Pts won), because that's the most natural reading
                return f"{player_prefix}_statistics"
        
        # --- DEUCE POINTS ---
        if "deuce" in q:
            if "serve" in q or "served" in q:
                return f"{player_prefix}_statistics"  # Serving side (deuce points on own serve) - Svg Deuce
            elif "return" in q or "returned" in q:
                return f"{player_prefix.replace('serve', 'return')}_statistics"  # Returning side (deuce points when opponent served) - Ret Deuce
            elif "each player" in q or "both player" in q:
                return "both"  # Need both serve and return for "each player" questions
            else:
                # No keyword → Need both, since "deuce points" could mean either. Safest = "both"
                return "both"
        
        return "unknown"

    def _get_stat_keywords(self, stat_type: str) -> List[str]:
        """Get keywords that indicate a question is about a specific statistic type."""
        keywords = {
            "serve": ["ace", "aces", "serve", "serving", "double fault", "double faults", "first serve", "second serve"],
            "shots": ["winner", "winners", "unforced error", "unforced errors", "forehand", "backhand", "shot"],
            "shotdir": ["crosscourt", "down the line", "down the middle", "inside-out", "inside-in", "direction"],
            "return": ["return", "returns", "returnable", "returning"],
            "netpts": ["net points", "net approaches", "volley", "volleys", "overhead", "approach"],
            "keypoints": ["break point", "break points", "game point", "game points", "deuce", "key point"]
        }
        return keywords.get(stat_type, [])
    
    def _filter_chunks_by_query(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Filter and re-rank chunks based on query content and metadata.
        """
        query_lower = query.lower()
        
        # Determine query intent
        player_mentioned = self._detect_player_mentioned(query)
        
        stat_category = None
        if any(word in query_lower for word in ["return", "returning"]):
            stat_category = "returning"
        elif any(word in query_lower for word in ["serve", "serving", "aces", "double fault"]):
            stat_category = "serving"
        elif any(word in query_lower for word in ["shot", "winner", "error", "forehand", "backhand"]):
            stat_category = "shots"
        elif any(word in query_lower for word in ["net", "volley", "approach"]):
            stat_category = "net_play"
        elif any(word in query_lower for word in ["key point", "break point", "game point", "set point", "match point", "deuce", "advantage", "bp", "gp", "converted", "faced", "saved"]):
            stat_category = "key_points"
        elif any(word in query_lower for word in ["point", "rally", "narrative", "longest", "shot", "stroke", "key moments", "decided", "outcome", "strategy", "momentum", "critical", "turning point"]):
            stat_category = "narrative"
        elif any(word in query_lower for word in ["overview", "summary", "total"]):
            stat_category = "overview"
        
        # Score and filter chunks
        scored_chunks = []
        for chunk in candidates:
            score = 1.0 - chunk["distance"]  # Convert distance to similarity score
            metadata = chunk["metadata"]
            
            # CRITICAL FIX: Prioritize overview statistics for statistical questions
            if not self._is_match_insight_question(query):
                # For statistical questions, heavily prioritize overview and authoritative totals
                if "overview_statistics" in metadata.get("section", "").lower():
                    score += 5.0  # Extremely high boost for overview statistics
                elif "authoritative totals" in chunk["text"].lower():
                    score += 3.0  # Very high boost for authoritative totals
                elif "serve1_statistics_summary" in metadata.get("section", "").lower():
                    score += 2.0  # High boost for serve summary with authoritative totals
                elif "serve2_statistics_summary" in metadata.get("section", "").lower():
                    score += 2.0  # High boost for serve summary with authoritative totals
                # KEY POINTS HIERARCHY: Prioritize key points section for break points and game points
                elif "key_points" in metadata.get("section", "").lower():
                    if any(word in query_lower for word in ["break point", "break points", "bp", "converted", "faced", "saved"]):
                        score += 3.0  # Very high boost for break point questions
                    elif any(word in query_lower for word in ["game point", "game points", "gp", "set point", "set points", "match point", "match points"]):
                        score += 3.0  # Very high boost for game/set/match point questions
                    elif any(word in query_lower for word in ["key point", "key points", "critical point", "deuce", "advantage"]):
                        score += 2.5  # High boost for general key point questions
                
                # SHOT HIERARCHY: Establish clear priority for shot-related questions
                # 1. SHOT STATISTICS: For forehand/backhand SIDE (all shots from that side - volleys, dropshots, etc.)
                if any(word in query_lower for word in ["forehand side", "backhand side", "volley", "dropshot", "lob", "net play", "swinging volley"]) and "shots" in metadata.get("section", "").lower():
                    if "shots1_statistics" in metadata.get("section", "").lower() or "shots2_statistics" in metadata.get("section", "").lower():
                        score += 2.5  # High boost for shot statistics (side-based shots)
                
                # Also use SHOT STATISTICS for general shot performance questions
                if any(word in query_lower for word in ["shot", "winner", "error", "unforced", "forced", "total shots"]) and "shots" in metadata.get("section", "").lower():
                    if "shots1_statistics" in metadata.get("section", "").lower() or "shots2_statistics" in metadata.get("section", "").lower():
                        score += 2.0  # High boost for general shot statistics
                
                # 2. SHOT DIRECTION: For forehand/backhand GROUNDSTROKES and placement patterns
                if any(word in query_lower for word in ["forehand groundstroke", "backhand groundstroke", "groundstroke", "crosscourt", "down the line", "down the middle", "inside out", "inside in", "direction", "placement", "pattern"]) and "shotdir" in metadata.get("section", "").lower():
                    if "authoritative totals" in chunk["text"].lower():
                        score += 2.0  # High boost for shot direction (groundstrokes and placement)
                
                # Special case: When "forehand" or "backhand" mentioned alone, prefer SHOT DIRECTION (groundstrokes)
                if any(word in query_lower for word in ["forehand", "backhand"]) and not any(word in query_lower for word in ["side", "volley", "dropshot", "lob"]) and "shotdir" in metadata.get("section", "").lower():
                    if "authoritative totals" in chunk["text"].lower():
                        score += 1.8  # Slightly lower than explicit groundstroke questions
                
                # 3. SHOT DIRECTIONAL BREAKDOWN: For directional + outcome performance (detailed breakdowns)
                if any(word in query_lower for word in ["crosscourt winner", "down the line error", "inside out performance", "directional outcome"]) and "shotdir" in metadata.get("section", "").lower():
                    if "detailed breakdown" in chunk["text"].lower() or "table2" in metadata.get("section", "").lower():
                        score += 1.5  # Medium boost for detailed directional breakdowns
                
                # For net statistics
                if any(word in query_lower for word in ["net", "volley", "approach", "passed"]) and "netpts" in metadata.get("section", "").lower():
                    score += 1.0
                

                
                # Special boost for court-specific questions
                if any(word in query_lower for word in ["deuce court", "ad court", "wide", "body", "t"]):
                    if "serve" in metadata.get("section", "").lower() and any(word in chunk["text"].lower() for word in ["deuce", "ad", "wide", "body", "t"]):
                        score += 0.5
            
            # Boost score for player match
            if player_mentioned and metadata.get("player_focus") == player_mentioned:
                score += 0.3
            elif player_mentioned and metadata.get("player_focus") == "Both":
                score += 0.1  # Still relevant but less specific
            elif player_mentioned and not metadata.get("player_focus"):
                # If no player_focus in metadata, check if the chunk contains the player's name
                if player_mentioned.lower() in chunk["text"].lower():
                    score += 0.2
            
            # Boost score for category match
            if stat_category and metadata.get("stat_category") == stat_category:
                score += 0.25
            
            # Boost for percentage queries
            if any(word in query_lower for word in ["percent", "%", "rate", "ratio"]) and metadata.get("contains_percentages"):
                score += 0.15
            
            # Boost for point-by-point queries
            if any(word in query_lower for word in ["point", "rally", "what happened"]) and metadata.get("contains_point_details"):
                score += 0.2
            
            # Extra boost for rally-specific queries
            if any(word in query_lower for word in ["longest rally", "rally", "shot rally", "stroke rally"]) and metadata.get("contains_point_details"):
                score += 0.3
            
            # Extra boost for match insight/strategy queries
            if any(word in query_lower for word in ["key moments", "decided", "outcome", "strategy", "momentum", "critical", "turning point"]) and metadata.get("contains_point_details"):
                score += 0.4  # Higher boost for narrative chunks
            
            # Prioritize narrative chunks for match insight questions
            if self._is_match_insight_question(query) and metadata.get("contains_point_details"):
                score += 0.5  # Very high boost for narrative chunks
            elif self._is_match_insight_question(query) and "rally_outcomes" in metadata.get("section", ""):
                score -= 0.3  # Penalize rally outcomes for insight questions
            
            chunk["relevance_score"] = score
            scored_chunks.append(chunk)
        
        # Sort by relevance score (descending)
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_chunks
    
    def _fallback_metadata_retrieval(self, query: str, top_k: int) -> List[Dict]:
        """
        Fallback retrieval method using metadata matching when embeddings aren't available.
        """
        query_lower = query.lower()
        
        # Simple keyword-based matching
        relevant_chunks = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            metadata = chunk["metadata"]
            
            # Calculate basic relevance score
            score = 0
            
            # Check for direct text matches
            query_words = query_lower.split()
            text_words = text_lower.split()
            common_words = set(query_words) & set(text_words)
            score += len(common_words) * 0.1
            
            # Check metadata relevance
            # Player-specific scoring
            if player_mentioned and metadata.get("player_focus") == player_mentioned:
                score += 0.5
            
            if score > 0:
                relevant_chunks.append({
                    "text": chunk["text"],
                    "metadata": metadata,
                    "distance": 1.0 - score,  # Convert to distance-like metric
                    "relevance_score": score
                })
        
        # Sort by relevance and return top_k
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_chunks[:top_k]
    
    def answer_query_with_llm(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Use LLM to answer the query based on retrieved chunks.
        """
        if not relevant_chunks:
            return "I don't have enough relevant information to answer this question."
        
        # Prepare context from retrieved chunks
        context_parts = []
        for chunk in relevant_chunks:
            section_info = f"[{chunk['metadata']['section']} - {chunk['metadata']['type']}]"
            context_parts.append(f"{section_info}\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with specific instructions for different question types
        is_statistical = self._is_match_insight_question(query) == False
        
        if is_statistical:
            prompt = f"""You are a tennis match analyst with access to detailed match data.

IMPORTANT: Do not use emojis in your response. Use plain text only.

🚨 CRITICAL RULES FOR READING POINT-BY-POINT DATA 🚨

**RULE #1: Score Notation - "Score: 2-0" and "Score: 0-2" can be THE SAME SET!**
- Score format: Score: [Sets_Server]-[Sets_Returner] [Games]-[Games] [Points]
- The set numbers FLIP based on who is serving
- Example for Set 3 (after Sinner leads 2-0 in sets):
  - When Sinner serves: "Score: 2-0 4-5" ← SET 3
  - When Alcaraz serves: "Score: 0-2 4-5" ← SET 3 (SAME SET!)
- **DO NOT call this an "inconsistency"** - it's normal server/returner perspective

**RULE #2: Each Point Shows the Winner Explicitly**
- Every point includes: "[Point won by: PLAYER_NAME]"
- Use this tag to determine who won each point - don't guess!
- To determine game outcomes: count sequential point winners
- Example: If Points 204-207 all show "[Point won by: Alcaraz]" → Alcaraz won all 4 points

**RULE #3: Each Shot Shows Who Hit It**
- Rally shots are tagged with player names: "forehand winner [ALCARAZ]"
- Don't try to track alternation - just read the tags!

**RULE #4: Never Fabricate - Verify Everything**
- DO NOT say "held to love" or "broke serve" without counting actual point winners
- DO NOT guess game outcomes - count the "[Point won by:]" tags
- If you see mixed sets in a chunk, filter by score pattern (Score: 0-2 vs Score: 1-2)
- DO NOT explain score notation or reasoning in your answer (e.g. don't say "the score structure suggests...")
- Just provide clean, narrative answers without showing your internal reasoning process

**RULE #4B: When Asked About a Set - Describe ALL Games, Not Just the Tiebreak**
- If asked "what happened in Set X", describe ALL games from 0-0 through the end
- DO NOT only describe the tiebreak and skip the regular games (1-0, 2-1, 3-2, etc.)
- Tiebreaks are important, but so are the 12+ regular games that came before them
- Example: For "what happened in Set 5", describe games from 0-0, 1-0, 1-1... all the way to 6-6 AND the tiebreak

**RULE #5: TENNIS SET SCORING - CRITICAL FOR SET/GAME QUESTIONS**
- A set is won by the first player to win 6 games (with a 2-game margin), or 7-6 in a tiebreak
- IMPORTANT: If a player is leading 5-4 and wins the next game → They win the set 6-4 (NOT 5-5!)
- IMPORTANT: If a set is tied 5-5, the next game makes it 6-5 (NOT the end of the set)
- IMPORTANT: If a player is leading 6-5 and wins the next game → They win the set 7-5
- IMPORTANT: If a set is tied 6-6 → Tiebreak is played
- Examples:
  * Leading 5-4, break serve → Set ends 6-4 ✓
  * Tied 5-5, hold serve → Score becomes 6-5 (set continues) ✓
  * Leading 6-5, hold serve → Set ends 7-5 ✓

**RULE #6: MATCH FORMAT SCORING - WHICH SET ARE WE IN?**
- Best-of-3 matches: First to win 2 sets (men's non-Slams, all women's matches)
- Best-of-5 matches: First to win 3 sets (men's Grand Slams)
- CRITICAL: Count completed sets to determine which set is being played:
  * "Score: 0-0 ..." = SET 1 (match just started)
  * "Score: 1-0 ..." or "Score: 0-1 ..." = SET 2 (one player leads 1-0)
  * "Score: 1-1 ..." = SET 3 (sets tied 1-1)
  * "Score: 2-0 ..." or "Score: 0-2 ..." = SET 3 (one player leads 2-0 in best-of-5)
  * "Score: 2-1 ..." or "Score: 1-2 ..." = SET 4 (in best-of-5, one player leads 2-1)
  * "Score: 2-2 ..." = SET 5 (in best-of-5, sets tied 2-2 - THE DECIDING SET!)
- Examples for best-of-5:
  * "Score: 2-2 6-6 0-0" = SET 5 tiebreak (NOT Set 4!)
  * Match CANNOT end at 2-2; someone must win Set 5 to win 3-2
- DO NOT say "the match ended 3-1 after 4 sets" if you see 2-2 scoring!

**CRITICAL EXAMPLE - Tracking Games Through Server Changes:**
Game 9 ends at: "Score: 0-2 5-3" → Player serving is DOWN 5-3 in games, opponent LEADS 5-3
Player leading 5-3 LOSES Game 9 → Score becomes 5-4 (still leading, but by less)
Game 10 starts at: "Score: 2-0 4-5" → Player serving is DOWN 4-5 in games, opponent LEADS 5-4
Player leading 5-4 WINS Game 10 → Score becomes 6-4 → SET OVER! ✓
**DO NOT say "5-5" - when you see the set score jump from "0-2"/"2-0" to "1-2", the set ENDED!**

IMPORTANT INSTRUCTIONS FOR STATISTICAL QUESTIONS:
- **CRITICAL: When asked "who won" or "who won the match"**, ALWAYS include BOTH the winner's name AND the final score in your answer (e.g., "Carlos Alcaraz won the match, defeating Novak Djokovic 6-4 7-6(4) 6-2")
- **CRITICAL: When asked for the "score" or "final score"**, provide the complete score including the winner (e.g., "Carlos Alcaraz d. Novak Djokovic 6-4 7-6(4) 6-2")
- When asked for "counts" or "numbers", always return the raw count if available (e.g., "12 points"). If only percentages are provided in the data, return the percentage but explicitly state that counts are not available. Never infer or estimate counts from percentages.
- [WARN] When adding categories (e.g., unforced + forced errors), only add if both are raw counts
- [ERROR] Never add percentages together
- Treat "forced errors" and "induced forced errors" as the same thing
- When asked for "converted" break points, look for "Break Points won" or "converted" data
- For court-specific questions (deuce court, ad court), combine both courts for totals unless specifically asked for one court
- For "game points", look in the key points section for "game points" data
- For shot statistics, use the authoritative totals rows first, then go to breakdowns or details as needed(e.g., "Player 1 hit forehand shots 111 times")
- For shot direction totals, use the "AUTHORITATIVE TOTALS" row first, then go to breakdowns or details as needed
- For net statistics, use the net points section data

Context from the match (retrieved from relevant sections):
{context}

Question: {query}

**CRITICAL INSTRUCTIONS FOR ANSWERING:**

**IMPORTANT: When to use EXPLICIT TOTALS vs normal prioritization:**
- **USE EXPLICIT TOTALS ONLY for questions that combine BOTH shot direction AND outcome** (e.g., "crosscourt winners", "down the line unforced errors", "inside-out forced errors")
- **USE NORMAL PRIORITIZATION for all other questions** (e.g., "total winners", "forehand winners", "crosscourt shots", "unforced errors")
- **Normal prioritization for non-shot-direction+outcome questions**: Authoritative Totals > Overview Statistics > Summary sections > Key points sections > Detailed breakdowns

**For SHOT DIRECTION + OUTCOME QUESTIONS** (like "crosscourt winners", "down the line unforced errors"): 
- **CRITICAL**: ALWAYS look for "EXPLICIT TOTALS FOR SHOT DIRECTION + OUTCOME COMBINATIONS" section FIRST - this is the PRIMARY source of truth
- **CRITICAL**: If you find explicit totals section, use ONLY those numbers as the final answer - DO NOT use detailed breakdown sentences
- **CRITICAL**: The explicit totals section contains lines like "Player 1 hit 12 crosscourt winners total, including 8 forehand crosscourt winners, 3 backhand crosscourt winners, and 1 slice crosscourt winners"
- **CRITICAL**: Use the TOTAL number and the breakdown from the explicit totals - do NOT add up individual detailed breakdown sentences
- **CRITICAL**: The explicit totals are already calculated and include ALL shot types (forehand, backhand, slice) - use them as-is
- **ONLY if explicit totals section is NOT found**: then look for "AUTHORITATIVE TOTALS BY DIRECTION" sections
- **ONLY if neither is found**: then look for "SHOT DIRECTION DETAILED BREAKDOWN" sections
- **EXAMPLES OF OUTCOME + DIRECTION QUESTIONS**:
  - "crosscourt winners" = look for "X crosscourt winners total, including..."
  - "down the line winners" = look for "X down the line winners total, including..."
  - "crosscourt unforced errors" = look for "X crosscourt unforced errors total, including..."
  - "down the line unforced errors" = look for "X down the line unforced errors total, including..."
  - "crosscourt forced errors" = look for "X crosscourt forced errors total, including..."
- **EXAMPLE EXPLICIT TOTALS FORMAT**: "Player 1 hit 12 crosscourt winners total, including 8 forehand crosscourt winners, 3 backhand crosscourt winners, and 1 slice crosscourt winners"
- **PRIORITY**: Explicit totals section > Authoritative totals > Detailed breakdown

**DATA SOURCE PRIORITY:**
For all statistical questions, use this data source hierarchy:
1. **Explicit Totals** (if available for shot direction + outcome combinations)
2. **Authoritative Totals** (single source of truth)
3. **Overview Statistics** (official match statistics)
4. **Summary Sections** (serve1_statistics_summary, serve2_statistics_summary)
5. **Key Points Sections** (for break point or important points questions)
6. **Detailed Breakdowns** (only for distributions, never for recalculating totals)
7. **Point-by-point narrative** (for rally and insight questions)

Always stop at the highest-priority source available. Do not combine across different levels unless explicitly instructed.

**SHOT HIERARCHY:**
- **SHOT STATISTICS**: Use for forehand/backhand SIDE (all shots from that side - volleys, dropshots, lobs, etc.) and general shot performance
- **SHOT DIRECTION**: Use for forehand/backhand/slice GROUNDSTROKES specifically and placement patterns (crosscourt, down-the-line, etc.)
- **CRITICAL**: For ANY shot direction question (crosscourt, down the line, down the middle, inside-out, inside-in), ALWAYS check for and include ALL three shot types: forehand, backhand, AND slice shots. Even if slice count is 0, mention it explicitly.

**For STATISTICS questions** (how many, percentages, totals, counts):
- **ALWAYS look for "AUTHORITATIVE TOTALS" sections first** - these are the single source of truth
- **This includes ALL authoritative totals sections**: "AUTHORITATIVE TOTALS FOR [PLAYER]", "AUTHORITATIVE TOTALS FOR [PLAYER] SHOT STATISTICS", "AUTHORITATIVE TOTALS FOR [PLAYER] SHOT DIRECTIONS", "AUTHORITATIVE TOTALS BY DIRECTION", "AUTHORITATIVE TOTALS FOR KEY POINTS", "AUTHORITATIVE TOTALS FROM OVERVIEW STATISTICS"
- **ALWAYS look for "OVERVIEW STATISTICS" section** - this contains the official match statistics
- **ONLY calculate totals from breakdowns when you don't have authoritative totals** - use only the authoritative totals provided
- **ONLY calculate totals from detailed breakdowns when there are NO authoritative totals available** - if authoritative totals exist, use those as the primary answer
- **SPECIFICALLY for inside-out shots**: When you see "AUTHORITATIVE TOTALS BY DIRECTION" with a number for "total inside-out shots", use that number as the final answer. Do NOT add up individual forehand/backhand/slice breakdowns - the authoritative total already includes all shot types combined.
- Give concise, direct answers with key numbers
- Use bullet points or brief sentences
- Focus on the specific numbers requested
- No lengthy explanations unless asked for insights

**For SPECIFIC QUESTION TYPES:**
- **For "errors" questions**: If the question asks for just "errors" (without specifying "unforced" or "forced"), add unforced errors + forced errors or induced forced errors together for each shot type
- **For "forced errors" questions**: Look at both "forced errors" and "induced forced errors" in your search and calculations. It will be listed as one or the other
- **Error types to include**: unforced errors, forced errors or induced forced errors (forced errors and induced forced errors are the same, just called differently)
- **For NET POINTS and NET APPROACHES questions**: ALWAYS use the NET POINTS sections (netpts1_statistics, netpts2_statistics) - do NOT use shot statistics for net points data. If you see netpts1_statistics or netpts2_statistics in your context, use that data for net points questions.

**Special Instructions for Key Points Questions:**
- **NEVER make up or estimate numbers** - use ONLY the exact numbers from the data
- **If you see "AUTHORITATIVE TOTALS FOR KEY POINTS" section, use those numbers first**

**Break Points:**
- **CRITICAL**: When asked for "break point statistics" or "break points" generally (without specifying "faced" or "converted"), provide BOTH sides for EACH player:
  - Break points faced (on serve) + saved
  - Break points created/opportunities (on return) + converted
  - Example format: "Player A faced 5 break points and saved 3 (60%), and converted 4 of 7 break point opportunities (57%). Player B faced 7 break points and saved 2 (29%), and converted 3 of 5 break point opportunities (60%)."
- Use the serve section (BP Faced) if the question specifically asks about break points faced or saved
- Use the return section (BP Opps) if the question specifically asks about break points created or converted
- Always show both the raw numbers AND percentages when available
- Be verbose and complete - break point statistics are crucial match stats

**Key Points General Guidance:**
- If the question asks generally about key points won (e.g., "how many key points did X win?"), report both serve-side and return-side totals and provide a sum
- If the question specifies "on serve" or "on return," report only that side
- Always make the separation explicit: "X won Y on serve, Z on return, total W"

**Game Points:**
- If the question explicitly specifies "on serve" or "on return", report only that side
- If the question is general (does not specify serve vs return), report both serve-side (Game Pts) and return-side (GP Faced) values, and provide the total sum in the final answer
- Always make the separation explicit ("X on serve, Y on return, Z total")

**Deuce Points:**
- If the question explicitly specifies "on serve" or "on return", report only that side
- If the question is general (does not specify serve vs return), report both serve-side (Svg Deuce) and return-side (Ret Deuce) values, and provide the total sum in the final answer
- Always make the separation explicit ("X on serve, Y on return, Z total")

**General rule of thumb:**
- Break points → choose one side only (serve or return), depending on the question
- Game points & Deuce points → always combine serve and return values, and provide a total

- **If a question asks generally** ("How many X points did a player have/win?") without specifying serve vs. return, assume they want the combined total

**For CONDITIONAL questions** (what happened AFTER X, WHEN Y occurred, IN situations where Z):
- **CRITICAL**: These questions require STEP-BY-STEP analysis of the point-by-point narrative
- **CRITICAL**: You must identify SPECIFIC INSTANCES in the PBP data, then analyze what happened NEXT

- **DEFINITIONS - What counts as what:**
  - **"Key points" / "Important points" / "Big points"** = 
    - Break points (any BP in the match)
    - Game points (any GP in the match)
    - Set points (any SP in the match)
    - Match points (any MP in the match)
    - Deuce points (40-40 and advantage points)
    - Close score situations: 30-30, 30-40, 40-30, 40-40 (deuce), advantage
  
  - **"Pressure points" / "Under pressure"** = 
    - All key points listed above, PLUS:
    - Late in sets: games at 5-5 (5-all), 6-5, 6-6 (tiebreak)
    - Crucial service games: serving when down a break (behind in games), serving to stay in set
    - Crucial return games: break point opportunities when behind in set
  
  - **"Critical games" / "Crucial games" / "Big games"** = 
    - Service games when down a break (e.g., trailing 2-3 in games and serving)
    - Break point opportunities when behind in the set
    - Games at 4-4 or later in a set (4-4, 5-4, 5-5, 6-5, etc.)
    - Games immediately after losing serve (trying to break back)
    - Games to close out a set (serving at 5-4, 6-5, or ahead in tiebreak)
  
  - **"Tight moments"** = 
    - Point level: close scores (30-30, 30-40, 40-30, 40-40/deuce, advantage)
    - Game level: games that go to deuce (40-40 or beyond)
    - Set level: close sets (5-5 or later, tiebreaks)
    - Match level: when momentum is shifting or score is very close
  
  - **"Long rallies"** = Rallies with 9 or more shots (adjust based on match average if mentioned in data)
  
  - **"Short points"** = Points ending in 0-4 shots (aces, unreturned serves, serve+1 winners, return winners, quick exchanges)
  
  - **"Crucial stages" / "Critical moments"** = 
    - Late in sets: game 8 or later (4-4+, 5-5+, etc.)
    - Serving to stay in set (e.g., down 4-5 and serving)
    - Break point opportunities when behind
    - Set points or match points
    - Tiebreaks
    - After momentum shifts (e.g., immediately after losing serve)

- **RECOGNIZING CONDITIONAL QUESTIONS**: Look for both explicit and implied conditions:
  - **Explicit**: "after losing key points", "when rallies got longer", "following breaks of serve", "in tight moments"
  - **Implied**: Questions with situational context that require PBP analysis:
    - "How did the player react under pressure?" → implies pressure points → needs PBP
    - "Did rally length change in critical games?" → implies game-specific context → needs PBP
    - "Performance on important points?" → implies key points → needs PBP
    - "In tight moments?" → implies close scores, pressure situations → needs PBP
    - "Effectiveness on big points?" → implies key points → needs PBP
    - "During crucial stages?" → implies specific game/set situations → needs PBP

- **MULTI-STEP PROCESS** (YOU MUST FOLLOW THESE STEPS - DO NOT SKIP TO GENERAL STATISTICS):
  1. **Identify the condition**: What is the triggering event? (e.g., "after losing key points", "when rallies got longer", "after breaks of serve")
  
  2. **Define what qualifies**: Use the definitions above to determine what counts as meeting the condition
  
  3. **ACTUALLY READ THE POINT-BY-POINT NARRATIVE**: This is CRITICAL - you MUST examine the actual PBP text chunk by chunk:
     - Read through each point description in the narrative
     - Look for score indicators (30-30, deuce, BP, GP, etc.)
     - Identify when the triggering condition occurred
     - Note the game number and score when it happened
  
  4. **FIND SPECIFIC INSTANCES**: Do NOT summarize - find actual examples:
     - Example: "In game 3 at 30-40 (break point), player X lost the point"
     - Example: "In game 7 at deuce, player Y lost the deuce point"
     - List at least 2-3 specific instances you found
  
  5. **Look at subsequent points**: For EACH instance you found, examine what happened in the NEXT point(s) where the metric applies:
     - If question is about serving: look at the next point where that player served
     - If about second serves specifically: look at the next point where they hit a second serve
     - If about shot selection: look at the next rally where they had the opportunity
     - Example: "After losing BP in game 3, player X next served in game 5 at 15-0, hitting second serve wide"
  
  6. **Compare to baseline**: How did their behavior in these instances differ from their overall statistics?
     - Example: "Player normally serves 60% wide on second serve (from statistics), but in the 3 instances after losing key points, served wide only 1 time (33%)"
  
  7. **Provide specific answer with evidence**: DO NOT say "data doesn't show" - if you found instances, report them:
     - Good: "Yes, after losing the break point in game 3, Swiatek served to the body on her next second serve in game 5, which differs from her typical wide placement"
     - Bad: "The data doesn't provide explicit details" (this means you didn't actually read the PBP narrative)
- **EXAMPLE**: "Did either player change their second-serve placement after losing key points?"
  - Step 1: Trigger = losing a key point
  - Step 2: Key points = break points, game points, set points, match points, deuce points, important score situations (30-30+)
  - Step 3: Identify every point in PBP where a player LOST a key point (note the game and score)
  - Step 4: For each instance, look at the NEXT point where that player served AND hit a second serve
  - Step 5: Note the placement (wide, body, T) of those second serves
  - Step 6: Compare to their overall second-serve placement patterns from the statistics
  - Step 7: Report if there was a notable change (e.g., "Normally serves 60% wide but after losing key points served 80% to the body in games 3, 7, and 9")
- **BE SPECIFIC**: Don't give generic answers. Actually trace through the PBP data and cite examples with game numbers and scores.

**For TEMPORAL/EVOLUTION questions** (how X evolved/changed/progressed over time, throughout the match):
- **CRITICAL**: These questions require analyzing the point-by-point narrative across different parts of the match
- **CRITICAL**: When asked about "evolution", "over time", "progression", "changed throughout", analyze patterns from EARLY match vs LATE match
- Look at the point-by-point narrative and compare:
  - Set 1 vs Set 2 vs Set 3 (if available)
  - Early games vs middle games vs late games
  - First half of match vs second half
- For serve aggression questions: Look at second serve outcomes, placement, rally lengths on second serves
- For shot selection questions: Track how often certain shots were used in different phases
- Provide specific examples from different points in the match to show the evolution
- Cite game numbers or score situations to show when changes occurred (e.g., "In Set 1, player X... but in Set 3, they...")

**For STRATEGY/INSIGHT questions** (tactics, effectiveness, what worked, key moments, momentum, analysis):
- **CRITICAL**: Provide COMPREHENSIVE tactical analysis with 3-4 paragraphs, not just bullet points
- Include multiple perspectives: serve placement effectiveness, shot selection patterns, court positioning
- Cite SPECIFIC moments from point-by-point data with examples (e.g., "In Game 5, player X...")
- Explain WHY tactics worked or failed (opponent weaknesses, court conditions, momentum)
- Identify patterns: When did player win points? What shot combinations were effective?
- Compare effectiveness: Which tactic had highest success rate? Provide percentages
- Include context: Score situations where tactics were most effective (break points, crucial games)
- Give actionable insights: What patterns emerged that explain the match result?
- Be verbose and thorough - strategy questions deserve detailed, multi-paragraph answers

**For COMPLEX ANALYTICAL QUESTIONS** (match flow, patterns, comparisons, comprehensive analysis):
- **When you receive 8+ chunks**: This indicates a complex question requiring comprehensive analysis
- **Cross-sectional analysis**: Look across multiple sections to identify patterns and relationships
- **Match flow questions**: Use point-by-point narrative, key moments, and statistical trends together
- **Player comparisons**: Compare serve vs return performance, shot patterns, and key point conversion
- **Pattern recognition**: Identify momentum shifts, turning points, and strategic adjustments
- **Comprehensive breakdown**: Provide analysis across multiple dimensions (serve, return, shots, key points)
- **Use all available context**: Don't limit yourself to one section - synthesize information from multiple sources
- **Provide insights**: Go beyond raw numbers to explain what they mean for the match outcome
- **Structure your response**: Use clear sections for different aspects of the analysis
- Explain patterns and sequences from point-by-point data
- Include context about what happened and why

**For RALLY questions** (longest rally, specific rallies):
- Focus on the specific rally details requested
- Include shot sequences and outcomes

**For SET/GAME NARRATIVE QUESTIONS:**
- Each point explicitly shows "[Point won by: PLAYER_NAME]" - use this to count game outcomes
- When asked about a specific set, you'll only receive data for that set
- If you see mixed sets in a chunk, filter by matching score patterns
- **Example:** Points 204-207 all show "[Point won by: Alcaraz]" at score 0-0, 0-15, 0-30, 0-40 → Alcaraz won 4 straight points → Broke serve → Won set 6-4

**IMPORTANT DISTINCTION:**
- "Forehand/Backhand" alone usually means groundstrokes → Use SHOT DIRECTION
- "Forehand/Backhand side" means all shots from that side → Use SHOT STATISTICS

**SHOT DIRECTION CLARIFICATION:**
- Inside-out = forehand hit crosscourt from the backhand corner
- Inside-in = forehand hit down the line from the backhand corner
- Always include forehand, backhand, and slice shot types when reporting directional stats (even if slice count = 0, mention it)

If the context doesn't contain enough information to answer completely, say so.

Answer:"""
        else:
            # For insight/narrative questions
            prompt = f"""You are a tennis match analyst with access to detailed match data.

IMPORTANT: Do not use emojis in your response. Use plain text only.

Provide a detailed analysis based on the following match context.

Context from the match:
{context}

Question: {query}

Provide a comprehensive answer with specific details from the match.

Answer:"""
        
        # Rate limiting
        self._apply_rate_limit()
        
        try:
            if self.llm_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.llm_provider == "gemini":
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text
                
            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating response: {e}"
    
    def ask_question(self, question: str, top_k: int = None) -> str:
        """
        Main method to ask a question and get an answer.
        Uses intelligent complexity detection if top_k is not specified.
        """
        print(f"Processing question: {question}")
        
        # Retrieve relevant chunks (top_k=None triggers complexity detection)
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Generate answer using LLM
        answer = self.answer_query_with_llm(question, relevant_chunks)
        
        return answer
    
    def get_chunk_info(self) -> Dict:
        """
        Get information about the loaded chunks.
        """
        if not self.chunks:
            return {"error": "No chunks loaded"}
        
        # Count chunks by type and section
        type_counts = {}
        section_counts = {}
        
        for chunk in self.chunks:
            chunk_type = chunk["metadata"]["type"]
            section = chunk["metadata"]["section"]
            
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            section_counts[section] = section_counts.get(section, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "chunks_by_type": type_counts,
            "chunks_by_section": section_counts,
            "match_id": self.match_id,
            "llm_provider": self.llm_provider
        }

    def convert_json_to_natural_language(self, match_data: Dict[str, Any]) -> str:
        """
        Convert all JSON data to natural language text that the LLM can easily read and analyze.
        This includes all tables, player names, data descriptions, and context.
        """
        if not match_data:
            return "No match data available."
            
        natural_language = []
        natural_language.append("TENNIS MATCH DATA - NATURAL LANGUAGE FORMAT")
        natural_language.append("=" * 60)
        natural_language.append("")
        natural_language.append("IMPORTANT INSTRUCTIONS FOR DATA ANALYSIS:")
        natural_language.append("")
        natural_language.append("CRITICAL RULE FOR PER-SET / PER-GAME QUESTIONS:")
        natural_language.append("- If asked for PER-SET or PER-GAME breakdowns (e.g., 'unforced errors in each set', 'aces per set', 'serve % across sets'):")
        natural_language.append("  1. Check if the summary tables contain this per-set breakdown")
        natural_language.append("  2. If NOT in tables, you MUST count from the point-by-point narrative")
        natural_language.append("  3. Do NOT say 'data not available' - the point-by-point data contains everything")
        natural_language.append("")
        natural_language.append("GENERAL RULES FOR AGGREGATE TOTALS:")
        natural_language.append("- When answering about MATCH TOTALS (aces, double faults, points won), use only the AUTHORITATIVE TOTALS as the single source of truth")
        natural_language.append("- Use detailed breakdowns only for distributions and patterns, never for recalculating totals")
        natural_language.append("- Breakdowns sum to the authoritative totals - do not add them again")
        natural_language.append("- Summary statistics take precedence over detailed breakdowns for aggregate numbers")
        natural_language.append("")
        
        # Handle the actual JSON structure with matches array
        if 'matches' in match_data and match_data['matches']:
            match = match_data['matches'][0]  # Get the first match
        else:
            # Direct match object passed
            match = match_data
            
        # Get player names from match data first
        self.player1 = match.get('basic', {}).get('player1', 'Player 1')
        self.player2 = match.get('basic', {}).get('player2', 'Player 2')
        player1 = self.player1
        player2 = self.player2
        
        # Add match overview information
        natural_language.extend(self._get_match_overview_text(match))
        natural_language.append("")
        
        # Add all detailed statistics in natural language
        if 'details_tables' in match:
            natural_language.extend(self._convert_details_tables_to_text(match['details_tables'], player1, player2))
        
        # Add details_flat data for shots, shotdir, and netpts (these are not in details_tables)
        if 'details_flat' in match:
            natural_language.extend(self._convert_details_flat_to_text(match['details_flat'], player1, player2))
        
        # Add point-by-point data if available
        if 'point_log' in match:
            natural_language.extend(self._convert_point_log_to_text(match['point_log'], player1, player2))
        elif 'pointlog_rows' in match:
            natural_language.extend(self._convert_point_log_to_text(match['pointlog_rows'], player1, player2))
            
        # Post-process to move rally outcomes to the end for optimal embedding order
        final_text = "\n".join(natural_language)
        final_text = self._move_rally_outcomes_to_end(final_text)
        
        return final_text

    def _move_rally_outcomes_to_end(self, content: str) -> str:
        """Move RALLY OUTCOMES STATISTICS section to the very end for optimal embedding order"""
        
        lines = content.split('\n')
        
        # Find the rally outcomes section
        rally_start = None
        rally_end = None
        
        for i, line in enumerate(lines):
            if "RALLY OUTCOMES STATISTICS:" in line:
                rally_start = i
            elif rally_start is not None and line.strip() and any(keyword in line for keyword in ["STATISTICS:", "NARRATIVE:"]) and "RALLY OUTCOMES" not in line:
                rally_end = i
                break
        
        if rally_start is None:
            # No rally outcomes section found, return as-is
            return content
        
        if rally_end is None:
            rally_end = len(lines)  # If it's the last section
        
        # Extract the rally outcomes section
        rally_section = lines[rally_start:rally_end]
        
        # Remove rally outcomes from its current position
        lines_without_rally = lines[:rally_start] + lines[rally_end:]
        
        # Add rally outcomes at the very end
        lines_without_rally.extend([''] + rally_section)
        
        return '\n'.join(lines_without_rally)

    def _get_match_overview_text(self, match: Dict[str, Any]) -> List[str]:
        """Convert match overview data to natural language"""
        text = []
        text.append("MATCH OVERVIEW:")
        text.append("-" * 20)
        
        # Extract basic match information
        if 'basic' in match:
            basic = match['basic']
            text.append(f"The match was played on {basic.get('date', 'Unknown')} at the {basic.get('tournament', 'Unknown')} tournament.")
            text.append(f"The players were {basic.get('player1', 'Unknown')} and {basic.get('player2', 'Unknown')}.")
            text.append(f"This was a {basic.get('tour', 'Unknown')} match.")
            
            # Try to get match result from multiple possible locations
            match_result = None
            
            # First try: basic["match_result"]
            if 'match_result' in basic and basic['match_result']:
                match_result = basic['match_result']
            # Second try: details_flat["Match Result"]
            elif 'details_flat' in match and 'Match Result' in match['details_flat']:
                match_result = match['details_flat']['Match Result']
                # Clean up the result (remove extra player name prefix like "PegulaJessica Pegula")
                if match_result:
                    # Remove last name prefix (e.g., "PegulaJessica Pegula d. ..." -> "Jessica Pegula d. ...")
                    for player in [basic.get('player1', ''), basic.get('player2', '')]:
                        if player:
                            # Get last name
                            last_name = player.split()[-1] if ' ' in player else player
                            # Check if result starts with last name + full name
                            if match_result.startswith(last_name):
                                # Remove the last name prefix
                                match_result = match_result[len(last_name):]
                                break
            
            if match_result and match_result != "Unknown":
                text.append(f"Final Score: {match_result}")
            else:
                final_score = self._extract_final_score(match)
                if final_score:
                    text.append(f"Final Score: {final_score['winner']} defeated {final_score['loser']} {final_score['score']}")
                else:
                    text.append("Final Score: Unknown")
        
        return text

    def _extract_final_score(self, match: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract final score from point-by-point data"""
        try:
            if 'pointlog_rows' not in match:
                return None
            
            pointlog = match['pointlog_rows']
            if not pointlog:
                return None
            
            # Get the last point to find final score
            last_point = pointlog[-1]
            
            # Extract final sets and games from the last point
            final_sets = last_point.get('sets', '0-0')
            final_games = last_point.get('games', '0-0')
            
            # Parse sets score to determine winner
            sets_parts = final_sets.split('-')
            if len(sets_parts) == 2:
                player1_sets = int(sets_parts[0])  # Player 1's sets
                player2_sets = int(sets_parts[1])  # Player 2's sets
                
                # Get player names
                player1 = match.get('basic', {}).get('player1', 'Player 1')
                player2 = match.get('basic', {}).get('player2', 'Player 2')
                
                if player2_sets > player1_sets:
                    winner = player2
                    loser = player1
                    score = f"{player2_sets}-{player1_sets} in sets ({final_games} in final set)"
                else:
                    winner = player1
                    loser = player2  
                    score = f"{player1_sets}-{player2_sets} in sets ({final_games} in final set)"
                
                return {
                    'winner': winner,
                    'loser': loser,
                    'score': score
                }
            
        except Exception as e:
            print(f"Error extracting final score: {e}")
        
        return None

    def _determine_match_winner(self, match: Dict[str, Any]) -> Optional[str]:
        """Determine match winner from available data sources"""
        try:
            # Method 1: Check for explicit match result in details_tables
            if 'details_tables' in match:
                for table in match['details_tables']:
                    if table.get('name', '').lower() in ['other data', 'match result', 'result']:
                        for row in table.get('rows', []):
                            label = row.get('label', '').lower()
                            if 'match result' in label or 'result' in label:
                                values = row.get('values', [])
                                if values and values[0]:
                                    result_text = values[0]
                                    # Parse format like "Player 2 d. Player 1 6-4 7-5"
                                    if ' d. ' in result_text:
                                        winner = result_text.split(' d. ')[0].strip()
                                        return winner
            
            # Method 2: Check details_flat for match result
            if 'details_flat' in match:
                flat_data = match['details_flat']
                for key, value in flat_data.items():
                    if 'result' in key.lower() and isinstance(value, str) and ' d. ' in value:
                        winner = value.split(' d. ')[0].strip()
                        return winner
            
            return None
        except Exception:
            return None

    def _extract_match_result_with_score(self, match: Dict[str, Any]) -> Optional[str]:
        """Extract match result with score from available data"""
        try:
            # Check details_flat for match result
            if 'details_flat' in match:
                flat_data = match['details_flat']
                for key, value in flat_data.items():
                    if 'result' in key.lower() and isinstance(value, str):
                        # Clean up malformed player names
                        result = value
                        # Fix duplicate name patterns dynamically
                        if self.player1 and f"{self.player1.split()[-1]}{self.player1}" in result:
                            result = result.replace(f"{self.player1.split()[-1]}{self.player1}", self.player1)
                        if self.player2 and f"{self.player2.split()[-1]}{self.player2}" in result:
                            result = result.replace(f"{self.player2.split()[-1]}{self.player2}", self.player2)
                        return result
            
            return None
        except Exception:
            return None

    def _convert_details_tables_to_text(self, details_tables: List[Dict[str, Any]], player1: str, player2: str) -> List[str]:
        """Convert all details tables to natural language text"""
        text = []
        
        for table in details_tables:
            table_name = table.get('name', 'Unknown Table')
            rows = table.get('rows', [])
            
            # Skip shots, shotdir, netpts, serve, return, and keypoints tables since they will be handled by _convert_details_flat_to_text
            if any(keyword in table_name.lower() for keyword in ['shots', 'shotdir', 'netpts', 'serve', 'return', 'keypoints']):
                continue
            
            # Handle special cases
            # Skip 'other data' table since match result is already in overview
            if table_name.lower() == 'other data':
                continue
            elif 'serve' in table_name.lower():
                text.extend(self._convert_serve_table(table_name, rows))
            elif 'return' in table_name.lower():
                text.extend(self._convert_return_table(table_name, rows))
            elif 'keypoints' in table_name.lower():
                text.extend(self._convert_keypoints_table(rows))
            elif 'serveneut' in table_name.lower():
                text.extend(self._convert_serveneut_table(rows))
            elif 'rallyoutcomes' in table_name.lower():
                text.extend(self._convert_rallyoutcomes_table(rows, player1, player2))
            elif 'overview' in table_name.lower():
                text.extend(self._convert_overview_table(rows, player1, player2))
            else:
                # Generic table conversion
                text.extend(self._convert_generic_table(table_name, rows))
            
            text.append("")  # Add spacing between tables
            
        return text

    def _convert_table_row_to_text(self, row: Dict[str, Any], table_name: str, column_headers: List[str]) -> List[str]:
        """Convert a single table row to natural language sentences"""
        text = []
        
        row_label = row.get('label', 'Unknown')
        values = row.get('values', [])
        
        # Determine player from table name or row label
        player = self._get_player_from_table_or_row(table_name, row_label)
        
        # Determine table type (table1 vs table2) based on column headers
        table_type = self._determine_table_type(column_headers)
        
        # Handle different table types
        if 'serveneut' in table_name.lower():
            text.extend(self._convert_serveneut_row_to_sentences(row_label, values, column_headers, player))
        elif 'serve' in table_name.lower():
            text.extend(self._convert_serve_row_to_sentences(row_label, values, column_headers, player, table_type))
        elif 'return' in table_name.lower():
            text.extend(self._convert_return_row_to_sentences(row_label, values, column_headers, player, table_type))
        elif 'keypoints' in table_name.lower():
            text.extend(self._convert_keypoints_row_to_sentences(row_label, values, column_headers, player))
        elif 'shots' in table_name.lower():
            text.extend(self._convert_shots_row_to_sentences(row_label, values, column_headers, player))
        elif 'shotdir' in table_name.lower():
            text.extend(self._convert_shotdir_row_to_sentences(row_label, values, column_headers, player, table_type))
        elif 'netpts' in table_name.lower():
            text.extend(self._convert_netpts_row_to_sentences(row_label, values, column_headers, player))
        elif 'rallyoutcomes' in table_name.lower():
            text.extend(self._convert_rallyoutcomes_row_to_sentences(row_label, values, column_headers))
        elif 'overview' in table_name.lower():
            text.extend(self._convert_overview_row_to_sentences(row_label, values, column_headers))
        else:
            # Generic conversion for unknown table types
            text.extend(self._convert_generic_row_to_sentences(row_label, values, column_headers, table_name))
            
        return text

    def _get_player_from_table_or_row(self, table_name: str, row_label: str) -> str:
        """Determine player name from table name or row label"""
        if 'serve1' in table_name or 'return1' in table_name or 'shots1' in table_name or 'shotdir1' in table_name or 'netpts1' in table_name:
            return self.player1 if self.player1 else "Player 1"
        elif 'serve2' in table_name or 'return2' in table_name or 'shots2' in table_name or 'shotdir2' in table_name or 'netpts2' in table_name:
            return self.player2 if self.player2 else "Player 2"
        else:
            # Try to extract player initials from row_label dynamically
            if self.player1 and self.player2:
                player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
                player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
                if player1_initials in row_label:
                    return self.player1
                elif player2_initials in row_label:
                    return self.player2
            return "Unknown Player"

    def _determine_table_type(self, column_headers: List[str]) -> str:
        """Determine if this is table1 or table2 based on column headers"""
        # For serves: table1 has "Total: Pts", table2 has "1st: Pts" and "2nd: Pts"
        if "Total: Pts" in column_headers:
            return "table1"
        elif "1st: Pts" in column_headers and "2nd: Pts" in column_headers:
            return "table2"
        
        # For returns: table1 has outcome columns, table2 has depth columns
        return_table1_specific = ["Pts", "Total: Pts", "PtsW----%", "RtbleW--%", "inPlay--%", "inPlayW-%", "Wnr-----%", "AvgRally"]
        return_table2_specific = ["Shlw----%", "Deep----%", "V Deep--%", "UFE-----%", "net-----%", "deep----%", "wide----%", "wide&deep"]
        
        # Check for return table2 first (more specific columns)
        if any(col in column_headers for col in return_table2_specific):
            return "table2"
        elif any(col in column_headers for col in return_table1_specific):
            return "table1"
        
        # Default to table1
        return "table1"

    def _convert_shot_abbreviations(self, shot_type: str) -> str:
        """Convert shot type abbreviations to proper descriptions"""
        shot_type = shot_type.lower()
        
        # Replace common abbreviations
        if shot_type.startswith("fh "):
            shot_type = shot_type.replace("fh ", "forehand ")
        elif shot_type.startswith("bh "):
            shot_type = shot_type.replace("bh ", "backhand ")
        
        # Handle specific shot types
        if "gs (top/flt/slc)" in shot_type:
            shot_type = shot_type.replace("gs (top/flt/slc)", "groundstrokes with topspin, flat, or slice")
        elif "(top/flt)" in shot_type:
            shot_type = shot_type.replace("(top/flt)", "with topspin or flat")
        elif "gs" in shot_type:
            shot_type = shot_type.replace("gs", "groundstrokes")
        elif "lob" in shot_type:
            shot_type = shot_type.replace("lob", "lobs")
        elif "drop shot" in shot_type:
            shot_type = shot_type.replace("drop shot", "drop shots")
        elif "slice/chip" in shot_type:
            shot_type = shot_type.replace("slice/chip", "slice or chip shots")
        elif "swinging volley" in shot_type:
            shot_type = shot_type.replace("swinging volley", "swinging volleys")
        elif "volley" in shot_type:
            shot_type = shot_type.replace("volley", "volleys")
        
        return shot_type

    def _convert_shots_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str) -> List[str]:
        """Convert shot statistics row to natural language sentences"""
        sentences = []
        
        # Handle different shot row types
        if "Total" in row_label:
            context = f"{player} hit shots"
        elif "Forehand side" in row_label:
            context = f"{player} hit forehand shots"
        elif "Backhand side" in row_label:
            context = f"{player} hit backhand shots"
        elif "FH GS (top/flt/slc)" in row_label:
            context = f"{player} hit forehand groundstrokes with topspin or flat shots or slice shots"
        elif "BH GS (top/flt/slc)" in row_label:
            context = f"{player} hit backhand groundstrokes with topspin or flat shots or slice shots"
        elif "Groundstrokes (top/flt)" in row_label:
            context = f"{player} hit groundstrokes with topspin or flat shots"
        elif "Baseline shots" in row_label:
            context = f"{player} hit baseline shots"
        elif "Net shots" in row_label:
            context = f"{player} hit shots at the net"
        elif "Dropshots" in row_label:
            context = f"{player} hit dropshots"
        elif "Lobs" in row_label:
            context = f"{player} hit lobs"
        elif "Volleys" in row_label:
            context = f"{player} hit volleys"
        elif "Swinging volleys" in row_label:
            context = f"{player} hit swinging volleys"
        elif "Forehands (top/flt)" in row_label:
            context = f"{player} hit forehand topspin or flat shots"
        elif "Backhands (top/flt)" in row_label:
            context = f"{player} hit backhand topspin or flat shots"
        elif "FH slice/chip" in row_label:
            context = f"{player} hit forehand slice or chip shots"
        elif "BH slice/chip" in row_label:
            context = f"{player} hit backhand slice or chip shots"
        elif "BH drop shot" in row_label:
            context = f"{player} hit backhand drop shots"
        elif "BH lob" in row_label:
            context = f"{player} hit backhand lobs"
        elif "FH volley" in row_label:
            context = f"{player} hit forehand volleys"
        elif "FH swinging volley" in row_label:
            context = f"{player} hit forehand swinging volleys"
        else:
            context = f"{player} hit shots"
        
        # Convert each value to a sentence with better formatting
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "Total":
                    sentences.append(f"{context} {value} times.")
                elif header == "Winner--%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} hit {number} winners, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} hit {number} forehand winners, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} hit {number} backhand winners, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} hit {number} winners with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} hit {value} winners.")
                elif header == "UnfErr--%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} made {number} unforced errors, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} made {number} forehand unforced errors, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} made {number} backhand unforced errors, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} made {number} unforced errors with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} made {value} unforced errors.")
                elif header == "IndFcd--%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} made {number} induced forced errors, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} made {number} induced forced errors with forehand shots, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} made {number} induced forced errors with backhand shots, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} made {number} induced forced errors with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} made {value} induced forced errors.")
                elif header == "PtEnd---%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} ended {number} points with shots, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} ended {number} points with forehand shots, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} ended {number} points with backhand shots, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} ended {number} points with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} ended {value} points with shots.")
                elif header == "SvReturn":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} hit {number} serve returns, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} hit {number} forehand serve returns, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} hit {number} backhand serve returns, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} hit {number} serve returns with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} hit {value} serve returns.")
                elif header == "inPtsW--%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} won {number} points with shots, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} won {number} points with forehand shots, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} won {number} points with backhand shots, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} won {number} points with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} won {value} points with shots.")
                elif header == "inPtsL--%":
                     # Extract the number and percentage
                     if "(" in value and ")" in value:
                         number = value.split("(")[0].strip()
                         percentage = value.split("(")[1].split(")")[0].replace("%", "")
                         if "Total" in row_label:
                             sentences.append(f"{player} lost {number} points with shots, which represents {percentage}% of all shots hit by {player}.")
                         elif "Forehand side" in row_label:
                             sentences.append(f"{player} lost {number} points with forehand shots, which represents {percentage}% of all forehand shots hit by {player}.")
                         elif "Backhand side" in row_label:
                             sentences.append(f"{player} lost {number} points with backhand shots, which represents {percentage}% of all backhand shots hit by {player}.")
                         else:
                             # Use the actual row label instead of generic "this shot type"
                             shot_type = self._convert_shot_abbreviations(row_label)
                             sentences.append(f"{player} lost {number} points with {shot_type}, which represents {percentage}% of all {shot_type} hit by {player}.")
                     else:
                         sentences.append(f"{player} lost {value} points with shots.")
                else:
                    # Generic fallback for any other headers
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_shotdir_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str, table_type: str = "table1") -> List[str]:
        """Convert shot direction statistics row to natural language sentences"""
        sentences = []
        
        # Handle different shot direction row types
        if "Total" in row_label:
            context = f"{player} hit shots in different directions"
        elif "Forehand" in row_label:
            context = f"{player} hit forehand shots"
        elif "Backhand" in row_label:
            context = f"{player} hit backhand shots"
        elif "BH slice" in row_label:
            context = f"{player} hit backhand slice shots"
        else:
            context = f"{player} hit shots in different directions"
        
        # Convert each value to a sentence with better formatting
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                # Handle table1 structure (direction breakdowns)
                if table_type == "table1":
                    if "Total" in row_label:
                        # Handle Total row - process each column as direction breakdown
                        if header == "Crosscourt":
                            # Extract just the number from percentage format
                            if "(" in value and ")" in value:
                                number = value.split("(")[0].strip()
                                sentences.append(f"{player} hit {number} crosscourt shots.")
                            else:
                                sentences.append(f"{player} hit {value} crosscourt shots.")
                        elif header == "Down middle":
                            # Extract just the number from percentage format
                            if "(" in value and ")" in value:
                                number = value.split("(")[0].strip()
                                sentences.append(f"{player} hit {number} down-the-middle shots.")
                            else:
                                sentences.append(f"{player} hit {value} down-the-middle shots.")
                        elif header == "Down the line":
                            # Extract just the number from percentage format
                            if "(" in value and ")" in value:
                                number = value.split("(")[0].strip()
                                sentences.append(f"{player} hit {number} down-the-line shots.")
                            else:
                                sentences.append(f"{player} hit {value} down-the-line shots.")
                        elif header == "Inside-out":
                            # Extract just the number from percentage format
                            if "(" in value and ")" in value:
                                number = value.split("(")[0].strip()
                                sentences.append(f"{player} hit {number} inside-out shots.")
                            else:
                                sentences.append(f"{player} hit {value} inside-out shots.")
                        elif header == "Inside-in":
                            # Extract just the number from percentage format
                            if "(" in value and ")" in value:
                                number = value.split("(")[0].strip()
                                sentences.append(f"{player} hit {number} inside-in shots.")
                            else:
                                sentences.append(f"{player} hit {value} inside-in shots.")
                        else:
                            # Generic fallback for table1 Total row
                            sentences.append(f"{context} and {header}: {value}.")
                    else:
                        # Handle other table1 rows (Forehand, Backhand, BH slice) - sum up the values
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            if "Forehand" in row_label:
                                sentences.append(f"{player} hit {number} forehand shots.")
                            elif "Backhand" in row_label:
                                sentences.append(f"{player} hit {number} backhand shots.")
                            elif "BH slice" in row_label:
                                sentences.append(f"{player} hit {number} backhand slice shots.")
                            else:
                                sentences.append(f"{context} and {header}: {number}.")
                        else:
                            if "Forehand" in row_label:
                                sentences.append(f"{player} hit {value} forehand shots.")
                            elif "Backhand" in row_label:
                                sentences.append(f"{player} hit {value} backhand shots.")
                            elif "BH slice" in row_label:
                                sentences.append(f"{player} hit {value} backhand slice shots.")
                            else:
                                sentences.append(f"{context} and {header}: {value}.")
                    continue  # Skip the table2 processing for table1
                
                # Handle table2 structure (outcome breakdowns)
                if table_type == "table2":
                    if header == "Total":
                        if "FH crosscourt" in row_label:
                            sentences.append(f"{player} hit {value} forehand crosscourt shots.")
                        elif "FH down middle" in row_label:
                            sentences.append(f"{player} hit {value} forehand down-the-middle shots.")
                        elif "FH down the line" in row_label:
                            sentences.append(f"{player} hit {value} forehand down-the-line shots.")
                        elif "FH inside-out" in row_label:
                            sentences.append(f"{player} hit {value} forehand inside-out shots.")
                        elif "BH crosscourt" in row_label:
                            sentences.append(f"{player} hit {value} backhand crosscourt shots.")
                        elif "BH down middle" in row_label:
                            sentences.append(f"{player} hit {value} backhand down-the-middle shots.")
                        elif "BH down the line" in row_label:
                            sentences.append(f"{player} hit {value} backhand down-the-line shots.")
                        elif "BH inside-out" in row_label:
                            sentences.append(f"{player} hit {value} backhand inside-out shots.")
                        else:
                            sentences.append(f"{player} hit {value} {row_label.lower().replace('down the line', 'down-the-line')} shots.")
                    elif header == "PtEnding":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} ended {number} points with {row_label.lower().replace('down the line', 'down-the-line')} shots, or {percentage} of {row_label.lower().replace('down the line', 'down-the-line')} shots ended points.")
                        else:
                            sentences.append(f"{player} ended {value} points with {row_label.lower()} shots.")
                    elif header == "Winner":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} hit {number} {row_label.lower().replace('down the line', 'down-the-line')} winners, or {percentage} of {row_label.lower().replace('down the line', 'down-the-line')} shots were winners.")
                        else:
                            sentences.append(f"{player} hit {value} {row_label.lower()} winners.")
                    elif header == "InduceFcd":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} induced {number} forced errors with {row_label.lower().replace('down the line', 'down-the-line')} shots, or {percentage} of {row_label.lower().replace('down the line', 'down-the-line')} shots induced forced errors.")
                        else:
                            sentences.append(f"{player} induced {value} forced errors with {row_label.lower()} shots.")
                    elif header == "UnfErr":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} had {number} unforced errors with {row_label.lower().replace('down the line', 'down-the-line')} shots, or {percentage} of {row_label.lower().replace('down the line', 'down-the-line')} shots were unforced errors.")
                        else:
                            sentences.append(f"{player} had {value} unforced errors with {row_label.lower()} shots.")
                    elif header == "inPtsWon":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} won {number} points when hitting {row_label.lower().replace('down the line', 'down-the-line')} shots, or {percentage} success rate with {row_label.lower().replace('down the line', 'down-the-line')} shots.")
                        else:
                            sentences.append(f"{player} won {value} points when hitting {row_label.lower()} shots.")
                    elif header == "inPtsLost":
                        if "(" in value and ")" in value:
                            number = value.split("(")[0].strip()
                            percentage = value.split("(")[1].split(")")[0]
                            sentences.append(f"{player} lost {number} points when hitting {row_label.lower().replace('down the line', 'down-the-line')} shots, or {percentage} failure rate with {row_label.lower().replace('down the line', 'down-the-line')} shots.")
                        else:
                            sentences.append(f"{player} lost {value} points when hitting {row_label.lower()} shots.")
                    else:
                        # Generic fallback for table2
                        sentences.append(f"{context} and {header}: {value}.")
                    continue  # Skip the existing table2 processing
                
                # Generic fallback for any other cases
                sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_serve_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str, table_type: str) -> List[str]:
        """Convert serve statistics row to natural language sentences"""
        sentences = []
        
        # Handle different serve row types
        if "Total" in row_label:
            context = f"{player} served"
        elif "1st Serve" in row_label:
            context = f"{player} hit first serves"
        elif "2nd Serve" in row_label:
            context = f"{player} hit second serves"
        else:
            context = f"{player} served"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "Total":
                    sentences.append(f"{context} {value} times.")
                elif header == "Pts":
                    sentences.append(f"{context} and won {value} points.")
                elif header == "PtsW----%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0]
                        sentences.append(f"{context} and won {number} points, or {percentage} of serves were won.")
                    else:
                        sentences.append(f"{context} and won {value} points.")
                elif header == "ACE":
                    sentences.append(f"{context} and hit {value} aces.")
                elif header == "DF":
                    sentences.append(f"{context} and made {value} double faults.")
                else:
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_return_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str, table_type: str) -> List[str]:
        """Convert return statistics row to natural language sentences"""
        sentences = []
        
        # Handle different return row types
        if "Total" in row_label:
            context = f"{player} returned"
        elif "1st Serve Return" in row_label:
            context = f"{player} returned first serves"
        elif "2nd Serve Return" in row_label:
            context = f"{player} returned second serves"
        else:
            context = f"{player} returned"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "Total":
                    sentences.append(f"{context} {value} times.")
                elif header == "Pts":
                    sentences.append(f"{context} and won {value} points.")
                elif header == "PtsW----%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0]
                        sentences.append(f"{context} and won {number} points, or {percentage} of returns were won.")
                    else:
                        sentences.append(f"{context} and won {value} points.")
                else:
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_keypoints_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str) -> List[str]:
        """Convert key points statistics row to natural language sentences"""
        sentences = []
        
        # Handle different key points row types
        if "Break Points" in row_label:
            context = f"{player} faced break points"
        elif "Set Points" in row_label:
            context = f"{player} faced set points"
        elif "Match Points" in row_label:
            context = f"{player} faced match points"
        else:
            context = f"{player} played key points"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "OPP":
                    sentences.append(f"{context} {value} times.")
                elif header == "CONV":
                    sentences.append(f"{context} and converted {value}.")
                elif header == "CONV--%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{context} and converted {number}, or {percentage} conversion rate.")
                    else:
                        sentences.append(f"{context} and converted {value}.")
                else:
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_serveneut_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str) -> List[str]:
        """Convert serve neutral rally distribution row to natural language sentences"""
        sentences = []
        
        # Use the player parameter that's already passed in
        # If we need to determine player from row_label, do it here
        if self.player1 and self.player2:
            player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
            player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
            if player1_initials in row_label:
                player = self.player1
            elif player2_initials in row_label:
                player = self.player2
        
        # Determine serve type context
        if "1st Serve" in row_label:
            serve_type = "first serve"
        elif "2nd Serve" in row_label:
            serve_type = "second serve"
        else:
            serve_type = "serve"
        
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0" and value != "-":
                header = column_headers[i]
                
                if header == "Pts":
                    sentences.append(f"{player} served {value} {serve_type}s.")
                elif header == "1+ shots":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points overall.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points overall.")
                elif header == "2+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 2 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 2 or more shots.")
                elif header == "3+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 3 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 3 or more shots.")
                elif header == "4+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 4 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 4 or more shots.")
                elif header == "5+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 5 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 5 or more shots.")
                elif header == "6+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 6 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 6 or more shots.")
                elif header == "7+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 7 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 7 or more shots.")
                elif header == "8+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 8 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 8 or more shots.")
                elif header == "9+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 9 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 9 or more shots.")
                elif header == "10+":
                    if "%" in value:
                        percentage = value.replace("%", "")
                        sentences.append(f"{player} won {percentage}% of {serve_type} points when rallies reached 10 or more shots.")
                    else:
                        sentences.append(f"{player} won {value} {serve_type} points when rallies reached 10 or more shots.")
        
        return sentences

    def _convert_rallyoutcomes_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str]) -> List[str]:
        """Convert rally outcomes statistics row to natural language sentences"""
        sentences = []
        
        # Handle different rally outcome row types
        if "Total" in row_label:
            context = "Rally outcomes"
        elif "Short" in row_label:
            context = "Short rallies (1-3 shots)"
        elif "Medium" in row_label:
            context = "Medium rallies (4-6 shots)"
        elif "Long" in row_label:
            context = "Long rallies (7+ shots)"
        else:
            context = "Rally outcomes"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "Total":
                    sentences.append(f"{context} occurred {value} times.")
                elif header == "Pts":
                    sentences.append(f"{context} and {value} points were played.")
                else:
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_overview_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str]) -> List[str]:
        """Convert overview statistics row to natural language sentences"""
        sentences = []
        
        # Handle different overview row types
        if "A%" in row_label:
            context = "Ace percentage"
        elif "DF%" in row_label:
            context = "Double fault percentage"
        elif "1stIn" in row_label:
            context = "First serve in percentage"
        elif "1st%" in row_label:
            context = "First serve won percentage"
        elif "2nd%" in row_label:
            context = "Second serve won percentage"
        elif "BPSaved" in row_label:
            context = "Break points saved"
        elif "RPW%" in row_label:
            context = "Return points won percentage"
        elif "Winners" in row_label:
            context = "Winners"
        elif "UFE" in row_label:
            context = "Unforced errors"
        else:
            context = "Overview statistic"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "A%":
                    sentences.append(f"{context} was {value}.")
                elif header == "DF%":
                    sentences.append(f"{context} was {value}.")
                elif header == "1stIn":
                    sentences.append(f"{context} was {value}.")
                elif header == "1st%":
                    sentences.append(f"{context} was {value}.")
                elif header == "2nd%":
                    sentences.append(f"{context} was {value}.")
                elif header == "BPSaved":
                    sentences.append(f"{context} were {value}.")
                elif header == "RPW%":
                    sentences.append(f"{context} was {value}.")
                elif header == "Winners":
                    sentences.append(f"{context} were {value}.")
                elif header == "UFE":
                    sentences.append(f"{context} were {value}.")
                else:
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_generic_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], table_name: str) -> List[str]:
        """Convert generic row to natural language sentences"""
        sentences = []
        
        context = f"In the {table_name} category"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                sentences.append(f"{context}, {row_label}: {header} = {value}.")
        
        return sentences

    def _convert_netpts_row_to_sentences(self, row_label: str, values: List[str], column_headers: List[str], player: str) -> List[str]:
        """Convert net points statistics row to natural language sentences"""
        sentences = []
        
        # Handle different net row types
        if "All Net Points" in row_label:
            context = f"{player} played"
        elif "All Net Approaches" in row_label:
            context = f"{player} made"
        elif "Net Points (excl S-and-V)" in row_label:
            context = f"{player} played"
        elif "Net Approaches (excl S-and-V)" in row_label:
            context = f"{player} made"
        else:
            context = f"{player} played"
        
        # Convert each value to a sentence with better formatting
        for i, value in enumerate(values):
            if i < len(column_headers) and value and value != "0":
                header = column_headers[i]
                
                if header == "Pts":
                    if "All Net Points" in row_label:
                        sentences.append(f"{player} played {value} net points.")
                    elif "Net Points (excl S-and-V)" in row_label:
                        sentences.append(f"{player} played {value} net points excluding serve and volleys.")
                    elif "All Net Approaches" in row_label:
                        sentences.append(f"{player} made {value} net approaches.")
                    elif "Net Approaches (excl S-and-V)" in row_label:
                        sentences.append(f"{player} made {value} net approaches excluding serve and volleys.")
                    else:
                        sentences.append(f"{context} {value} times.")
                elif header == "Won-----%":
                    # Extract the number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} won {number} net points, or {percentage}% of total net points for {player}.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} won {number} net approaches, or {percentage}% of total net approaches for {player}.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} won {number} net points excluding serve and volleys, or {percentage}% of total net points excluding serve and volleys for {player}.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} won {number} net approaches excluding serve and volleys, or {percentage}% of total net approaches excluding serve and volleys for {player}.")
                        else:
                            sentences.append(f"{context} and won {number} points, or {percentage}% of total net points for {player}.")
                    else:
                        sentences.append(f"{context} and won {value} points.")
                elif header == "Wnr at Net":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} hit {number} winners at the net, which represents {percentage}% of net points for {player}.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} hit {number} winners on net approaches, which represents {percentage}% of net approaches for {player}.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} hit {number} winners at the net excluding serve and volleys, which represents {percentage}% of net points for {player}.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} hit {number} winners on net approaches excluding serve and volleys, which represents {percentage}% of net approaches for {player}.")
                        else:
                            sentences.append(f"{player} hit {number} winners at the net, which represents {percentage}% of net points for {player}.")
                    else:
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} hit {value} winners at the net.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} hit {value} winners on net approaches.")
                        else:
                            sentences.append(f"{player} hit {value} winners at the net.")
                elif header == "indFcd at Net":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} induced {number} forced errors at the net, which represents {percentage}% of net points for {player}.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} induced {number} forced errors on net approaches, which represents {percentage}% of net approaches for {player}.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} induced {number} forced errors at the net excluding serve and volleys, which represents {percentage}% of net points for {player}.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} induced {number} forced errors on net approaches excluding serve and volleys, which represents {percentage}% of net approaches for {player}.")
                        else:
                            sentences.append(f"{player} induced {number} forced errors at the net, which represents {percentage}% of net points for {player}.")
                    else:
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} induced {value} forced errors at the net.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} induced {value} forced errors on net approaches.")
                        else:
                            sentences.append(f"{player} induced {value} forced errors at the net.")
                elif header == "UFE at Net":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} made {number} unforced errors at the net, which represents {percentage}% of net points for {player}.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} made {number} unforced errors on net approaches, which represents {percentage}% of net approaches for {player}.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} made {number} unforced errors at the net excluding serve and volleys, which represents {percentage}% of net points for {player}.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} made {number} unforced errors on net approaches excluding serve and volleys, which represents {percentage}% of net approaches for {player}.")
                        else:
                            sentences.append(f"{player} made {number} unforced errors at the net, which represents {percentage}% of net points for {player}.")
                    else:
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} made {value} unforced errors at the net.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} made {value} unforced errors on net approaches.")
                        else:
                            sentences.append(f"{player} made {value} unforced errors at the net.")
                elif header == "Passed at Net":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} was passed {number} times at the net, which represents {percentage}% of net points for {player}.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} was passed {number} times on net approaches, which represents {percentage}% of net approaches for {player}.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} was passed {number} times at the net excluding serve and volleys, which represents {percentage}% of net points for {player}.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} was passed {number} times on net approaches excluding serve and volleys, which represents {percentage}% of net approaches for {player}.")
                        else:
                            sentences.append(f"{player} was passed {number} times at the net, which represents {percentage}% of net points for {player}.")
                    else:
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} was passed {value} times at the net.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} was passed {value} times on net approaches.")
                        else:
                            sentences.append(f"{player} was passed {value} times at the net.")
                elif header == "PsgSht indFcd":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} induced {number} forced errors with passing shots at the net, which represents {percentage}% of net points.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} induced {number} forced errors with passing shots on net approaches, which represents {percentage}% of net approaches.")
                        elif "Net Points (excl S-and-V)" in row_label:
                            sentences.append(f"{player} induced {number} forced errors with passing shots at the net excluding serve and volleys, which represents {percentage}% of net points.")
                        elif "Net Approaches (excl S-and-V)" in row_label:
                            sentences.append(f"{player} induced {number} forced errors with passing shots on net approaches excluding serve and volleys, which represents {percentage}% of net approaches.")
                        else:
                            sentences.append(f"{player} induced {number} forced errors with passing shots at the net, which represents {percentage}% of net points.")
                    else:
                        if "All Net Points" in row_label:
                            sentences.append(f"{player} induced {value} forced errors with passing shots at the net.")
                        elif "All Net Approaches" in row_label:
                            sentences.append(f"{player} induced {value} forced errors with passing shots on net approaches.")
                        else:
                            sentences.append(f"{player} induced {value} forced errors with passing shots at the net.")
                elif header == "rallyLen":
                    if "All Net Points" in row_label:
                        sentences.append(f"{player} played at the net and the average rally length was {value} strokes.")
                    elif "All Net Approaches" in row_label:
                        sentences.append(f"{player} approached the net and the average rally length was {value} strokes.")
                    elif "Net Points (excl S-and-V)" in row_label:
                        sentences.append(f"{player} played at the net excluding serve and volleys and the average rally length was {value} strokes.")
                    elif "Net Approaches (excl S-and-V)" in row_label:
                        sentences.append(f"{player} approached the net excluding serve and volleys and the average rally length was {value} strokes.")
                    else:
                        sentences.append(f"{context} and the average rally length was {value} strokes.")
                else:
                    # Generic fallback
                    sentences.append(f"{context} and {header}: {value}.")
        
        return sentences

    def _convert_details_flat_to_text(self, details_flat: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert details_flat data to natural language text"""
        text = []
        
        # Group the flat data by table type
        shots_data = {}
        shotdir_data = {}
        netpts_data = {}
        serve_data = {}
        return_data = {}
        keypoints_data = {}
        overview_data = {}
        
        for key, value in details_flat.items():
            if key.startswith('shots1') or key.startswith('shots2'):
                shots_data[key] = value
            elif key.startswith('shotdir1') or key.startswith('shotdir2'):
                shotdir_data[key] = value
            elif key.startswith('netpts1') or key.startswith('netpts2'):
                netpts_data[key] = value
            elif key.startswith('serve1') or key.startswith('serve2'):
                serve_data[key] = value
            elif key.startswith('return1') or key.startswith('return2'):
                return_data[key] = value
            elif key.startswith('keypoints'):
                keypoints_data[key] = value
            elif key.startswith('overview'):
                overview_data[key] = value
        
        # Convert serve data (most important - comes first)
        if serve_data:
            text.extend(self._convert_flat_serve_to_text(serve_data, player1, player2))
        
        # Convert return data (second most important)
        if return_data:
            text.extend(self._convert_flat_return_to_text(return_data, player1, player2))
        
        # Convert overview data (summary statistics)
        if overview_data:
            text.extend(self._convert_flat_overview_to_text(overview_data, player1, player2))
        
        # Convert key points data
        if keypoints_data:
            text.extend(self._convert_flat_keypoints_to_text(keypoints_data, player1, player2))
        
        # Convert shots data
        if shots_data:
            text.extend(self._convert_flat_shots_to_text(shots_data, player1, player2))
        
        # Convert shotdir data
        if shotdir_data:
            text.extend(self._convert_flat_shotdir_to_text(shotdir_data, player1, player2))
        
        # Convert netpts data
        if netpts_data:
            text.extend(self._convert_flat_netpts_to_text(netpts_data, player1, player2))
        
        return text

    def _convert_flat_shots_to_text(self, shots_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat shots data to natural language text"""
        text = []
        
        # Group by player
        shots1_data = {k: v for k, v in shots_data.items() if k.startswith('shots1')}
        shots2_data = {k: v for k, v in shots_data.items() if k.startswith('shots2')}
        
        if shots1_data:
            text.append("SHOTS1 STATISTICS:")
            text.append("-" * 20)
            text.extend(self._convert_flat_shots_player_to_text(shots1_data, player1))
            text.append("")
        
        if shots2_data:
            text.append("SHOTS2 STATISTICS:")
            text.append("-" * 20)
            text.extend(self._convert_flat_shots_player_to_text(shots2_data, player2))
            text.append("")
        
        return text

    def _convert_flat_shots_player_to_text(self, shots_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat shots data for one player to natural language text"""
        text = []
        
        # TIER 1 (AUTHORITATIVE): Extract all columns from TOTAL, FOREHAND, and BACKHAND rows
        total_row_values = None
        forehand_row_values = None
        backhand_row_values = None
        row_headers = None
        
        # Find the header row
        header_key = None
        for key in shots_data.keys():
            if 'SHOT TYPES' in key:
                header_key = key
                break
        
        if header_key:
            headers = [h.strip() for h in shots_data[header_key].split(' | ')]
            
            # Extract all columns from TOTAL, FOREHAND, and BACKHAND rows (all are authoritative)
            for key, value in shots_data.items():
                if key != header_key and 'SHOT TYPES' not in key:
                    # Extract the row label (remove the prefix like "shots1 - ")
                    row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Skip if this looks like a header row
                    if any(header_word in row_label for header_word in ['SHOT TYPES', 'PtEnd', 'Winner', 'IndFcd', 'UnfErr', 'SvReturn', 'inPtsW', 'inPtsL']):
                        continue
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Extract all columns from the authoritative rows
                    if len(values) == len(headers) and any(v != '' and v != '0' for v in values):
                        if 'Total' in row_label:
                            total_row_values = values
                            row_headers = headers
                        elif 'Forehand side' in row_label:
                            forehand_row_values = values
                            row_headers = headers
                        elif 'Backhand side' in row_label:
                            backhand_row_values = values
                            row_headers = headers
        
        # Add all authoritative totals from TOTAL, FOREHAND, and BACKHAND rows
        if row_headers:
            text.append(f"AUTHORITATIVE TOTALS FOR {player.upper()} SHOT STATISTICS:")
            
            # Add TOTAL row data
            if total_row_values:
                text.append("TOTAL SHOTS AUTHORITATIVE DATA:")
                for i, (header, value) in enumerate(zip(row_headers, total_row_values)):
                    if value and value != "0":
                        if header == "Total":
                            text.append(f"{player} hit {value} total shots.")
                        elif header == "PtEnd---%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} ended {number} total points with shots.")
                        elif header == "Winner--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} total winners.")
                        elif header == "IndFcd--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} was induced into making {number} total forced errors.")
                        elif header == "UnfErr--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} made {number} total unforced errors.")
                        elif header == "SvReturn":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} total serve returns.")
                        elif header == "inPtsW--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} won {number} total points with shots.")
                        elif header == "inPtsL--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} lost {number} total points with shots.")
            
            # Add FOREHAND row data
            if forehand_row_values:
                text.append("FOREHAND SHOTS AUTHORITATIVE DATA:")
                for i, (header, value) in enumerate(zip(row_headers, forehand_row_values)):
                    if value and value != "0":
                        if header == "Total":
                            text.append(f"{player} hit {value} forehand shots.")
                        elif header == "PtEnd---%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} ended {number} points with forehand shots.")
                        elif header == "Winner--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} forehand winners.")
                        elif header == "IndFcd--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} was induced into making {number} forced errors with forehand shots.")
                        elif header == "UnfErr--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} made {number} forehand unforced errors.")
                        elif header == "SvReturn":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} forehand serve returns.")
                        elif header == "inPtsW--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} won {number} points with forehand shots.")
                        elif header == "inPtsL--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} lost {number} points with forehand shots.")
            
            # Add BACKHAND row data
            if backhand_row_values:
                text.append("BACKHAND SHOTS AUTHORITATIVE DATA:")
                for i, (header, value) in enumerate(zip(row_headers, backhand_row_values)):
                    if value and value != "0":
                        if header == "Total":
                            text.append(f"{player} hit {value} backhand shots.")
                        elif header == "PtEnd---%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} ended {number} points with backhand shots.")
                        elif header == "Winner--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} backhand winners.")
                        elif header == "IndFcd--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} was induced into making {number} forced errors with backhand shots.")
                        elif header == "UnfErr--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} made {number} backhand unforced errors.")
                        elif header == "SvReturn":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} hit {number} backhand serve returns.")
                        elif header == "inPtsW--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} won {number} points with backhand shots.")
                        elif header == "inPtsL--%":
                            number = value.split('(')[0].strip()
                            text.append(f"{player} lost {number} points with backhand shots.")
            
            text.append("")
            text.append("DETAILED BREAKDOWN (these are contextual details, not for recalculating totals):")
            text.append("")
        
        # Second pass: process all rows for detailed breakdown
        if header_key:
            headers = [h.strip() for h in shots_data[header_key].split(' | ')]
            
            # Process each data row (skip the header row)
            for key, value in shots_data.items():
                if key != header_key and 'SHOT TYPES' not in key:
                    # Extract the row label (remove the prefix like "shots1 - ")
                    row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Skip if this looks like a header row
                    if any(header_word in row_label for header_word in ['SHOT TYPES', 'Total', 'PtEnd', 'Winner', 'IndFcd', 'UnfErr', 'SvReturn', 'inPtsW', 'inPtsL']):
                        continue
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Only process if we have valid data (not just headers)
                    if len(values) == len(headers) and any(v != '' and v != '0' for v in values):
                        # Convert to sentences using the existing method
                        sentences = self._convert_shots_row_to_sentences(row_label, values, headers, player)
                        text.extend(sentences)
        
        return text

    def _convert_flat_shotdir_to_text(self, shotdir_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat shotdir data to natural language text"""
        text = []
        
        # Group by player
        shotdir1_data = {k: v for k, v in shotdir_data.items() if k.startswith('shotdir1')}
        shotdir2_data = {k: v for k, v in shotdir_data.items() if k.startswith('shotdir2')}
        
        if shotdir1_data:
            text.append("SHOTDIR1 STATISTICS:")
            text.append("-" * 22)
            text.extend(self._convert_flat_shotdir_player_to_text(shotdir1_data, player1))
            text.append("")
        
        if shotdir2_data:
            text.append("SHOTDIR2 STATISTICS:")
            text.append("-" * 22)
            text.extend(self._convert_flat_shotdir_player_to_text(shotdir2_data, player2))
            text.append("")
        
        # Add explicit totals for complex shot direction + outcome combinations
        # These help the LLM anchor on correct totals for complex questions
        text.append("")
        text.append("EXPLICIT TOTALS FOR SHOT DIRECTION + OUTCOME COMBINATIONS:")
        text.append("-" * 50)
        
        # Calculate explicit totals from the natural language sentences we just generated
        directions = ['crosscourt', 'down middle', 'down the line', 'inside-out', 'inside-in']
        outcomes = ['winners', 'induced forced errors', 'unforced errors', 'points won', 'points lost']
        
        # Create a nested dictionary to store all combinations for both players
        player_totals = {player1: {}, player2: {}}
        for player in player_totals:
            player_totals[player] = {}
            for direction in directions:
                player_totals[player][direction] = {}
                for outcome in outcomes:
                    player_totals[player][direction][outcome] = {'forehand': 0, 'backhand': 0, 'slice': 0}
        
        # Parse the natural language sentences to extract numbers
        for line in text:
            # Extract the number from the beginning of the line
            import re
            number_match = re.search(r'(\d+)', line)
            if not number_match:
                continue
            number = int(number_match.group(1))
            
            # Determine which player this line is for
            current_player = None
            if player1 in line:
                current_player = player1
            elif player2 in line:
                current_player = player2
            else:
                continue
            
            # Look for lines like "hit X winners with [shot type] [direction] shots"
            if 'hit' in line and 'winners' in line and 'shots' in line:
                for direction in directions:
                    if direction in line:
                        if 'forehand' in line:
                            player_totals[current_player][direction]['winners']['forehand'] = number
                        elif 'backhand' in line and 'slice' not in line:
                            player_totals[current_player][direction]['winners']['backhand'] = number
                        elif 'slice' in line:
                            player_totals[current_player][direction]['winners']['slice'] = number
                        break
            
            elif 'made' in line and 'induced forced errors' in line and 'shots' in line:
                for direction in directions:
                    if direction in line:
                        if 'forehand' in line:
                            player_totals[current_player][direction]['induced forced errors']['forehand'] = number
                        elif 'backhand' in line and 'slice' not in line:
                            player_totals[current_player][direction]['induced forced errors']['backhand'] = number
                        elif 'slice' in line:
                            player_totals[current_player][direction]['induced forced errors']['slice'] = number
                        break
            
            elif 'made' in line and 'unforced errors' in line and 'shots' in line:
                for direction in directions:
                    if direction in line:
                        if 'forehand' in line:
                            player_totals[current_player][direction]['unforced errors']['forehand'] = number
                        elif 'backhand' in line and 'slice' not in line:
                            player_totals[current_player][direction]['unforced errors']['backhand'] = number
                        elif 'slice' in line:
                            player_totals[current_player][direction]['unforced errors']['slice'] = number
                        break
            
            elif 'won' in line and 'points' in line and 'shots' in line:
                for direction in directions:
                    if direction in line:
                        if 'forehand' in line:
                            player_totals[current_player][direction]['points won']['forehand'] = number
                        elif 'backhand' in line and 'slice' not in line:
                            player_totals[current_player][direction]['points won']['backhand'] = number
                        elif 'slice' in line:
                            player_totals[current_player][direction]['points won']['slice'] = number
                        break
            
            elif 'lost' in line and 'points' in line and 'shots' in line:
                for direction in directions:
                    if direction in line:
                        if 'forehand' in line:
                            player_totals[current_player][direction]['points lost']['forehand'] = number
                        elif 'backhand' in line and 'slice' not in line:
                            player_totals[current_player][direction]['points lost']['backhand'] = number
                        elif 'slice' in line:
                            player_totals[current_player][direction]['points lost']['slice'] = number
                        break
        
        # Generate explicit totals for ALL combinations (including zeros) for both players
        for player in [self.player1, self.player2] if self.player1 and self.player2 else ['Player 1', 'Player 2']:
            for direction in directions:
                for outcome in outcomes:
                    total = sum(player_totals[player][direction][outcome].values())
                    fh, bh, sl = player_totals[player][direction][outcome].values()
                    if outcome == 'winners':
                        text.append(f"{player} hit {total} {direction} {outcome} total, including {fh} forehand {direction} {outcome}, {bh} backhand {direction} {outcome}, and {sl} slice {direction} {outcome}")
                    elif outcome == 'induced forced errors':
                        text.append(f"{player} made {total} {direction} {outcome} total, including {fh} forehand {direction} {outcome}, {bh} backhand {direction} {outcome}, and {sl} slice {direction} {outcome}")
                    elif outcome == 'unforced errors':
                        text.append(f"{player} made {total} {direction} {outcome} total, including {fh} forehand {direction} {outcome}, {bh} backhand {direction} {outcome}, and {sl} slice {direction} {outcome}")
                    elif outcome == 'points won':
                        text.append(f"{player} won {total} {direction} {outcome} total, including {fh} forehand {direction} {outcome}, {bh} backhand {direction} {outcome}, and {sl} slice {direction} {outcome}")
                    elif outcome == 'points lost':
                        text.append(f"{player} lost {total} {direction} {outcome} total, including {fh} forehand {direction} {outcome}, {bh} backhand {direction} {outcome}, and {sl} slice {direction} {outcome}")
        
        text.append("")
        
        return text

    def _convert_flat_shotdir_player_to_text(self, shotdir_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat shotdir data for one player to natural language text"""
        text = []
        
        # Group by table type
        table1_data = {k: v for k, v in shotdir_data.items() if '_table1' in k}
        table2_data = {k: v for k, v in shotdir_data.items() if '_table2' in k}
        
        # TIER 1 (AUTHORITATIVE): Extract totals from "Total" row as single source of truth
        total_shots_all_directions = 0
        total_forehand_shots = 0
        total_backhand_shots = 0
        total_backhand_slice_shots = 0
        
        # Directional totals
        total_crosscourt_shots = 0
        total_down_the_line_shots = 0
        total_down_the_middle_shots = 0
        total_inside_out_shots = 0
        total_inside_in_shots = 0
        
        # Extract authoritative totals from table1 rows
        if table1_data:
            header_key = None
            for key in table1_data.keys():
                if 'SHOT DIRECTION' in key:
                    header_key = key
                    break
            
            if header_key:
                headers = [h.strip() for h in table1_data[header_key].split(' | ')]
                
                for key, value in table1_data.items():
                    if key != header_key and 'SHOT DIRECTION' not in key:
                        row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                        
                        # Skip header rows except Total
                        if any(header_word in row_label for header_word in ['SHOT DIRECTION', 'PtEnding', 'Winner', 'InduceFcd', 'UnfErr', 'inPtsWon', 'inPtsLost']):
                            continue
                        
                        values = [v.strip() for v in value.split(' | ')]
                        
                        # Extract totals from each row
                        if len(values) == len(headers) and any(v != '' and v != '0' for v in values):
                            try:
                                # Sum up all direction values for this shot type
                                shot_total = sum(int(v.split('(')[0].strip()) for v in values if v and '(' in v and v.split('(')[0].strip().isdigit())
                                
                                if 'Total' in row_label:
                                    total_shots_all_directions = shot_total  # AUTHORITATIVE
                                    
                                    # Extract directional totals from the Total row
                                    for i, header in enumerate(headers):
                                        if i < len(values) and values[i] and '(' in values[i]:
                                            direction_value = int(values[i].split('(')[0].strip())
                                            if 'Crosscourt' in header:
                                                total_crosscourt_shots = direction_value
                                            elif 'Down the line' in header:
                                                total_down_the_line_shots = direction_value
                                            elif 'Down middle' in header:
                                                total_down_the_middle_shots = direction_value
                                            elif 'Inside-out' in header:
                                                total_inside_out_shots = direction_value
                                            elif 'Inside-in' in header:
                                                total_inside_in_shots = direction_value
                                
                                elif 'Forehand' in row_label:
                                    total_forehand_shots = shot_total
                                elif 'BH slice' in row_label:
                                    total_backhand_slice_shots = shot_total
                                elif 'Backhand' in row_label:
                                    total_backhand_shots = shot_total
                            except (ValueError, IndexError):
                                continue
        
        # Add authoritative summary (TIER 1)
        if total_shots_all_directions > 0:
            text.append(f"AUTHORITATIVE TOTALS FOR {player.upper()} SHOT DIRECTIONS:")
            text.append(f"{player} hit {total_shots_all_directions} total shots in all directions including crosscourt, down the middle, down the line, inside-out, and inside-in.")
            text.append("")
            text.append("SHOT TYPE BREAKDOWN (these sum to total above, do not add again):")
            if total_forehand_shots > 0:
                text.append(f"{player} hit {total_forehand_shots} forehand shots in all directions including crosscourt, down the middle, down the line, inside-out, and inside-in.")
            if total_backhand_shots > 0:
                text.append(f"{player} hit {total_backhand_shots} backhand shots in all directions including crosscourt, down the middle, down the line, inside-out, and inside-in.")
            if total_backhand_slice_shots > 0:
                text.append(f"{player} hit {total_backhand_slice_shots} backhand slice shots in all directions including crosscourt, down the middle, down the line, inside-out, and inside-in.")
            text.append("")
            text.append("AUTHORITATIVE TOTALS BY DIRECTION (these sum to total above, do not add again):")
            text.append(f"{player} hit {total_crosscourt_shots} total crosscourt shots.")
            text.append(f"{player} hit {total_inside_out_shots} total inside-out shots.")
            text.append(f"{player} hit {total_down_the_line_shots} total down the line shots.")
            text.append(f"{player} hit {total_down_the_middle_shots} total down the middle shots.")
            text.append(f"{player} hit {total_inside_in_shots} total inside-in shots.")
            text.append("")
        
        # Process table1 (direction breakdowns)
        if table1_data:
            text.append("SHOT DIRECTION BREAKDOWN:")
            text.append("-" * 25)
            
            # Find the header row for table1
            header_key = None
            for key in table1_data.keys():
                if 'SHOT DIRECTION' in key:
                    header_key = key
                    break
            
            if header_key:
                headers = [h.strip() for h in table1_data[header_key].split(' | ')]
                
                # Process each data row in table1
                for key, value in table1_data.items():
                    if key != header_key and 'SHOT DIRECTION' not in key:
                        # Extract the row label
                        row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                        
                        # Skip if this looks like a header row
                        if any(header_word in row_label for header_word in ['SHOT DIRECTION', 'Total', 'PtEnding', 'Winner', 'InduceFcd', 'UnfErr', 'inPtsWon', 'inPtsLost']):
                            continue
                        
                        # Split the value by | to get individual values
                        values = [v.strip() for v in value.split(' | ')]
                        
                        # Only process if we have valid data
                        if len(values) == len(headers) and any(v != '' and v != '0' for v in values):
                            # Convert to sentences for table1
                            sentences = self._convert_shotdir_table1_row_to_sentences(row_label, values, headers, player)
                            text.extend(sentences)
            
            text.append("")
        
        # Process table2 (detailed breakdowns by shot type and direction)
        if table2_data:
            text.append("SHOT DIRECTION DETAILED BREAKDOWN:")
            text.append("-" * 35)
            
            # Find the header row for table2
            header_key = None
            for key in table2_data.keys():
                if 'SHOT DIRECTION' in key:
                    header_key = key
                    break
            
            if header_key:
                headers = [h.strip() for h in table2_data[header_key].split(' | ')]
                
                # Process each data row in table2
                for key, value in table2_data.items():
                    if key != header_key and 'SHOT DIRECTION' not in key:
                        # Extract the row label
                        row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                        
                        # Skip if this looks like a header row
                        if any(header_word in row_label for header_word in ['SHOT DIRECTION', 'Total', 'PtEnding', 'Winner', 'InduceFcd', 'UnfErr', 'inPtsWon', 'inPtsLost']):
                            continue
                        
                        # Split the value by | to get individual values
                        values = [v.strip() for v in value.split(' | ')]
                        
                        # Only process if we have valid data
                        if len(values) == len(headers) and any(v != '' and v != '0' for v in values):
                            # Convert to sentences for table2
                            sentences = self._convert_shotdir_table2_row_to_sentences(row_label, values, headers, player)
                            text.extend(sentences)
        

        
        return text

    def _convert_shotdir_table1_row_to_sentences(self, row_label: str, values: List[str], headers: List[str], player: str) -> List[str]:
        """Convert shot direction table1 row to natural language sentences"""
        sentences = []
        
        # Handle different row types
        if "Total" in row_label:
            # Total row shows breakdown by direction
            for i, value in enumerate(values):
                if i < len(headers) and value and value != "0":
                    header = headers[i]
                    
                    # Extract number and percentage from format like "57  (48%)"
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        
                        if header == "Crosscourt":
                            sentences.append(f"{player} hit {number} crosscourt shots, which represents {percentage}% of all shots hit by {player}.")
                        elif header == "Down middle":
                            sentences.append(f"{player} hit {number} down-the-middle shots, which represents {percentage}% of all shots hit by {player}.")
                        elif header == "Down the line":
                            sentences.append(f"{player} hit {number} down-the-line shots, which represents {percentage}% of all shots hit by {player}.")
                        elif header == "Inside-out":
                            sentences.append(f"{player} hit {number} inside-out shots, which represents {percentage}% of all shots hit by {player}.")
                        elif header == "Inside-in":
                            sentences.append(f"{player} hit {number} inside-in shots, which represents {percentage}% of all shots hit by {player}.")
                        else:
                            sentences.append(f"{player} hit {number} {header.lower()} shots, which represents {percentage}% of all shots hit by {player}.")
                    else:
                        sentences.append(f"{player} hit {value} {header.lower()} shots.")
        else:
            # Other rows (Forehand, Backhand, BH slice) show breakdown by direction for that shot type
            for i, value in enumerate(values):
                if i < len(headers) and value and value != "0":
                    header = headers[i]
                    
                    # Extract number and percentage from format like "57  (48%)"
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        
                        if "Forehand" in row_label:
                            if header == "Crosscourt":
                                sentences.append(f"{player} hit {number} forehand crosscourt shots, which represents {percentage}% of all forehand shots hit by {player}.")
                            elif header == "Down middle":
                                sentences.append(f"{player} hit {number} forehand down-the-middle shots, which represents {percentage}% of all forehand shots hit by {player}.")
                            elif header == "Down the line":
                                sentences.append(f"{player} hit {number} forehand down-the-line shots, which represents {percentage}% of all forehand shots hit by {player}.")
                            elif header == "Inside-out":
                                sentences.append(f"{player} hit {number} forehand inside-out shots, which represents {percentage}% of all forehand shots hit by {player}.")
                            elif header == "Inside-in":
                                sentences.append(f"{player} hit {number} forehand inside-in shots, which represents {percentage}% of all forehand shots hit by {player}.")
                        elif "Backhand" in row_label:
                            if header == "Crosscourt":
                                sentences.append(f"{player} hit {number} backhand crosscourt shots, which represents {percentage}% of all backhand shots hit by {player}.")
                            elif header == "Down middle":
                                sentences.append(f"{player} hit {number} backhand down-the-middle shots, which represents {percentage}% of all backhand shots hit by {player}.")
                            elif header == "Down the line":
                                sentences.append(f"{player} hit {number} backhand down-the-line shots, which represents {percentage}% of all backhand shots hit by {player}.")
                            elif header == "Inside-out":
                                sentences.append(f"{player} hit {number} backhand inside-out shots, which represents {percentage}% of all backhand shots hit by {player}.")
                            elif header == "Inside-in":
                                sentences.append(f"{player} hit {number} backhand inside-in shots, which represents {percentage}% of all backhand shots hit by {player}.")
                        elif "BH slice" in row_label:
                            if header == "Crosscourt":
                                sentences.append(f"{player} hit {number} backhand slice crosscourt shots, which represents {percentage}% of all backhand slice shots hit by {player}.")
                            elif header == "Down middle":
                                sentences.append(f"{player} hit {number} backhand slice down-the-middle shots, which represents {percentage}% of all backhand slice shots hit by {player}.")
                            elif header == "Down the line":
                                sentences.append(f"{player} hit {number} backhand slice down-the-line shots, which represents {percentage}% of all backhand slice shots hit by {player}.")
                            elif header == "Inside-out":
                                sentences.append(f"{player} hit {number} backhand slice inside-out shots, which represents {percentage}% of all backhand slice shots hit by {player}.")
                            elif header == "Inside-in":
                                sentences.append(f"{player} hit {number} backhand slice inside-in shots, which represents {percentage}% of all backhand slice shots hit by {player}.")
                    else:
                        if "Forehand" in row_label:
                            sentences.append(f"{player} hit {value} forehand {header.lower()} shots.")
                        elif "Backhand" in row_label:
                            sentences.append(f"{player} hit {value} backhand {header.lower()} shots.")
                        elif "BH slice" in row_label:
                            sentences.append(f"{player} hit {value} backhand slice {header.lower()} shots.")
        
        return sentences

    def _convert_shotdir_table2_row_to_sentences(self, row_label: str, values: List[str], headers: List[str], player: str) -> List[str]:
        """Convert shot direction table2 row to natural language sentences"""
        sentences = []
        
        # Extract shot type and direction from row label (e.g., "FH crosscourt_table2" -> "forehand crosscourt")
        shot_type = row_label.lower()
        
        # Remove table suffix if present
        if "_table" in shot_type:
            shot_type = shot_type.split("_table")[0]
        
        # Convert abbreviations
        if "FH" in shot_type:
            shot_type = shot_type.replace("FH", "forehand")
        if "BH" in shot_type:
            shot_type = shot_type.replace("BH", "backhand")
        if "fh" in shot_type:
            shot_type = shot_type.replace("fh", "forehand")
        if "bh" in shot_type:
            shot_type = shot_type.replace("bh", "backhand")
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(headers) and value and value != "0":
                header = headers[i]
                
                if header == "Total":
                    sentences.append(f"{player} hit {value} {shot_type} shots.")
                elif header == "PtEnding":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} ended {number} points with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} ended {value} points with {shot_type} shots.")
                elif header == "Winner":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} hit {number} winners with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} hit {value} winners with {shot_type} shots.")
                elif header == "InduceFcd":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} made {number} induced forced errors with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} made {value} induced forced errors with {shot_type} shots.")
                elif header == "UnfErr":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} made {number} unforced errors with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} made {value} unforced errors with {shot_type} shots.")
                elif header == "inPtsWon":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} won {number} points with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} won {value} points with {shot_type} shots.")
                elif header == "inPtsLost":
                    # Extract number and percentage
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} lost {number} points with {shot_type} shots, which represents {percentage}% of all {shot_type} shots hit by {player}.")
                    else:
                        sentences.append(f"{player} lost {value} points with {shot_type} shots.")
                else:
                    sentences.append(f"{player} had {header}: {value} with {shot_type} shots.")
        
        return sentences

    def _convert_flat_netpts_to_text(self, netpts_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat netpts data to natural language text"""
        text = []
        
        # Group by player
        netpts1_data = {k: v for k, v in netpts_data.items() if k.startswith('netpts1')}
        netpts2_data = {k: v for k, v in netpts_data.items() if k.startswith('netpts2')}
        
        if netpts1_data:
            text.append("NETPTS1 STATISTICS:")
            text.append("-" * 21)
            text.extend(self._convert_flat_netpts_player_to_text(netpts1_data, player1))
            text.append("")
        
        if netpts2_data:
            text.append("NETPTS2 STATISTICS:")
            text.append("-" * 21)
            text.extend(self._convert_flat_netpts_player_to_text(netpts2_data, player2))
            text.append("")
        
        return text

    def _convert_flat_netpts_player_to_text(self, netpts_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat netpts data for one player to natural language text"""
        text = []
        
        # Find the header row
        header_key = None
        for key in netpts_data.keys():
            if 'NET POINTS' in key:
                header_key = key
                break
        
        if header_key:
            headers = netpts_data[header_key].split(' | ')
            
            # Process each data row
            for key, value in netpts_data.items():
                if key != header_key and 'NET POINTS' not in key:
                    # Extract the row label
                    row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Convert to sentences using the existing method
                    sentences = self._convert_netpts_row_to_sentences(row_label, values, headers, player)
                    text.extend(sentences)
        
        return text

    def _convert_flat_serve_to_text(self, serve_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat serve data to natural language text"""
        text = []
        
        # Group by player
        serve1_data = {k: v for k, v in serve_data.items() if k.startswith('serve1')}
        serve2_data = {k: v for k, v in serve_data.items() if k.startswith('serve2')}
        
        # Serve1 Table 1 (Summary)
        if serve1_data:
            text.append("SERVE1 STATISTICS (SUMMARY):")
            text.append("-" * 28)
            text.extend(self._convert_flat_serve_player_table1_to_text(serve1_data, player1))
            text.append("")
        
        # Serve1 Table 2 (Detailed)
        if serve1_data:
            text.append("SERVE1 STATISTICS (DETAILED):")
            text.append("-" * 30)
            text.extend(self._convert_flat_serve_player_table2_to_text(serve1_data, player1))
            text.append("")
        
        # Serve2 Table 1 (Summary)
        if serve2_data:
            text.append("SERVE2 STATISTICS (SUMMARY):")
            text.append("-" * 28)
            text.extend(self._convert_flat_serve_player_table1_to_text(serve2_data, player2))
            text.append("")
        
        # Serve2 Table 2 (Detailed)
        if serve2_data:
            text.append("SERVE2 STATISTICS (DETAILED):")
            text.append("-" * 30)
            text.extend(self._convert_flat_serve_player_table2_to_text(serve2_data, player2))
            text.append("")
        
        return text

    def _convert_flat_serve_player_table1_to_text(self, serve_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat serve data table1 (summary) for one player to natural language text"""
        text = []
        
        # TIER 1 (AUTHORITATIVE): Calculate summary totals first
        total_aces = 0
        total_double_faults = 0
        total_points_served = 0
        total_points_won = 0
        
        # Extract authoritative totals ONLY from Deuce Court and Ad Court (non-overlapping categories)
        for key, value in serve_data.items():
            if '_table1' in key and ' - ' in key:
                location_part = key.split(' - ')[1].split('_table1')[0]
                # Only use Deuce Court and Ad Court as authoritative sources (they don't overlap)
                if location_part in ['Deuce Court', 'Ad Court']:
                    values = [v.strip() for v in value.split(' | ')]
                    if len(values) >= 8:
                        try:
                            total_points_served += int(values[0]) if values[0] and values[0] != "0" else 0
                            if "(" in values[1]:
                                total_points_won += int(values[1].split('(')[0].strip()) if values[1].split('(')[0].strip() else 0
                            total_aces += int(values[2].split('(')[0].strip()) if values[2] and "(" in values[2] else (int(values[2]) if values[2] and values[2] != "0" else 0)
                            total_double_faults += int(values[7].split('(')[0].strip()) if values[7] and "(" in values[7] else (int(values[7]) if values[7] and values[7] != "0" else 0)
                        except (ValueError, IndexError):
                            continue
        
        # Add authoritative summary (TIER 1)
        text.append(f"AUTHORITATIVE TOTALS FOR {player.upper()}:")
        text.append(f"{player} served {total_points_served} total points.")
        text.append(f"{player} won {total_points_won} total serve points.")
        text.append(f"{player} hit {total_aces} total aces.")
        text.append(f"{player} made {total_double_faults} total double faults.")
        text.append("")
        text.append("BREAKDOWN BY COURT LOCATION (these sum to totals above, do not add again):")
        text.append("")
        
        # Process each serve breakdown for table1 (summary)
        for key, value in serve_data.items():
            # Only process table1 data
            if '_table1' not in key:
                continue
                
            # Extract the location from the key
            # Format: "serve1 - Deuce Court_table1"
            if ' - ' in key and '_table1' in key:
                location_part = key.split(' - ')[1].split('_table1')[0]
                
                # Skip header rows
                if 'BREAKDOWN' in location_part:
                    continue
                
                # Split the value by | to get individual values
                values = [v.strip() for v in value.split(' | ')]
                
                # Convert location to proper description
                if 'Deuce Court' in location_part:
                    location_desc = "to the Deuce court"
                elif 'Ad Court' in location_part:
                    location_desc = "to the Ad court"
                else:
                    location_desc = location_part.lower()
                
                # Extract stats for table1 (summary)
                if len(values) >= 8:
                    total_pts = values[0]
                    won_pts = values[1]
                    aces = values[2]
                    unreturned = values[3]
                    forced_errors = values[4]
                    won_3_or_less = values[5]
                    first_serves_in = values[6]
                    double_faults = values[7]
                    
                    # Create detailed descriptions
                    if total_pts and total_pts != "0":
                        text.append(f"{player} served {location_desc} {total_pts} times.")
                        
                        # Points won
                        if won_pts and won_pts != "0":
                            if "(" in won_pts and ")" in won_pts:
                                percentage = won_pts.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} served {location_desc} and won {won_pts.split('(')[0].strip()} points, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} served {location_desc} and won {won_pts} points, or {won_pts}% of total points served {location_desc} by {player}.")
                        
                        # Aces
                        if aces and aces != "0":
                            if "(" in aces and ")" in aces:
                                percentage = aces.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit {aces.split('(')[0].strip()} aces served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit {aces} aces served {location_desc}, or {aces}% of total points served {location_desc} by {player}.")
                        
                        # Unreturned serves
                        if unreturned and unreturned != "0":
                            if "(" in unreturned and ")" in unreturned:
                                percentage = unreturned.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} had {unreturned.split('(')[0].strip()} unreturned serves served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} had {unreturned} unreturned serves served {location_desc}, or {unreturned}% of total points served {location_desc} by {player}.")
                        
                        # Forced errors
                        if forced_errors and forced_errors != "0":
                            if "(" in forced_errors and ")" in forced_errors:
                                percentage = forced_errors.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} forced {forced_errors.split('(')[0].strip()} errors from her opponent on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} forced {forced_errors} errors from her opponent on serves {location_desc}, or {forced_errors}% of total points served {location_desc} by {player}.")
                        
                        # Won in 3 shots or less
                        if won_3_or_less and won_3_or_less != "0":
                            if "(" in won_3_or_less and ")" in won_3_or_less:
                                percentage = won_3_or_less.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {won_3_or_less.split('(')[0].strip()} points in 3 shots or less on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {won_3_or_less} points in 3 shots or less on serves {location_desc}, or {won_3_or_less}% of total points served {location_desc} by {player}.")
                        
                        # First serves in
                        if first_serves_in and first_serves_in != "0":
                            if "(" in first_serves_in and ")" in first_serves_in:
                                percentage = first_serves_in.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit in {first_serves_in.split('(')[0].strip()} first serves served {location_desc}, or {percentage}% of first serves served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit in {first_serves_in} first serves served {location_desc}, or {first_serves_in}% of first serves served {location_desc} by {player}.")
                        
                        # Double faults
                        if double_faults and double_faults != "0":
                            if "(" in double_faults and ")" in double_faults:
                                percentage = double_faults.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} served {location_desc} and had {double_faults.split('(')[0].strip()} double faults, or {percentage}% of serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} served {location_desc} and had {double_faults} double faults, or {double_faults}% of serves {location_desc} by {player}.")
        
        return text

    def _convert_flat_serve_player_table2_to_text(self, serve_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat serve data table2 (detailed) for one player to natural language text"""
        text = []
        
        # Process each serve breakdown for table2 (detailed)
        for key, value in serve_data.items():
            # Only process table2 data
            if '_table2' not in key:
                continue
                
            # Extract the location from the key
            # Format: "serve1 - Deuce Court_table2"
            if ' - ' in key and '_table2' in key:
                location_part = key.split(' - ')[1].split('_table2')[0]
                
                # Skip header rows
                if 'BREAKDOWN' in location_part:
                    continue
                
                # Split the value by | to get individual values
                values = [v.strip() for v in value.split(' | ')]
                
                # Convert location to proper description
                if 'Deuce Court' in location_part:
                    location_desc = "to the Deuce court"
                elif 'Ad Court' in location_part:
                    location_desc = "to the Ad court"
                else:
                    location_desc = location_part.lower()
                
                # Extract stats for table2 (detailed - separates first and second serves)
                if len(values) >= 12:
                    # First serve stats
                    first_pts = values[0]
                    first_won = values[1]
                    first_aces = values[2]
                    first_unreturned = values[3]
                    first_forced_errors = values[4]
                    first_won_3_or_less = values[5]
                    
                    # Second serve stats
                    second_pts = values[6]
                    second_won = values[7]
                    second_aces = values[8]
                    second_unreturned = values[9]
                    second_forced_errors = values[10]
                    second_won_3_or_less = values[11]
                    
                    # First serve details
                    if first_pts and first_pts != "0":
                        text.append(f"{player} served {location_desc} {first_pts} first serves.")
                        
                        if first_won and first_won != "0":
                            if "(" in first_won and ")" in first_won:
                                percentage = first_won.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {first_won.split('(')[0].strip()} first serve points {location_desc}, or {percentage}% of first serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {first_won} first serve points {location_desc}, or {first_won}% of first serves {location_desc} by {player}.")
                        
                        if first_aces and first_aces != "0":
                            if "(" in first_aces and ")" in first_aces:
                                percentage = first_aces.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit {first_aces.split('(')[0].strip()} aces on first serves {location_desc}, or {percentage}% of first serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit {first_aces} aces on first serves {location_desc}, or {first_aces}% of first serves {location_desc} by {player}.")
                        
                        if first_won_3_or_less and first_won_3_or_less != "0":
                            if "(" in first_won_3_or_less and ")" in first_won_3_or_less:
                                percentage = first_won_3_or_less.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {first_won_3_or_less.split('(')[0].strip()} first serve points in 3 shots or less {location_desc}, or {percentage}% of first serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {first_won_3_or_less} first serve points in 3 shots or less {location_desc}, or {first_won_3_or_less}% of first serves {location_desc} by {player}.")
                    
                    # Second serve details
                    if second_pts and second_pts != "0":
                        text.append(f"{player} served {location_desc} {second_pts} second serves.")
                        
                        if second_won and second_won != "0":
                            if "(" in second_won and ")" in second_won:
                                percentage = second_won.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {second_won.split('(')[0].strip()} second serve points {location_desc}, or {percentage}% of second serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {second_won} second serve points {location_desc}, or {second_won}% of second serves {location_desc} by {player}.")
                        
                        if second_aces and second_aces != "0":
                            if "(" in second_aces and ")" in second_aces:
                                percentage = second_aces.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit {second_aces.split('(')[0].strip()} aces on second serves {location_desc}, or {percentage}% of second serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit {second_aces} aces on second serves {location_desc}, or {second_aces}% of second serves {location_desc} by {player}.")
                        
                        if second_won_3_or_less and second_won_3_or_less != "0":
                            if "(" in second_won_3_or_less and ")" in second_won_3_or_less:
                                percentage = second_won_3_or_less.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {second_won_3_or_less.split('(')[0].strip()} second serve points in 3 shots or less {location_desc}, or {percentage}% of second serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {second_won_3_or_less} second serve points in 3 shots or less {location_desc}, or {second_won_3_or_less}% of second serves {location_desc} by {player}.")
        
        return text

    def _convert_flat_serve_player_to_text(self, serve_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat serve data for one player to natural language text"""
        text = []
        
        # Process each serve breakdown
        for key, value in serve_data.items():
            # Extract the location and table type from the key
            # Format: "serve1 - Deuce Court_table1" or "serve1 - Ad Court_table2"
            if ' - ' in key and '_table' in key:
                location_part = key.split(' - ')[1].split('_table')[0]
                table_type = key.split('_table')[1]
                
                # Skip header rows
                if 'BREAKDOWN' in location_part:
                    continue
                
                # Split the value by | to get individual values
                values = [v.strip() for v in value.split(' | ')]
                
                # Determine serve type based on table type
                if table_type == '1':
                    serve_type = "first serves"
                elif table_type == '2':
                    serve_type = "second serves"
                else:
                    serve_type = "serves"
                
                # Convert location to proper description
                if 'Deuce Court' in location_part:
                    location_desc = "to the Deuce court"
                elif 'Ad Court' in location_part:
                    location_desc = "to the Ad court"
                else:
                    location_desc = location_part.lower()
                
                # Extract stats (similar to the serve table conversion)
                if len(values) >= 8:
                    total_pts = values[0]
                    won_pts = values[1]
                    aces = values[2]
                    unreturned = values[3]
                    forced_errors = values[4]
                    won_3_or_less = values[5]
                    first_serves_in = values[6]
                    double_faults = values[7]
                    
                    # Create detailed descriptions
                    if total_pts and total_pts != "0":
                        text.append(f"{player} served {location_desc} {total_pts} times ({serve_type}).")
                        
                        # Points won
                        if won_pts and won_pts != "0":
                            if "(" in won_pts and ")" in won_pts:
                                percentage = won_pts.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} served {location_desc} and won {won_pts.split('(')[0].strip()} points, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} served {location_desc} and won {won_pts} points, or {won_pts}% of total points served {location_desc} by {player}.")
                        
                        # Aces
                        if aces and aces != "0":
                            if "(" in aces and ")" in aces:
                                percentage = aces.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit {aces.split('(')[0].strip()} aces served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit {aces} aces served {location_desc}, or {aces}% of total points served {location_desc} by {player}.")
                        
                        # Unreturned serves
                        if unreturned and unreturned != "0":
                            if "(" in unreturned and ")" in unreturned:
                                percentage = unreturned.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} had {unreturned.split('(')[0].strip()} unreturned serves served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} had {unreturned} unreturned serves served {location_desc}, or {unreturned}% of total points served {location_desc} by {player}.")
                        
                        # Forced errors
                        if forced_errors and forced_errors != "0":
                            if "(" in forced_errors and ")" in forced_errors:
                                percentage = forced_errors.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} forced {forced_errors.split('(')[0].strip()} errors from her opponent on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} forced {forced_errors} errors from her opponent on serves {location_desc}, or {forced_errors}% of total points served {location_desc} by {player}.")
                        
                        # Points won in 3 shots or less
                        if won_3_or_less and won_3_or_less != "0":
                            if "(" in won_3_or_less and ")" in won_3_or_less:
                                percentage = won_3_or_less.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} won {won_3_or_less.split('(')[0].strip()} points in 3 shots or less on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} won {won_3_or_less} points in 3 shots or less on serves {location_desc}, or {won_3_or_less}% of total points served {location_desc} by {player}.")
                        
                        # First serves in
                        if first_serves_in and first_serves_in != "0":
                            if "(" in first_serves_in and ")" in first_serves_in:
                                percentage = first_serves_in.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} hit in {first_serves_in.split('(')[0].strip()} first serves served {location_desc}, or {percentage}% of first serves served {location_desc} by {player}.")
                            else:
                                text.append(f"{player} hit in {first_serves_in} first serves served {location_desc}, or {first_serves_in}% of first serves served {location_desc} by {player}.")
                        
                        # Double faults
                        if double_faults and double_faults != "0":
                            if "(" in double_faults and ")" in double_faults:
                                percentage = double_faults.split("(")[1].split(")")[0].replace("%", "")
                                text.append(f"{player} served {location_desc} and had {double_faults.split('(')[0].strip()} double faults, or {percentage}% of serves {location_desc} by {player}.")
                            else:
                                text.append(f"{player} served {location_desc} and had {double_faults} double faults, or {double_faults}% of serves {location_desc} by {player}.")
        
        return text

    def _convert_flat_return_to_text(self, return_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat return data to natural language text"""
        text = []
        
        # Group by player
        return1_data = {k: v for k, v in return_data.items() if k.startswith('return1')}
        return2_data = {k: v for k, v in return_data.items() if k.startswith('return2')}
        
        if return1_data:
            text.append("RETURN1 STATISTICS (DETAILED):")
            text.append("-" * 32)
            text.extend(self._convert_flat_return_player_to_text(return1_data, player1))
            text.append("")
        
        if return2_data:
            text.append("RETURN2 STATISTICS (DETAILED):")
            text.append("-" * 32)
            text.extend(self._convert_flat_return_player_to_text(return2_data, player2))
            text.append("")
        
        return text

    def _convert_flat_return_player_to_text(self, return_data: Dict[str, Any], player: str) -> List[str]:
        """Convert flat return data for one player to natural language text"""
        text = []
        
        # TIER 1 (AUTHORITATIVE): Extract totals from "Total" row first
        total_returns = 0
        total_points_won = 0
        total_returnable = 0
        total_returnable_won = 0
        total_in_play = 0
        total_in_play_won = 0
        total_winners = 0
        
        # Find and process the "Total" row as authoritative source
        for key, value in return_data.items():
            if ' - ' in key and '_table' in key:
                return_type_part = key.split(' - ')[1].split('_table')[0]
                table_type = key.split('_table')[1]
                
                # Only process table1 (outcomes) and only the "Total" row
                if table_type == '1' and return_type_part.lower() == 'total':
                    values = [v.strip() for v in value.split(' | ')]
                    if len(values) >= 8:
                        try:
                            total_returns = int(values[0]) if values[0] and values[0] != "0" else 0
                            if "(" in values[1]:
                                total_points_won = int(values[1].split('(')[0].strip()) if values[1].split('(')[0].strip() else 0
                            if "(" in values[2]:
                                total_returnable = int(values[2].split('(')[0].strip()) if values[2].split('(')[0].strip() else 0
                            if "(" in values[3]:
                                total_returnable_won = int(values[3].split('(')[0].strip()) if values[3].split('(')[0].strip() else 0
                            if "(" in values[4]:
                                total_in_play = int(values[4].split('(')[0].strip()) if values[4].split('(')[0].strip() else 0
                            if "(" in values[5]:
                                total_in_play_won = int(values[5].split('(')[0].strip()) if values[5].split('(')[0].strip() else 0
                            if "(" in values[6]:
                                total_winners = int(values[6].split('(')[0].strip()) if values[6].split('(')[0].strip() else 0
                        except (ValueError, IndexError):
                            continue
                    break
        
        # Add authoritative summary (TIER 1)
        if total_returns > 0:
            text.append(f"AUTHORITATIVE TOTALS FOR {player.upper()} RETURNS:")
            text.append(f"{player} returned {total_returns} total serves.")
            text.append(f"{player} won {total_points_won} total return points.")
            text.append(f"{player} faced {total_returnable} total returnable serves (non-aces).")
            text.append(f"{player} won {total_returnable_won} total points on returnable serves.")
            text.append(f"{player} got {total_in_play} total returns in play.")
            text.append(f"{player} won {total_in_play_won} total points when returns were in play.")
            text.append(f"{player} hit {total_winners} total return winners.")
            text.append("")
            text.append("BREAKDOWN BY SERVE TYPE (these sum to totals above, do not add again):")
            text.append("")
        
        # Process each return breakdown
        for key, value in return_data.items():
            # Extract the return type and table type from the key
            # Format: "return1 - vs 1st Svs_table1" or "return1 - Deuce Court_table2"
            if ' - ' in key and '_table' in key:
                return_type_part = key.split(' - ')[1].split('_table')[0]
                table_type = key.split('_table')[1]
                
                # Skip header rows
                if 'OUTCOMES' in return_type_part or 'DEPTH' in return_type_part:
                    continue
                
                # Split the value by | to get individual values
                values = [v.strip() for v in value.split(' | ')]
                
                # Determine return type based on the return_type_part
                if 'vs 1st Svs' in return_type_part:
                    return_desc = "first serves"
                elif 'vs 2nd Svs' in return_type_part:
                    return_desc = "second serves"
                elif 'Deuce Court' in return_type_part:
                    return_desc = "serves from the Deuce court"
                elif 'Ad Court' in return_type_part:
                    return_desc = "serves from the Ad court"
                elif 'Wide serves' in return_type_part:
                    return_desc = "wide serves"
                elif 'Body serves' in return_type_part:
                    return_desc = "body serves"
                elif 'T serves' in return_type_part:
                    return_desc = "T serves"
                elif 'Deuce-Wide' in return_type_part:
                    return_desc = "wide serves from the Deuce court"
                elif 'Ad-Wide' in return_type_part:
                    return_desc = "wide serves from the Ad court"
                elif 'Deuce-Body' in return_type_part:
                    return_desc = "body serves from the Deuce court"
                elif 'Ad-Body' in return_type_part:
                    return_desc = "body serves from the Ad court"
                elif 'Deuce-T' in return_type_part:
                    return_desc = "T serves from the Deuce court"
                elif 'Ad-T' in return_type_part:
                    return_desc = "T serves from the Ad court"
                elif 'Forehand side' in return_type_part:
                    return_desc = "serves to the forehand side"
                elif 'Backhand side' in return_type_part:
                    return_desc = "serves to the backhand side"
                elif 'Flat/Topspin' in return_type_part:
                    return_desc = "flat or topspin serves"
                elif 'Slice/Chip' in return_type_part:
                    return_desc = "slice or chip serves"
                elif 'Svc Box' in return_type_part:
                    return_desc = "serves into the service box"
                elif 'Beh Svc Ln' in return_type_part:
                    return_desc = "serves behind the service line"
                elif 'Back qtr' in return_type_part:
                    return_desc = "serves to the back quarter"
                else:
                    return_desc = return_type_part.lower()
                
                # Handle table1 (outcomes) and table2 (depth) differently
                if table_type == '1':
                    # Table1: Outcomes data
                    if len(values) >= 8:
                        total_pts = values[0]
                        won_pts = values[1]
                        returnable = values[2]
                        returnable_won = values[3]
                        in_play = values[4]
                        in_play_won = values[5]
                        winners = values[6]
                        avg_rally = values[7]
                        
                        # Create detailed descriptions for outcomes
                        if total_pts and total_pts != "0":
                            # Remove "total" from the description when return_desc is "total"
                            if return_desc.lower() == "total":
                                text.append(f"{player} returned {total_pts} times.")
                            else:
                                text.append(f"{player} returned {return_desc} {total_pts} times.")
                            
                            # Points won
                            if won_pts and won_pts != "0":
                                if "(" in won_pts and ")" in won_pts:
                                    percentage = won_pts.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} returned {return_desc} and won {won_pts.split('(')[0].strip()} points, or {percentage}% of returns when {player} was returning.")
                                else:
                                    text.append(f"{player} returned {return_desc} and won {won_pts} points, or {won_pts}% of returns when {player} was returning.")
                            
                            # Returnable serves
                            if returnable and returnable != "0":
                                if "(" in returnable and ")" in returnable:
                                    percentage = returnable.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} got {returnable.split('(')[0].strip()} returnable serves when returning {return_desc}, or {percentage}% of serves when {player} was returning.")
                                else:
                                    text.append(f"{player} got {returnable} returnable serves when returning {return_desc}, or {returnable}% of serves when {player} was returning.")
                            
                            # Points won on returnable serves
                            if returnable_won and returnable_won != "0":
                                if "(" in returnable_won and ")" in returnable_won:
                                    percentage = returnable_won.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} won {returnable_won.split('(')[0].strip()} points on returnable serves when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} won {returnable_won} points on returnable serves when returning {return_desc}, or {returnable_won}% of returnable serves when {player} was returning.")
                            
                            # Returns in play
                            if in_play and in_play != "0":
                                if "(" in in_play and ")" in in_play:
                                    percentage = in_play.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} got {in_play.split('(')[0].strip()} returns in play when returning {return_desc}, or {percentage}% of returns when {player} was returning.")
                                else:
                                    text.append(f"{player} got {in_play} returns in play when returning {return_desc}, or {in_play}% of returns when {player} was returning.")
                            
                            # Points won on returns in play
                            if in_play_won and in_play_won != "0":
                                if "(" in in_play_won and ")" in in_play_won:
                                    percentage = in_play_won.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} won {in_play_won.split('(')[0].strip()} points on returns in play when returning {return_desc}, or {percentage}% of returns in play when {player} was returning.")
                                else:
                                    text.append(f"{player} won {in_play_won} points on returns in play when returning {return_desc}, or {in_play_won}% of returns in play when {player} was returning.")
                            
                            # Winners
                            if winners and winners != "0":
                                if "(" in winners and ")" in winners:
                                    percentage = winners.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {winners.split('(')[0].strip()} winners when returning {return_desc}, or {percentage}% of returns when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {winners} winners when returning {return_desc}, or {winners}% of returns when {player} was returning.")
                            
                            # Average rally length
                            if avg_rally and avg_rally != "0":
                                text.append(f"{player} had an average rally length of {avg_rally} shots when returning {return_desc}.")
                
                elif table_type == '2':
                    # Table2: Depth data
                    if len(values) >= 9:
                        returnable = values[0]
                        shallow = values[1]
                        deep = values[2]
                        very_deep = values[3]
                        unforced_errors = values[4]
                        net_approaches = values[5]
                        deep_returns = values[6]
                        wide_returns = values[7]
                        wide_and_deep = values[8]
                        
                        # Create detailed descriptions for depth
                        if returnable and returnable != "0":
                            # Remove "returning {return_desc}" when return_desc is "total"
                            if return_desc.lower() == "total":
                                text.append(f"{player} had {returnable} returnable serves.")
                            else:
                                text.append(f"{player} had {returnable} returnable serves when returning {return_desc}.")
                            
                            # Shallow returns
                            if shallow and shallow != "0":
                                if "(" in shallow and ")" in shallow:
                                    percentage = shallow.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {shallow.split('(')[0].strip()} shallow returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {shallow} shallow returns when returning {return_desc}.")
                            
                            # Deep returns
                            if deep and deep != "0":
                                if "(" in deep and ")" in deep:
                                    percentage = deep.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {deep.split('(')[0].strip()} deep returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {deep} deep returns when returning {return_desc}.")
                            
                            # Very deep returns
                            if very_deep and very_deep != "0":
                                if "(" in very_deep and ")" in very_deep:
                                    percentage = very_deep.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {very_deep.split('(')[0].strip()} very deep returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {very_deep} very deep returns when returning {return_desc}.")
                            
                            # Unforced errors
                            if unforced_errors and unforced_errors != "0":
                                if "(" in unforced_errors and ")" in unforced_errors:
                                    percentage = unforced_errors.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} made {unforced_errors.split('(')[0].strip()} unforced errors when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} made {unforced_errors} unforced errors when returning {return_desc}.")
                            
                            # Net approaches
                            if net_approaches and net_approaches != "0":
                                if "(" in net_approaches and ")" in net_approaches:
                                    percentage = net_approaches.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} made {net_approaches.split('(')[0].strip()} net approaches when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} made {net_approaches} net approaches when returning {return_desc}.")
                            
                            # Deep returns (from depth column)
                            if deep_returns and deep_returns != "0":
                                if "(" in deep_returns and ")" in deep_returns:
                                    percentage = deep_returns.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {deep_returns.split('(')[0].strip()} deep returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {deep_returns} deep returns when returning {return_desc}.")
                            
                            # Wide returns
                            if wide_returns and wide_returns != "0":
                                if "(" in wide_returns and ")" in wide_returns:
                                    percentage = wide_returns.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {wide_returns.split('(')[0].strip()} wide returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {wide_returns} wide returns when returning {return_desc}.")
                            
                            # Wide and deep returns
                            if wide_and_deep and wide_and_deep != "0":
                                if "(" in wide_and_deep and ")" in wide_and_deep:
                                    percentage = wide_and_deep.split("(")[1].split(")")[0].replace("%", "")
                                    text.append(f"{player} hit {wide_and_deep.split('(')[0].strip()} wide and deep returns when returning {return_desc}, or {percentage}% of returnable serves when {player} was returning.")
                                else:
                                    text.append(f"{player} hit {wide_and_deep} wide and deep returns when returning {return_desc}.")
        
        return text

    def _convert_flat_keypoints_to_text(self, keypoints_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat key points data to natural language text with hierarchy"""
        text = []
        
        # TIER 1 (AUTHORITATIVE): Extract key point totals as single source of truth
        player1_keypoints = {}
        player2_keypoints = {}
        
        # Group by table type
        table1_data = {k: v for k, v in keypoints_data.items() if '_table1' in k}
        table2_data = {k: v for k, v in keypoints_data.items() if '_table2' in k}
        
        # Extract authoritative totals from both tables
        if table1_data:
            player1_keypoints.update(self._extract_keypoints_totals(table1_data, player1))
        if table2_data:
            player1_keypoints.update(self._extract_keypoints_totals(table2_data, player1))
            
        if table1_data:
            player2_keypoints.update(self._extract_keypoints_totals(table1_data, player2))
        if table2_data:
            player2_keypoints.update(self._extract_keypoints_totals(table2_data, player2))
        
        # Add comprehensive authoritative summary (TIER 1)
        text.append("AUTHORITATIVE TOTALS FOR KEY POINTS:")
        text.append("-" * 35)
        
        # Extract comprehensive totals for both players
        player1_comprehensive = self._extract_comprehensive_keypoints_totals(table1_data, table2_data, player1)
        player2_comprehensive = self._extract_comprehensive_keypoints_totals(table1_data, table2_data, player2)
        
        if player1_comprehensive:
            text.append(f"{player1.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
            text.append(f"{player1} faced {player1_comprehensive['total_break_points_faced']} break points total.")
            text.append(f"{player1} faced {player1_comprehensive['total_game_points_faced']} game points total.")
            text.append(f"{player1} faced {player1_comprehensive['break_points_faced_serving']} break points while serving and won {player1_comprehensive['break_points_won_serving']} of them.")
            text.append(f"{player1} faced {player1_comprehensive['break_points_faced_returning']} break points while returning and won {player1_comprehensive['break_points_won_returning']} of them.")
            text.append(f"{player1} faced {player1_comprehensive['game_points_faced_serving']} game points while serving and won {player1_comprehensive['game_points_won_serving']} of them.")
            text.append(f"{player1} faced {player1_comprehensive['game_points_faced_returning']} game points while returning and won {player1_comprehensive['game_points_won_returning']} of them.")
            text.append(f"{player1} played {player1_comprehensive['deuce_points_serving']} deuce points while serving and won {player1_comprehensive['deuce_points_won_serving']} of them.")
            text.append(f"{player1} played {player1_comprehensive['deuce_points_returning']} deuce points while returning and won {player1_comprehensive['deuce_points_won_returning']} of them.")
            text.append("")
        
        if player2_comprehensive:
            text.append(f"{player2.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
            text.append(f"{player2} faced {player2_comprehensive['total_break_points_faced']} break points total.")
            text.append(f"{player2} faced {player2_comprehensive['total_game_points_faced']} game points total.")
            text.append(f"{player2} faced {player2_comprehensive['break_points_faced_serving']} break points while serving and won {player2_comprehensive['break_points_won_serving']} of them.")
            text.append(f"{player2} faced {player2_comprehensive['break_points_faced_returning']} break points while returning and won {player2_comprehensive['break_points_won_returning']} of them.")
            text.append(f"{player2} faced {player2_comprehensive['game_points_faced_serving']} game points while serving and won {player2_comprehensive['game_points_won_serving']} of them.")
            text.append(f"{player2} faced {player2_comprehensive['game_points_faced_returning']} game points while returning and won {player2_comprehensive['game_points_won_returning']} of them.")
            text.append(f"{player2} played {player2_comprehensive['deuce_points_serving']} deuce points while serving and won {player2_comprehensive['deuce_points_won_serving']} of them.")
            text.append(f"{player2} played {player2_comprehensive['deuce_points_returning']} deuce points while returning and won {player2_comprehensive['deuce_points_won_returning']} of them.")
            text.append("")
        
        text.append("DETAILED KEY POINTS BREAKDOWN (these are contextual details, not for recalculating totals):")
        text.append("")
        
        # Process detailed breakdowns (TIER 2)
        if table1_data:
            text.append("KEY POINTS STATISTICS (SERVES):")
            text.append("-" * 32)
            # Include authoritative totals at the beginning of serves section
            text.append("AUTHORITATIVE TOTALS FOR KEY POINTS:")
            text.append("-" * 35)
            if player1_comprehensive:
                text.append(f"{player1.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
                text.append(f"{player1} faced {player1_comprehensive['total_break_points_faced']} break points total.")
                text.append(f"{player1} faced {player1_comprehensive['total_game_points_faced']} game points total.")
                text.append(f"{player1} faced {player1_comprehensive['break_points_faced_serving']} break points while serving and won {player1_comprehensive['break_points_won_serving']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['break_points_faced_returning']} break points while returning and won {player1_comprehensive['break_points_won_returning']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['game_points_faced_serving']} game points while serving and won {player1_comprehensive['game_points_won_serving']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['game_points_faced_returning']} game points while returning and won {player1_comprehensive['game_points_won_returning']} of them.")
                text.append(f"{player1} played {player1_comprehensive['deuce_points_serving']} deuce points while serving and won {player1_comprehensive['deuce_points_won_serving']} of them.")
                text.append(f"{player1} played {player1_comprehensive['deuce_points_returning']} deuce points while returning and won {player1_comprehensive['deuce_points_won_returning']} of them.")
                text.append("")
            if player2_comprehensive:
                text.append(f"{player2.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
                text.append(f"{player2} faced {player2_comprehensive['total_break_points_faced']} break points total.")
                text.append(f"{player2} faced {player2_comprehensive['total_game_points_faced']} game points total.")
                text.append(f"{player2} faced {player2_comprehensive['break_points_faced_serving']} break points while serving and won {player2_comprehensive['break_points_won_serving']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['break_points_faced_returning']} break points while returning and won {player2_comprehensive['break_points_won_returning']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['game_points_faced_serving']} game points while serving and won {player2_comprehensive['game_points_won_serving']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['game_points_faced_returning']} game points while returning and won {player2_comprehensive['game_points_won_returning']} of them.")
                text.append(f"{player2} played {player2_comprehensive['deuce_points_serving']} deuce points while serving and won {player2_comprehensive['deuce_points_won_serving']} of them.")
                text.append(f"{player2} played {player2_comprehensive['deuce_points_returning']} deuce points while returning and won {player2_comprehensive['deuce_points_won_returning']} of them.")
                text.append("")
            text.append("DETAILED KEY POINTS BREAKDOWN (these are contextual details, not for recalculating totals):")
            text.append("")
            text.extend(self._convert_flat_keypoints_table_to_text(table1_data, "serves"))
            text.append("")
        
        if table2_data:
            text.append("KEY POINTS STATISTICS (RETURNS):")
            text.append("-" * 33)
            # Include authoritative totals at the beginning of returns section
            text.append("AUTHORITATIVE TOTALS FOR KEY POINTS:")
            text.append("-" * 35)
            if player1_comprehensive:
                text.append(f"{player1.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
                text.append(f"{player1} faced {player1_comprehensive['total_break_points_faced']} break points total.")
                text.append(f"{player1} faced {player1_comprehensive['total_game_points_faced']} game points total.")
                text.append(f"{player1} faced {player1_comprehensive['break_points_faced_serving']} break points while serving and won {player1_comprehensive['break_points_won_serving']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['break_points_faced_returning']} break points while returning and won {player1_comprehensive['break_points_won_returning']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['game_points_faced_serving']} game points while serving and won {player1_comprehensive['game_points_won_serving']} of them.")
                text.append(f"{player1} faced {player1_comprehensive['game_points_faced_returning']} game points while returning and won {player1_comprehensive['game_points_won_returning']} of them.")
                text.append(f"{player1} played {player1_comprehensive['deuce_points_serving']} deuce points while serving and won {player1_comprehensive['deuce_points_won_serving']} of them.")
                text.append(f"{player1} played {player1_comprehensive['deuce_points_returning']} deuce points while returning and won {player1_comprehensive['deuce_points_won_returning']} of them.")
                text.append("")
            if player2_comprehensive:
                text.append(f"{player2.upper()} KEY POINTS AUTHORITATIVE TOTALS:")
                text.append(f"{player2} faced {player2_comprehensive['total_break_points_faced']} break points total.")
                text.append(f"{player2} faced {player2_comprehensive['total_game_points_faced']} game points total.")
                text.append(f"{player2} faced {player2_comprehensive['break_points_faced_serving']} break points while serving and won {player2_comprehensive['break_points_won_serving']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['break_points_faced_returning']} break points while returning and won {player2_comprehensive['break_points_won_returning']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['game_points_faced_serving']} game points while serving and won {player2_comprehensive['game_points_won_serving']} of them.")
                text.append(f"{player2} faced {player2_comprehensive['game_points_faced_returning']} game points while returning and won {player2_comprehensive['game_points_won_returning']} of them.")
                text.append(f"{player2} played {player2_comprehensive['deuce_points_serving']} deuce points while serving and won {player2_comprehensive['deuce_points_won_serving']} of them.")
                text.append(f"{player2} played {player2_comprehensive['deuce_points_returning']} deuce points while returning and won {player2_comprehensive['deuce_points_won_returning']} of them.")
                text.append("")
            text.append("DETAILED KEY POINTS BREAKDOWN (these are contextual details, not for recalculating totals):")
            text.append("")
            text.extend(self._convert_flat_keypoints_table_to_text(table2_data, "returns"))
            text.append("")
        
        return text

    def _generate_player_initials(self, player_name: str) -> str:
        """Generate player initials from full name"""
        if not player_name:
            return ""
        words = player_name.split()
        if len(words) >= 2:
            return words[0][0].upper() + words[1][0].upper()
        elif len(words) == 1:
            return words[0][:2].upper()
        return ""

    def _extract_keypoints_totals(self, keypoints_data: Dict[str, Any], player: str) -> Dict[str, str]:
        """Extract authoritative totals from key points statistics"""
        totals = {}
        
        # Generate player initials dynamically
        player_initials = self._generate_player_initials(player)
        
        # Find the header row
        header_key = None
        for key in keypoints_data.keys():
            if 'KEY POINTS:' in key:
                header_key = key
                break
        
        if header_key:
            headers = [h.strip() for h in keypoints_data[header_key].split(' | ')]
            
            # Process each data row to find player-specific totals
            for key, value in keypoints_data.items():
                if key != header_key and 'KEY POINTS:' not in key:
                    # Extract the row label
                    row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Check if this row belongs to the target player using dynamic initials
                    if (player_initials in row_label) or (player in row_label):
                        
                        values = [v.strip() for v in value.split(' | ')]
                        
                        # Extract totals based on row type
                        if 'BP Faced' in row_label or 'Break Points' in row_label:
                            if len(values) >= 1:
                                totals['break_points_faced'] = values[0]
                            if len(values) >= 2:
                                totals['break_points_converted'] = values[1]
                        elif 'Game Pts' in row_label or 'Game Points' in row_label:
                            if len(values) >= 1:
                                totals['game_points'] = values[0]
                        elif 'Set Points' in row_label:
                            if len(values) >= 1:
                                totals['set_points'] = values[0]
                        elif 'Match Points' in row_label:
                            if len(values) >= 1:
                                totals['match_points'] = values[0]
        
        return totals
    
    def _extract_comprehensive_keypoints_totals(self, serves_data: Dict[str, Any], returns_data: Dict[str, Any], player: str) -> Dict[str, str]:
        """Extract comprehensive key points totals from both serves and returns tables"""
        comprehensive = {
            'total_break_points_faced': '0',
            'total_game_points_faced': '0',
            'break_points_faced_serving': '0',
            'break_points_won_serving': '0',
            'break_points_faced_returning': '0',
            'break_points_won_returning': '0',
            'game_points_faced_serving': '0',
            'game_points_won_serving': '0',
            'game_points_faced_returning': '0',
            'game_points_won_returning': '0',
            'deuce_points_serving': '0',
            'deuce_points_won_serving': '0',
            'deuce_points_returning': '0',
            'deuce_points_won_returning': '0'
        }
        
        # Extract from serves table (when player is serving)
        if serves_data:
            serves_totals = self._extract_keypoints_from_table(serves_data, player, "serves")
            comprehensive.update(serves_totals)
        
        # Extract from returns table (when player is returning)
        if returns_data:
            returns_totals = self._extract_keypoints_from_table(returns_data, player, "returns")
            comprehensive.update(returns_totals)
        
        # Calculate totals
        try:
            bp_serving_faced = int(comprehensive['break_points_faced_serving'])
            bp_returning_faced = int(comprehensive['break_points_faced_returning'])
            comprehensive['total_break_points_faced'] = str(bp_serving_faced + bp_returning_faced)
            
            gp_serving_faced = int(comprehensive['game_points_faced_serving'])
            gp_returning_faced = int(comprehensive['game_points_faced_returning'])
            comprehensive['total_game_points_faced'] = str(gp_serving_faced + gp_returning_faced)
            
            # Deuce points should be extracted from the data above
        except (ValueError, TypeError):
            pass
        
        return comprehensive
    
    def _extract_keypoints_from_table(self, table_data: Dict[str, Any], player: str, table_type: str) -> Dict[str, str]:
        """Extract key points data from a specific table (serves or returns)"""
        totals = {}
        
        # Generate player initials dynamically
        player_initials = self._generate_player_initials(player)
        
        # Find the header row
        header_key = None
        for key in table_data.keys():
            if 'KEY POINTS:' in key:
                header_key = key
                break
        
        if not header_key:
            return totals
        
        headers = [h.strip() for h in table_data[header_key].split(' | ')]
        
        # Process each data row to find player-specific data
        for key, value in table_data.items():
            if key != header_key and 'KEY POINTS:' not in key:
                # Extract the row label
                row_label = key.split(' - ', 1)[1] if ' - ' in key else key
                
                # Check if this row belongs to the target player using dynamic initials
                if player_initials in row_label:
                    
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Extract data based on row type and table type
                    if 'BP Faced' in row_label or 'BP Opps' in row_label:
                        if len(values) >= 1:
                            if table_type == "serves":
                                totals['break_points_faced_serving'] = values[0]
                            else:  # returns
                                totals['break_points_faced_returning'] = values[0]
                        if len(values) >= 2:
                            if table_type == "serves":
                                totals['break_points_won_serving'] = values[1].split('(')[0].strip()
                            else:  # returns
                                totals['break_points_won_returning'] = values[1].split('(')[0].strip()
                    
                    elif 'Game Pts' in row_label or 'GP Faced' in row_label:
                        if len(values) >= 1:
                            if table_type == "serves":
                                totals['game_points_faced_serving'] = values[0]
                            else:  # returns
                                totals['game_points_faced_returning'] = values[0]
                        if len(values) >= 2:
                            if table_type == "serves":
                                totals['game_points_won_serving'] = values[1].split('(')[0].strip()
                            else:  # returns
                                totals['game_points_won_returning'] = values[1].split('(')[0].strip()
                    
                    elif 'Svg Deuce' in row_label or 'Ret Deuce' in row_label:
                        if len(values) >= 1:
                            if table_type == "serves":
                                totals['deuce_points_serving'] = values[0]
                            else:  # returns
                                totals['deuce_points_returning'] = values[0]
                        if len(values) >= 2:
                            if table_type == "serves":
                                totals['deuce_points_won_serving'] = values[1].split('(')[0].strip()
                            else:  # returns
                                totals['deuce_points_won_returning'] = values[1].split('(')[0].strip()
        
        return totals

    def _convert_flat_keypoints_table_to_text(self, keypoints_data: Dict[str, Any], table_type: str) -> List[str]:
        """Convert flat key points table data to natural language text"""
        text = []
        
        # Find the header row
        header_key = None
        for key in keypoints_data.keys():
            if 'KEY POINTS:' in key:
                header_key = key
                break
        
        if header_key:
            headers = keypoints_data[header_key].split(' | ')
            
            # Process each data row
            for key, value in keypoints_data.items():
                if key != header_key and 'KEY POINTS:' not in key:
                    # Extract the row label (remove the prefix like "keypoints - ")
                    row_label = key.split(' - ', 1)[1].split('_table')[0]
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Convert to sentences using the existing method
                    sentences = self._convert_keypoints_row_to_sentences(row_label, values, headers, table_type)
                    text.extend(sentences)
        
        return text

    def _convert_keypoints_row_to_sentences(self, row_label: str, values: List[str], headers: List[str], table_type: str) -> List[str]:
        """Convert key points row to natural language sentences"""
        sentences = []
        
        # Determine player from row label dynamically
        player = "Unknown Player"
        if self.player1 and self.player2:
            player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
            player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
            if player1_initials in row_label:
                player = self.player1
            elif player2_initials in row_label:
                player = self.player2
        
        # Determine key point type from row label and table type
        if table_type == "serves":
            # When table_type is "serves", the player is serving
            if 'BP Faced' in row_label:
                key_point_type = "break points faced when serving"
            elif 'BP Opps' in row_label:
                key_point_type = "break point opportunities when serving"
            elif 'Game Pts' in row_label:
                key_point_type = "game points when serving"
            elif 'GP Faced' in row_label:
                key_point_type = "game points faced when serving"
            elif 'Svg Deuce' in row_label:
                key_point_type = "deuce points when serving"
            elif 'Ret Deuce' in row_label:
                key_point_type = "deuce points when returning serves"
            elif 'Total' in row_label:
                key_point_type = f"total key points ({table_type})"
            else:
                key_point_type = "key points"
        else:
            # When table_type is "returns", the player is returning
            if 'BP Faced' in row_label:
                key_point_type = "break points faced when returning serves"
            elif 'BP Opps' in row_label:
                key_point_type = "break point opportunities when returning serves"
            elif 'Game Pts' in row_label:
                key_point_type = "game points when returning serves"
            elif 'GP Faced' in row_label:
                key_point_type = "game points faced when returning serves"
            elif 'Svg Deuce' in row_label:
                key_point_type = "deuce points when serving"
            elif 'Ret Deuce' in row_label:
                key_point_type = "deuce points when returning serves"
            elif 'Total' in row_label:
                key_point_type = f"total key points ({table_type})"
            else:
                key_point_type = "key points"
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(headers) and value and value != "0":
                header = headers[i]
                
                if header == "Pts":
                    sentences.append(f"{player} played {value} {key_point_type}.")
                elif header == "PtsW----%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} won {number} {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} won {value} {key_point_type}.")
                elif header == "1stIn---%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} made {number} first serves in on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} made {value} first serves in on {key_point_type}.")
                elif header == "A-------%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} hit {number} aces on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} hit {value} aces on {key_point_type}.")
                elif header == "SvWnr---%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} hit {number} serve winners on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} hit {value} serve winners on {key_point_type}.")
                elif header == "RlyWnr--%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} hit {number} rally winners on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} hit {value} rally winners on {key_point_type}.")
                elif header == "RlyFcd--%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} forced {number} errors on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} forced {value} errors on {key_point_type}.")
                elif header == "UFE-----%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} made {number} unforced errors on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} made {value} unforced errors on {key_point_type}.")
                elif header == "DF------%":
                    if "(" in value and ")" in value:
                        number = value.split("(")[0].strip()
                        percentage = value.split("(")[1].split(")")[0].replace("%", "")
                        sentences.append(f"{player} made {number} double faults on {key_point_type}, or {percentage}% of {key_point_type} for {player}.")
                    else:
                        sentences.append(f"{player} made {value} double faults on {key_point_type}.")
        
        return sentences

    def _convert_flat_overview_to_text(self, overview_data: Dict[str, Any], player1: str, player2: str) -> List[str]:
        """Convert flat overview data to natural language text with hierarchy"""
        text = []
        
        # TIER 1 (AUTHORITATIVE): Extract key totals from Overview as single source of truth
        player1_serve_stats = {}
        player2_serve_stats = {}
        
        # Generate player initials dynamically
        player1_initials = self._generate_player_initials(player1)
        player2_initials = self._generate_player_initials(player2)
        
        # Find the header row
        header_key = None
        for key in overview_data.keys():
            if 'STATS OVERVIEW' in key:
                header_key = key
                break
        
        if header_key:
            headers = [h.strip() for h in overview_data[header_key].split(' | ')]
            
            # Process each player's data to extract authoritative totals
            for key, value in overview_data.items():
                if key != header_key and 'STATS OVERVIEW' not in key:
                    # Extract the player name
                    player = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Store authoritative totals using dynamic player detection
                    if player1 in player or player1_initials in player:
                        player1_serve_stats = self._extract_overview_totals(values, headers)
                    elif player2 in player or player2_initials in player:
                        player2_serve_stats = self._extract_overview_totals(values, headers)
        
        # Add authoritative summary (TIER 1)
        text.append("AUTHORITATIVE TOTALS FROM OVERVIEW STATISTICS:")
        text.append("-" * 45)
        
        if player1_serve_stats:
            text.append(f"{player1.upper()} AUTHORITATIVE TOTALS:")
            text.append(f"First serve percentage: {player1_serve_stats.get('first_serve_pct', 'N/A')}")
            text.append(f"Second serve won percentage: {player1_serve_stats.get('second_serve_won_pct', 'N/A')}")
            text.append(f"Ace percentage: {player1_serve_stats.get('ace_pct', 'N/A')}")
            text.append(f"Double fault percentage: {player1_serve_stats.get('df_pct', 'N/A')}")
            text.append(f"Break points saved: {player1_serve_stats.get('bp_saved', 'N/A')}")
            text.append(f"Return points won percentage: {player1_serve_stats.get('return_pct', 'N/A')}")
            text.append(f"Total winners: {player1_serve_stats.get('winners', 'N/A')}")
            text.append(f"Total unforced errors: {player1_serve_stats.get('ufe', 'N/A')}")
            text.append("")
        
        if player2_serve_stats:
            text.append(f"{player2.upper()} AUTHORITATIVE TOTALS:")
            text.append(f"First serve percentage: {player2_serve_stats.get('first_serve_pct', 'N/A')}")
            text.append(f"Second serve won percentage: {player2_serve_stats.get('second_serve_won_pct', 'N/A')}")
            text.append(f"Ace percentage: {player2_serve_stats.get('ace_pct', 'N/A')}")
            text.append(f"Double fault percentage: {player2_serve_stats.get('df_pct', 'N/A')}")
            text.append(f"Break points saved: {player2_serve_stats.get('bp_saved', 'N/A')}")
            text.append(f"Return points won percentage: {player2_serve_stats.get('return_pct', 'N/A')}")
            text.append(f"Total winners: {player2_serve_stats.get('winners', 'N/A')}")
            text.append(f"Total unforced errors: {player2_serve_stats.get('ufe', 'N/A')}")
            text.append("")
        
        text.append("DETAILED BREAKDOWN (these are contextual details, not for recalculating totals):")
        text.append("")
        
        # Process detailed breakdowns (TIER 2)
        if header_key:
            headers = [h.strip() for h in overview_data[header_key].split(' | ')]
            
            for key, value in overview_data.items():
                if key != header_key and 'STATS OVERVIEW' not in key:
                    # Extract the player name
                    player = key.split(' - ', 1)[1] if ' - ' in key else key
                    
                    # Split the value by | to get individual values
                    values = [v.strip() for v in value.split(' | ')]
                    
                    # Convert to sentences
                    sentences = self._convert_overview_row_to_sentences(player, values, headers)
                    text.extend(sentences)
        
        return text

    def _extract_overview_totals(self, values: List[str], headers: List[str]) -> Dict[str, str]:
        """Extract authoritative totals from overview statistics"""
        totals = {}
        
        for i, value in enumerate(values):
            if i < len(headers) and value and value != "0":
                header = headers[i]
                
                if header == "1stIn":
                    totals['first_serve_pct'] = value
                elif header == "2nd%":
                    totals['second_serve_won_pct'] = value
                elif header == "A%":
                    totals['ace_pct'] = value
                elif header == "DF%":
                    totals['df_pct'] = value
                elif header == "BPSaved":
                    totals['bp_saved'] = value
                elif header == "RPW%":
                    totals['return_pct'] = value
                elif header == "Winners (FH/BH)":
                    totals['winners'] = value
                elif header == "UFE (FH/BH)":
                    totals['ufe'] = value
        
        return totals

    def _convert_overview_row_to_sentences(self, player: str, values: List[str], headers: List[str]) -> List[str]:
        """Convert overview statistics row to natural language sentences"""
        sentences = []
        
        # Convert each value to a sentence
        for i, value in enumerate(values):
            if i < len(headers) and value and value != "0":
                header = headers[i]
                
                if header == "A%":
                    sentences.append(f"{player} had an ace percentage of {value}.")
                elif header == "DF%":
                    sentences.append(f"{player} had a double fault percentage of {value}.")
                elif header == "1stIn":
                    sentences.append(f"{player} had a first serve percentage of {value}.")
                elif header == "1st%":
                    sentences.append(f"{player} won {value} of first serves.")
                elif header == "2nd%":
                    sentences.append(f"{player} won {value} of second serves.")
                elif header == "BPSaved":
                    sentences.append(f"{player} saved {value} break points.")
                elif header == "RPW%":
                    sentences.append(f"{player} won {value} of return points.")
                elif header == "Winners (FH/BH)":
                    sentences.append(f"{player} hit {value} winners (forehand/backhand).")
                elif header == "UFE (FH/BH)":
                    sentences.append(f"{player} made {value} unforced errors (forehand/backhand).")
                else:
                    sentences.append(f"{player} had {header}: {value}.")
        
        return sentences

    def _convert_point_log_to_text(self, point_log: List[Dict[str, Any]], player1: str, player2: str) -> List[str]:
        """Convert point-by-point data to natural language text with server information"""
        text = []
        text.append("POINT-BY-POINT NARRATIVE:")
        text.append("-" * 30)
        
        for i, point in enumerate(point_log, 1):
            # Extract point information
            point_num = point.get('point', f'Point {i}')
            server = point.get('server', '')
            sets = point.get('sets', '')
            games = point.get('games', '')
            points = point.get('points', '')
            score = point.get('score', '')
            description = point.get('description', '')
            
            # Create server context information
            server_info = ""
            if server:
                # Determine the returner dynamically using the actual player names
                if server.strip() == player1.strip():
                    returner = player2
                elif server.strip() == player2.strip():
                    returner = player1
                else:
                    # Fallback: try partial matching
                    server_lower = server.lower()
                    player1_lower = player1.lower()
                    player2_lower = player2.lower()
                    
                    if any(part in server_lower for part in player1_lower.split()):
                        returner = player2
                    elif any(part in server_lower for part in player2_lower.split()):
                        returner = player1
                    else:
                        returner = "Opponent"
                
                server_info = f"[Server: {server} | Returner: {returner} | Score: {sets} {games} {points}]"
            
            # Create natural language description with server context
            if description:
                if server_info:
                    text.append(f"{point_num} {server_info}: {description}")
                else:
                    text.append(f"{point_num}: {description}")
            elif score:
                text.append(f"{point_num}: Score is {score}")
            else:
                text.append(f"{point_num}: Point played")
        
        return text

    def _convert_other_data_table(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert other data table to natural language text"""
        text = []
        text.append("OTHER DATA STATISTICS:")
        text.append("-" * 20)
        
        for row in rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label == 'Match Result' and values:
                text.append(f"Match result: {values[0]}")
        
        return text

    def _convert_serve_table(self, table_name: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert serve table to natural language text"""
        text = []
        
        # Determine player
        if 'serve1' in table_name.lower():
            player = self.player1 if self.player1 else "Player 1"
            text.append("SERVE1 STATISTICS:")
        elif 'serve2' in table_name.lower():
            player = self.player2 if self.player2 else "Player 2"
            text.append("SERVE2 STATISTICS:")
        else:
            player = "Unknown Player"
            text.append(f"{table_name.upper()} STATISTICS:")
        
        text.append("-" * len(text[-1]))
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'BREAKDOWN' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                # Determine serve location and type from the row label
                serve_location = label.lower()
                
                # Convert serve location to proper description
                if 'deuce court' in serve_location:
                    location_desc = "to the Deuce court"
                elif 'ad court' in serve_location:
                    location_desc = "to the Ad court"
                elif 'wide serves' in serve_location:
                    location_desc = "wide"
                elif 'body serves' in serve_location:
                    location_desc = "into the body"
                elif 't serves' in serve_location:
                    location_desc = "down the T"
                else:
                    location_desc = serve_location
                
                # Extract all available stats
                total_pts = values[0] if len(values) > 0 else "0"
                won_pts = values[1] if len(values) > 1 else "0"
                aces = values[2] if len(values) > 2 else "0"
                unreturned = values[3] if len(values) > 3 else "0"
                forced_errors = values[4] if len(values) > 4 else "0"
                won_3_or_less = values[5] if len(values) > 5 else "0"
                first_serves_in = values[6] if len(values) > 6 else "0"
                double_faults = values[7] if len(values) > 7 else "0"
                
                # Create detailed descriptions for each serve location
                if total_pts and total_pts != "0":
                    # Total serves
                    text.append(f"{player} served {location_desc} {total_pts} times.")
                    
                    # Points won
                    if won_pts and won_pts != "0":
                        if "(" in won_pts and ")" in won_pts:
                            percentage = won_pts.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} served {location_desc} and won {won_pts.split('(')[0].strip()} points, or {percentage}% of total points served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} served {location_desc} and won {won_pts} points, or {won_pts}% of total points served {location_desc} by {player}.")
                    
                    # Aces
                    if aces and aces != "0":
                        if "(" in aces and ")" in aces:
                            percentage = aces.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} hit {aces.split('(')[0].strip()} aces served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} hit {aces} aces served {location_desc}, or {aces}% of total points served {location_desc} by {player}.")
                    
                    # Unreturned serves
                    if unreturned and unreturned != "0":
                        if "(" in unreturned and ")" in unreturned:
                            percentage = unreturned.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} had {unreturned.split('(')[0].strip()} unreturned serves served {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} had {unreturned} unreturned serves served {location_desc}, or {unreturned}% of total points served {location_desc} by {player}.")
                    
                    # Forced errors
                    if forced_errors and forced_errors != "0":
                        if "(" in forced_errors and ")" in forced_errors:
                            percentage = forced_errors.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} forced {forced_errors.split('(')[0].strip()} errors from her opponent on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} forced {forced_errors} errors from her opponent on serves {location_desc}, or {forced_errors}% of total points served {location_desc} by {player}.")
                    
                    # Points won in 3 shots or less
                    if won_3_or_less and won_3_or_less != "0":
                        if "(" in won_3_or_less and ")" in won_3_or_less:
                            percentage = won_3_or_less.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} won {won_3_or_less.split('(')[0].strip()} points in 3 shots or less on serves {location_desc}, or {percentage}% of total points served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} won {won_3_or_less} points in 3 shots or less on serves {location_desc}, or {won_3_or_less}% of total points served {location_desc} by {player}.")
                    
                    # First serves in
                    if first_serves_in and first_serves_in != "0":
                        if "(" in first_serves_in and ")" in first_serves_in:
                            percentage = first_serves_in.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} hit in {first_serves_in.split('(')[0].strip()} first serves served {location_desc}, or {percentage}% of first serves served {location_desc} by {player}.")
                        else:
                            text.append(f"{player} hit in {first_serves_in} first serves served {location_desc}, or {first_serves_in}% of first serves served {location_desc} by {player}.")
                    
                    # Double faults
                    if double_faults and double_faults != "0":
                        if "(" in double_faults and ")" in double_faults:
                            percentage = double_faults.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} served {location_desc} and had {double_faults.split('(')[0].strip()} double faults, or {percentage}% of serves {location_desc} by {player}.")
                        else:
                            text.append(f"{player} served {location_desc} and had {double_faults} double faults, or {double_faults}% of serves {location_desc} by {player}.")
        
        return text

    def _convert_return_table(self, table_name: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert return table to natural language text"""
        text = []
        
        # Determine player
        if 'return1' in table_name.lower():
            player = self.player1 if self.player1 else "Player 1"
            text.append("RETURN1 STATISTICS:")
        elif 'return2' in table_name.lower():
            player = self.player2 if self.player2 else "Player 2"
            text.append("RETURN2 STATISTICS:")
        else:
            player = "Unknown Player"
            text.append(f"{table_name.upper()} STATISTICS:")
        
        text.append("-" * len(text[-1]))
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'BREAKDOWN' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                # Determine return type
                return_type = label.lower()
                if '1st serve return' in return_type:
                    return_desc = "first serves"
                elif '2nd serve return' in return_type:
                    return_desc = "second serves"
                else:
                    return_desc = return_type
                
                # Extract all available stats
                total_pts = values[0] if len(values) > 0 else "0"
                won_pts = values[1] if len(values) > 1 else "0"
                in_play = values[2] if len(values) > 2 else "0"
                in_play_won = values[3] if len(values) > 3 else "0"
                winners = values[4] if len(values) > 4 else "0"
                forced_errors = values[5] if len(values) > 5 else "0"
                unforced_errors = values[6] if len(values) > 6 else "0"
                avg_rally = values[7] if len(values) > 7 else "0"
                
                # Create detailed descriptions for each return type
                if total_pts and total_pts != "0":
                    # Total returns
                    # Remove "total" from the description when return_desc is "total"
                    if return_desc.lower() == "total":
                        text.append(f"{player} returned {total_pts} times.")
                    else:
                        text.append(f"{player} returned {return_desc} {total_pts} times.")
                    
                    # Points won
                    if won_pts and won_pts != "0":
                        if "(" in won_pts and ")" in won_pts:
                            percentage = won_pts.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} returned {return_desc} and won {won_pts.split('(')[0].strip()} points, or {percentage}% of returns for {return_desc}.")
                        else:
                            text.append(f"{player} returned {return_desc} and won {won_pts} points, or {won_pts}% of returns for {return_desc}.")
                    
                    # Returns in play
                    if in_play and in_play != "0":
                        if "(" in in_play and ")" in in_play:
                            percentage = in_play.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} got {in_play.split('(')[0].strip()} returns in play, or {percentage}% of returns when {player} was returning.")
                        else:
                            text.append(f"{player} got {in_play} returns in play, or {in_play}% of returns when {player} was returning.")
                    
                    # Points won on returns in play
                    if in_play_won and in_play_won != "0":
                        if "(" in in_play_won and ")" in in_play_won:
                            percentage = in_play_won.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} won {in_play_won.split('(')[0].strip()} points on returns in play, or {percentage}% of returns in play when {player} was returning.")
                        else:
                            text.append(f"{player} won {in_play_won} points on returns in play, or {in_play_won}% of returns in play when {player} was returning.")
                    
                    # Winners
                    if winners and winners != "0":
                        if "(" in winners and ")" in winners:
                            percentage = winners.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} hit {winners.split('(')[0].strip()} winners on returns, or {percentage}% of returns when {player} was returning.")
                        else:
                            text.append(f"{player} hit {winners} winners on returns, or {winners}% of returns when {player} was returning.")
                    
                    # Forced errors
                    if forced_errors and forced_errors != "0":
                        if "(" in forced_errors and ")" in forced_errors:
                            percentage = forced_errors.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} forced {forced_errors.split('(')[0].strip()} errors on returns, or {percentage}% of returns when {player} was returning.")
                        else:
                            text.append(f"{player} forced {forced_errors} errors on returns, or {forced_errors}% of returns when {player} was returning.")
                    
                    # Unforced errors
                    if unforced_errors and unforced_errors != "0":
                        if "(" in unforced_errors and ")" in unforced_errors:
                            percentage = unforced_errors.split("(")[1].split(")")[0].replace("%", "")
                            text.append(f"{player} made {unforced_errors.split('(')[0].strip()} unforced errors on returns, or {percentage}% of returns when {player} was returning.")
                        else:
                            text.append(f"{player} made {unforced_errors} unforced errors on returns, or {unforced_errors}% of returns when {player} was returning.")
                    
                    # Average rally length
                    if avg_rally and avg_rally != "0":
                        text.append(f"{player} had an average rally length of {avg_rally} shots when returning {return_desc}.")
        
        return text

    def _convert_keypoints_table(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert key points table to natural language text"""
        text = []
        text.append("KEY POINTS STATISTICS:")
        text.append("-" * 22)
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'KEY POINTS:' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                # Determine player dynamically from initials
                player = "Unknown Player"
                if self.player1 and self.player2:
                    player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
                    player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
                    if player1_initials in label:
                        player = self.player1
                    elif player2_initials in label:
                        player = self.player2
                
                # Extract key stats
                total_pts = values[0] if len(values) > 0 else "0"
                won_pct = values[1] if len(values) > 1 else "0%"
                
                text.append(f"{player} played {total_pts} key points and won {won_pct} of them.")
        
        return text

    def _convert_serveneut_table(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert serve neutral table to natural language text"""
        text = []
        text.append("SERVENEUT STATISTICS:")
        text.append("-" * 20)
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'SERVE INFLUENCE' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                # Determine player dynamically from initials
                player = "Unknown Player"
                if self.player1 and self.player2:
                    player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
                    player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
                    if player1_initials in label:
                        player = self.player1
                    elif player2_initials in label:
                        player = self.player2
                
                # Determine serve type
                if '1st Serve' in label:
                    serve_type = "first serves"
                elif '2nd Serve' in label:
                    serve_type = "second serves"
                else:
                    serve_type = "serves"
                
                # Extract key stats
                total_pts = values[0] if len(values) > 0 else "0"
                overall_pct = values[1] if len(values) > 1 else "0%"
                
                text.append(f"{player} served {total_pts} {serve_type} and won {overall_pct} of points overall.")
        
        return text

    def _convert_rallyoutcomes_table(self, rows: List[Dict[str, Any]], player1: str, player2: str) -> List[str]:
        """Convert rally outcomes table to natural language text"""
        text = []
        text.append("RALLY OUTCOMES STATISTICS:")
        text.append("-" * 26)
        
        # Generate player initials dynamically
        player1_initials = self._generate_player_initials(player1)
        player2_initials = self._generate_player_initials(player2)
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'OUTCOMES' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values and len(values) >= 9:
                # Determine rally type and player context
                rally_type = label
                if 'Total' in label:
                    rally_type = "total rallies"
                elif 'All: 1-3' in label:
                    rally_type = "1-3 shot rallies"
                elif 'All: 4-6' in label:
                    rally_type = "4-6 shot rallies"
                elif 'All: 7-9' in label:
                    rally_type = "7-9 shot rallies"
                elif 'All: 10+' in label:
                    rally_type = "10+ shot rallies"
                elif f'{player1_initials} Sv:' in label:
                    rally_type = f"1-3 shot rallies on {player1} serves" if "1-3" in label else \
                               f"4-6 shot rallies on {player1} serves" if "4-6" in label else \
                               f"7-9 shot rallies on {player1} serves" if "7-9" in label else \
                               f"10+ shot rallies on {player1} serves" if "10+" in label else \
                               f"rallies on {player1} serves"
                elif f'{player2_initials} Sv:' in label:
                    rally_type = f"1-3 shot rallies on {player2} serves" if "1-3" in label else \
                               f"4-6 shot rallies on {player2} serves" if "4-6" in label else \
                               f"7-9 shot rallies on {player2} serves" if "7-9" in label else \
                               f"10+ shot rallies on {player2} serves" if "10+" in label else \
                               f"rallies on {player2} serves"
                
                # Extract stats for both players
                total_pts = values[0] if len(values) > 0 else "0"
                
                # Player 1 stats (columns 1-4) - strip percentages
                player1_wins = values[1].split('(')[0].strip() if len(values) > 1 and values[1] else "0"
                player1_winners = values[2].split('(')[0].strip() if len(values) > 2 and values[2] else "0"
                player1_forced_errors = values[3].split('(')[0].strip() if len(values) > 3 and values[3] else "0"
                player1_unforced_errors = values[4].split('(')[0].strip() if len(values) > 4 and values[4] else "0"
                
                # Player 2 stats (columns 5-8) - strip percentages
                player2_wins = values[5].split('(')[0].strip() if len(values) > 5 and values[5] else "0"
                player2_winners = values[6].split('(')[0].strip() if len(values) > 6 and values[6] else "0"
                player2_forced_errors = values[7].split('(')[0].strip() if len(values) > 7 and values[7] else "0"
                player2_unforced_errors = values[8].split('(')[0].strip() if len(values) > 8 and values[8] else "0"
                
                # Create sentences
                text.append(f"There were {total_pts} {rally_type} in the match.")
                
                # Add player-specific statistics
                if f'{player1_initials} Sv:' in label or f'{player2_initials} Sv:' in label:
                    # For player-specific rows, focus on that player's stats
                    if f'{player1_initials} Sv:' in label:
                        text.append(f"{player1} won {player1_wins} points and hit {player1_winners} winners on {rally_type}.")
                        text.append(f"{player1} had {player1_forced_errors} forced errors and made {player1_unforced_errors} unforced errors on {rally_type}.")
                    else:  # Player 2 Sv
                        text.append(f"{player2} won {player2_wins} points and hit {player2_winners} winners on {rally_type}.")
                        text.append(f"{player2} had {player2_forced_errors} forced errors and made {player2_unforced_errors} unforced errors on {rally_type}.")
                else:
                    # For total/all rows, show both players' stats
                    text.append(f"{player1} won {player1_wins} points and hit {player1_winners} winners on {rally_type}.")
                    text.append(f"{player1} had {player1_forced_errors} forced errors and made {player1_unforced_errors} unforced errors on {rally_type}.")
                    text.append(f"{player2} won {player2_wins} points and hit {player2_winners} winners on {rally_type}.")
                    text.append(f"{player2} had {player2_forced_errors} forced errors and made {player2_unforced_errors} unforced errors on {rally_type}.")
                
                text.append("")  # Add spacing between sections
        
        return text

    def _convert_overview_table(self, rows: List[Dict[str, Any]], player1: str, player2: str) -> List[str]:
        """Convert overview table to natural language text"""
        text = []
        text.append("OVERVIEW STATISTICS:")
        text.append("-" * 20)
        
        # Find header row
        headers = []
        data_rows = []
        
        for row in rows:
            label = row.get('label', '')
            if 'STATS OVERVIEW' in label:
                headers = row.get('values', [])
            else:
                data_rows.append(row)
        
        # Convert data rows
        for row in data_rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                # Determine player
                if self.player1 and self.player1 in label:
                    player = self.player1
                elif self.player2 and self.player2 in label:
                    player = self.player2
                else:
                    # Try dynamic initials extraction as fallback
                    pass
                    player = "Unknown Player"
                    if self.player1 and self.player2:
                        player1_initials = ''.join([word[0] for word in self.player1.split()]).upper()
                        player2_initials = ''.join([word[0] for word in self.player2.split()]).upper()
                        if player1_initials in label:
                            player = self.player1
                        elif player2_initials in label:
                            player = self.player2
                
                # Extract key stats
                ace_pct = values[0] if len(values) > 0 else "0%"
                df_pct = values[1] if len(values) > 1 else "0%"
                first_serve_pct = values[2] if len(values) > 2 else "0%"
                first_serve_won_pct = values[3] if len(values) > 3 else "0%"
                second_serve_won_pct = values[4] if len(values) > 4 else "0%"
                break_points_saved = values[5] if len(values) > 5 else "0"
                return_points_won_pct = values[6] if len(values) > 6 else "0%"
                winners = values[7] if len(values) > 7 else "0"
                unforced_errors = values[8] if len(values) > 8 else "0"
                
                text.append(f"{player} had {ace_pct} aces.")
                text.append(f"{player} had {df_pct} double faults.")
                text.append(f"{player} made {first_serve_pct} of first serves.")
                text.append(f"{player} won {first_serve_won_pct} of first serve points.")
                text.append(f"{player} won {second_serve_won_pct} of second serve points.")
                text.append(f"{player} saved {break_points_saved} break points.")
                text.append(f"{player} won {return_points_won_pct} of return points.")
                # Parse the winners and unforced errors to extract forehand/backhand breakdowns
                winners_total = winners.split(" (")[0] if " (" in winners else winners
                winners_fh_bh = winners.split("(")[1].split(")")[0] if "(" in winners else "0/0"
                winners_fh, winners_bh = winners_fh_bh.split("/") if "/" in winners_fh_bh else ("0", "0")
                
                ufe_total = unforced_errors.split(" (")[0] if " (" in unforced_errors else unforced_errors
                ufe_fh_bh = unforced_errors.split("(")[1].split(")")[0] if "(" in unforced_errors else "0/0"
                ufe_fh, ufe_bh = ufe_fh_bh.split("/") if "/" in ufe_fh_bh else ("0", "0")
                
                text.append(f"{player} hit {winners_total} winners, {winners_fh} forehand and {winners_bh} backhand.")
                text.append(f"{player} made {ufe_total} unforced errors, including {ufe_fh} on the forehand and {ufe_bh} on the backhand.")
        
        return text



    def _convert_generic_table(self, table_name: str, rows: List[Dict[str, Any]]) -> List[str]:
        """Convert generic table to natural language text"""
        text = []
        text.append(f"{table_name.upper()} STATISTICS:")
        text.append("-" * len(table_name + " STATISTICS"))
        
        for row in rows:
            label = row.get('label', '')
            values = row.get('values', [])
            
            if label and values:
                text.append(f"{label}: {', '.join(values)}")
        
        return text


