import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from typing import List, Dict, Optional
from crewai import Agent
from config import TENNIS_ABSTRACT_BASE_URL, TENNIS_ABSTRACT_CHARTING_URL

def clean_unicode_text(text: str) -> str:
    """
    Clean Unicode characters that cause issues in JSON export
    """
    if not text:
        return text
    
    # Replace common Unicode issues
    text = text.replace('\u00c2\u00a0', ' ')  # Non-breaking space
    text = text.replace('â€', '-')  # Em dash
    text = text.replace('â€˜', '-')  # En dash
    text = text.replace('\u2011', '-')  # Non-breaking hyphen
    text = text.replace('‑', '-')  # Another type of hyphen
    text = text.replace('Â', '')  # Extra space character
    
    return text.strip()

class TennisDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def search_player_matches(self, player1: str, player2: str) -> List[Dict]:
        """
        Search for matches between two players on TennisAbstract.com
        """
        matches = []
        
        try:
            # Get the main charting page
            response = self.session.get(TENNIS_ABSTRACT_CHARTING_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all match entries
            match_entries = soup.find_all('a', href=True)
            
            for entry in match_entries:
                href = entry.get('href', '')
                text = clean_unicode_text(entry.get_text(strip=True))
                
                # Look for matches between the two players
                if self._is_match_between_players(text, player1, player2):
                    match_data = self._extract_match_data(text, href)
                    if match_data:
                        matches.append(match_data)
            
            # Also search for individual player pages
            player1_matches = self._search_player_page(player1, player2)
            player2_matches = self._search_player_page(player2, player1)
            
            matches.extend(player1_matches)
            matches.extend(player2_matches)
            
            # Remove duplicates
            unique_matches = self._remove_duplicate_matches(matches)
            
            return unique_matches
            
        except Exception as e:
            print(f"Error collecting data: {e}")
            return []
    
    def _is_match_between_players(self, text: str, player1: str, player2: str) -> bool:
        """
        Check if the match text contains both players exactly
        """
        player1_lower = player1.lower()
        player2_lower = player2.lower()
        text_lower = text.lower()
        
        # Look for exact player names in the text
        # Handle "vs" format: "Player1 vs Player2"
        vs_pattern = f"{player1_lower} vs {player2_lower}"
        vs_pattern_reverse = f"{player2_lower} vs {player1_lower}"
        
        # Also check for just the names being present (in case of different formatting)
        has_player1_exact = player1_lower in text_lower
        has_player2_exact = player2_lower in text_lower
        
        # Must have both players exactly
        return has_player1_exact and has_player2_exact
    
    def _extract_match_data(self, text: str, href: str) -> Optional[Dict]:
        """
        Extract match data from the text
        """
        try:
            # Parse match information
            # Format: "YYYY-MM-DD Tournament Round: Player1 vs Player2 (Tour)"
            match_pattern = r'(\d{4}-\d{2}-\d{2})\s+([^:]+):\s+([^v]+)\s+vs\s+([^(]+)\s+\(([^)]+)\)'
            match = re.search(match_pattern, text)
            
            if match:
                date, tournament, player1, player2, tour = match.groups()
                return {
                    'date': date,
                    'tournament': tournament.strip(),
                    'player1': player1.strip(),
                    'player2': player2.strip(),
                    'tour': tour.strip(),
                    'url': TENNIS_ABSTRACT_BASE_URL + '/charting/' + href if not href.startswith('http') else href,
                    'raw_text': text
                }
        except Exception as e:
            print(f"Error extracting match data: {e}")
        
        return None
    
    def _search_player_page(self, player: str, opponent: str) -> List[Dict]:
        """
        Search individual player pages for matches against specific opponent
        """
        matches = []
        try:
            # Construct player search URL
            player_search_url = f"{TENNIS_ABSTRACT_BASE_URL}/cgi-bin/player.cgi?p={player.replace(' ', '+')}"
            response = self.session.get(player_search_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for match links
            match_links = soup.find_all('a', href=re.compile(r'/charting/'))
            
            for link in match_links:
                text = clean_unicode_text(link.get_text(strip=True))
                if self._is_match_between_players(text, player, opponent):
                    match_data = self._extract_match_data(text, link.get('href', ''))
                    if match_data:
                        matches.append(match_data)
        
        except Exception as e:
            print(f"Error searching player page for {player}: {e}")
        
        return matches
    
    def _remove_duplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Remove duplicate matches based on date and players
        """
        seen = set()
        unique_matches = []
        
        for match in matches:
            key = f"{match['date']}_{match['player1']}_{match['player2']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    def get_match_details(self, match_url: str) -> Dict:
        """
        Get detailed match statistics from a specific match page
        """
        try:
            response = self.session.get(match_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match statistics
            stats = {}
            
            # Extract match result from the page
            match_result = self._extract_match_result(soup)
            if match_result:
                stats['Match Result'] = match_result
            else:
                # If we couldn't extract it from the page, try to find it in the JavaScript data later
                stats['Match Result'] = 'Unknown'
            
            # Define all the specific sections we want to extract
            section_links = [
                'Stats Overview',
                'Serve Statistics Overview', 
                'Serve Influence',
                'Key point outcomes',
                'Point outcomes by rally length',
                'Point-by-point description',
                'Serve Breakdown',
                'Return Breakdown', 
                'Net Points',
                'Shot Types',
                'Shot Direction'
            ]
            
            # Extract data from each section link
            for section in section_links:
                section_data = self._extract_section_data(soup, section)
                if section_data:
                    for key, value in section_data.items():
                        stats[f"{section} - {key}"] = value
            
            # NOTE: Legacy player-specific section extraction removed
            # Modern Tennis Abstract uses JavaScript variables (serve1, serve2, etc.) instead of
            # player-named sections like "Player Name: Serve Breakdown"
            
            # Look for JavaScript data that contains the actual statistics
            scripts = soup.find_all('script')
            for script in scripts:
                script_text = script.get_text()
                if script_text:
                    # Look for JavaScript variables that contains the actual statistics
                    # Pattern: var variableName = '<pre><table>...</table></pre>';
                    table_patterns = re.findall(r'var\s+(\w+)\s*=\s*[\'"](<pre>.*?</pre>)[\'"]', script_text, re.DOTALL)
                    
                    for var_name, table_html in table_patterns:
                        # Parse the table HTML
                        table_soup = BeautifulSoup(table_html, 'html.parser')
                        
                        # Extract table data - handle multiple tables within the pre tag
                        tables = table_soup.find_all('table')
                        print(f"Found {len(tables)} tables in {var_name}")
                        
                        for table_idx, table in enumerate(tables):
                            rows = table.find_all('tr')
                            current_table_name = None
                            
                            for row in rows:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    # First cell is usually the key/label
                                    key = clean_unicode_text(cells[0].get_text(strip=True))
                                    if key and len(key) < 50:
                                        # Get all other cells as values
                                        values = []
                                        for cell in cells[1:]:
                                            cell_text = clean_unicode_text(cell.get_text(strip=True))
                                            if cell_text:
                                                values.append(cell_text)
                                        
                                        if values:
                                            # Check if this is a table header row
                                            is_header = (
                                                key.isupper() or 
                                                any(keyword in key.upper() for keyword in ['OUTCOMES', 'DEPTH', 'BREAKDOWN', 'OVERVIEW', 'DIRECTION']) or
                                                cells[0].name == 'th'
                                            )
                                            
                                            # For overview sections, handle player data specially
                                            # For overview sections, handle player data specially (any player name)
                                            if var_name == 'overview' and key:
                                                # Check if we've already processed this player
                                                if f"{var_name} - {key}" not in stats:
                                                    # This is the first occurrence (overall match totals)
                                                    data_key = f"{var_name} - {key}"
                                                    stats[data_key] = " | ".join(values)
                                                # Skip subsequent occurrences (individual sets)
                                            else:
                                                # Process all other rows normally
                                                if is_header:
                                                    # This is a table header - store the table name
                                                    current_table_name = key
                                                    # Create a special key for the header
                                                    table_suffix = f"_table{table_idx+1}" if len(tables) > 1 else ""
                                                    header_key = f"{var_name} - {key}{table_suffix}"
                                                    stats[header_key] = " | ".join(values)
                                                else:
                                                    # This is a data row - use table-specific key
                                                    table_suffix = f"_table{table_idx+1}" if len(tables) > 1 else ""
                                                    data_key = f"{var_name} - {key}{table_suffix}"
                                                    stats[data_key] = " | ".join(values)
                    
                    # Look specifically for the pointlog variable that contains point-by-point description data
                    # Robustly capture the full string assigned to var pointlog (handles escaped quotes)
                    pointlog_start = script_text.find('var pointlog =')
                    if pointlog_start != -1:
                        # Find the first quote character after the equals
                        after_eq_idx = script_text.find('=', pointlog_start) + 1
                        # Skip whitespace
                        while after_eq_idx < len(script_text) and script_text[after_eq_idx] in [' ', '\t', '\r', '\n']:
                            after_eq_idx += 1
                        if after_eq_idx < len(script_text) and script_text[after_eq_idx] in ['\'', '"']:
                            quote_char = script_text[after_eq_idx]
                            i = after_eq_idx + 1
                            escaped = False
                            buf = []
                            while i < len(script_text):
                                ch = script_text[i]
                                if escaped:
                                    # Keep escaped content as-is (e.g., \' or \n)
                                    buf.append(ch)
                                    escaped = False
                                else:
                                    if ch == '\\':
                                        escaped = True
                                    elif ch == quote_char:
                                        break
                                    else:
                                        buf.append(ch)
                                i += 1
                            pointlog_data = ''.join(buf)
                            if pointlog_data:
                                pointlog_soup = BeautifulSoup(pointlog_data, 'html.parser')
                                rows = pointlog_soup.find_all('tr')
                                for row in rows:
                                    cells = row.find_all(['td', 'th'])
                                    if len(cells) >= 5:
                                        server = clean_unicode_text(cells[0].get_text(strip=True))
                                        sets = clean_unicode_text(cells[1].get_text(strip=True))
                                        games = clean_unicode_text(cells[2].get_text(strip=True))
                                        points = clean_unicode_text(cells[3].get_text(strip=True))
                                        description = clean_unicode_text(cells[4].get_text(strip=True))
                                        # Skip completely empty separators
                                        if server and description:
                                            stats[f"Point-by-point - {server} {sets} {games} {points}"] = description
                    
                    # Also look for any JavaScript variable that might contain point-by-point data
                    point_log_patterns = re.findall(r'var\s+(\w+)\s*=\s*[\'\"](.*?point.*?log.*?)[\'\"]', script_text, re.DOTALL | re.IGNORECASE)
                    for var_name, point_log_data in point_log_patterns:
                        # Parse the HTML table instead of storing raw HTML
                        if '<table' in point_log_data:
                            try:
                                table_soup = BeautifulSoup(point_log_data, 'html.parser')
                                tables = table_soup.find_all('table')
                                for table_idx, table in enumerate(tables):
                                    rows = table.find_all('tr')
                                    for row in rows:
                                        cells = row.find_all(['td', 'th'])
                                        if len(cells) >= 2:
                                            key = clean_unicode_text(cells[0].get_text(strip=True))
                                            if key and len(key) < 50:
                                                values = []
                                                for cell in cells[1:]:
                                                    cell_text = clean_unicode_text(cell.get_text(strip=True))
                                                    if cell_text:
                                                        values.append(cell_text)
                                                
                                                if values:
                                                    table_suffix = f"_table{table_idx+1}" if len(tables) > 1 else ""
                                                    data_key = f"{var_name} - {key}{table_suffix}"
                                                    stats[data_key] = " | ".join(values)
                            except Exception as e:
                                print(f"Error parsing point_log table: {e}")
                                # Fallback to storing raw data if parsing fails
                                stats[f"point_log - {var_name}"] = point_log_data.strip()
                        else:
                            stats[f"point_log - {var_name}"] = point_log_data.strip()
                    
                    # NOTE: We intentionally avoid scraping generic <table> blobs into
                    # a fabricated 'point_by_point' section here, because it caused
                    # misclassification of Serve/Return overview tables.
                    # The canonical point-by-point narrative is parsed from 'var pointlog'
                    # above into 'Point-by-point - {server} {sets} {games} {points}'.
                    
                    # Skip simple key-value patterns that create garbage HTML attribute entries
                    # This was creating entries like "js - serve": "<table id=" which is useless
            
            # Look for point-by-point description data in the main page content
            # This data might be directly in the HTML, not in JavaScript
            point_description_section = soup.find('div', string=re.compile(r'Point-by-point description', re.IGNORECASE))
            if point_description_section:
                # Look for the next table or div containing the point-by-point data
                point_data_container = point_description_section.find_next(['table', 'div'])
                if point_data_container:
                    # Extract all text from this container
                    point_text = point_data_container.get_text(separator='\n', strip=True)
                    if point_text and len(point_text) > 50:
                        stats['Point-by-point description - Full Text'] = point_text
            
            # Look for the point-by-point description span element
            pointlog_span = soup.find('span', id='pointlog')
            if pointlog_span:
                print(f"✅ Found pointlog span: {clean_unicode_text(pointlog_span.get_text(strip=True))}")
                
                # Look for JavaScript that handles the pointlog click
                scripts = soup.find_all('script')
                for script in scripts:
                    script_text = script.get_text()
                    if script_text:
                        # Look for JavaScript that handles pointlog clicks or loads point data
                        if 'pointlog' in script_text or 'pointlog' in script_text.lower():
                            print(f"✅ Found JavaScript handling pointlog")
                            
                            # Look for any data that might be loaded for pointlog
                            point_data_patterns = re.findall(r'var\s+(\w+)\s*=\s*[\'"](.*?point.*?log.*?)[\'"]', script_text, re.DOTALL | re.IGNORECASE)
                            for var_name, point_data in point_data_patterns:
                                # Parse the HTML table instead of storing raw HTML
                                if '<table' in point_data:
                                    try:
                                        table_soup = BeautifulSoup(point_data, 'html.parser')
                                        tables = table_soup.find_all('table')
                                        for table_idx, table in enumerate(tables):
                                            rows = table.find_all('tr')
                                            for row in rows:
                                                cells = row.find_all(['td', 'th'])
                                                if len(cells) >= 2:
                                                    key = clean_unicode_text(cells[0].get_text(strip=True))
                                                    if key and len(key) < 50:
                                                        values = []
                                                        for cell in cells[1:]:
                                                            cell_text = clean_unicode_text(cell.get_text(strip=True))
                                                            if cell_text:
                                                                values.append(cell_text)
                                                        
                                                        if values:
                                                            table_suffix = f"_table{table_idx+1}" if len(tables) > 1 else ""
                                                            data_key = f"{var_name} - {key}{table_suffix}"
                                                            stats[data_key] = " | ".join(values)
                                    except Exception as e:
                                        print(f"Error parsing pointlog_js table: {e}")
                                        # Skip storing raw data to avoid garbage
                                else:
                                    # Only store non-HTML data
                                    if not any(tag in point_data for tag in ['<table', '<div', '<span', '<td', '<th']):
                                        stats[f"pointlog_js - {var_name}"] = point_data.strip()
                            
                            # Look for AJAX calls or data loading for pointlog
                            ajax_patterns = re.findall(r'\.get\([\'"](.*?point.*?)[\'"]', script_text, re.IGNORECASE)
                            for ajax_url in ajax_patterns:
                                print(f"Found potential AJAX URL: {ajax_url}")
                                # Try to fetch this URL
                                try:
                                    if not ajax_url.startswith('http'):
                                        ajax_url = f"https://tennisabstract.com{ajax_url}"
                                    ajax_response = self.session.get(ajax_url)
                                    if ajax_response.status_code == 200:
                                        ajax_content = ajax_response.text
                                        if re.search(r'Roger Federer.*0‑0.*0‑0.*0‑0.*1st serve', ajax_content):
                                            stats['Point-by-point description - AJAX Data'] = ajax_content
                                            print(f"✅ Found point-by-point data via AJAX!")
                                except Exception as e:
                                    print(f"Error fetching AJAX URL: {e}")
            
            # Also look for any table that contains descriptive point data
            all_tables = soup.find_all('table')
            for i, table in enumerate(all_tables):
                table_text = table.get_text(separator='\n', strip=True)
                # Check if this table contains descriptive point data (like "1st serve down the T, ace")
                if re.search(r'\d+[‑-]\d+.*[a-zA-Z].*serve.*[a-zA-Z]', table_text):
                    stats[f'Point-by-point Table {i+1}'] = table_text
            
            # Look for any div or section that contains point-by-point descriptive data
            all_divs = soup.find_all('div')
            for i, div in enumerate(all_divs):
                div_text = div.get_text(separator='\n', strip=True)
                # Check if this div contains point-by-point descriptive data
                if re.search(r'(Roger Federer|Rafael Nadal).*\d+[‑-]\d+.*\d+[‑-]\d+.*\d+[‑-]\d+', div_text):
                    if len(div_text) > 100:  # Only if it's substantial
                        stats[f'Point-by-point Div {i+1}'] = div_text
            
            # Clean up the match result if it's in a messy format
            if 'Match Result' in stats:
                match_result = stats['Match Result']
                # Look for patterns like "Player1 d. Player2 6-4 6-2 6-1" or with tiebreaks "6-7(5)"
                clean_result_pattern = re.search(r'(\w+\s+\w+)\s+d\.\s+(\w+\s+\w+)\s+((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)', match_result)
                if clean_result_pattern:
                    winner = clean_result_pattern.group(1)
                    loser = clean_result_pattern.group(2)
                    score = clean_result_pattern.group(3).strip()
                    stats['Match Result'] = f"{winner} d. {loser} {score}"
                # Also look for other patterns (keep the full result if it looks good)
                elif 'd.' in match_result and re.search(r'\d+-\d+', match_result):
                    # If it already looks clean, don't modify it
                    pass
            
            return stats
            
        except Exception as e:
            print(f"Error getting match details: {e}")
            return {}
    
    def _extract_section_data(self, soup: BeautifulSoup, section_name: str) -> Dict:
        """
        Extract data from a specific section link
        """
        section_data = {}
        
        try:
            # Look for the section link
            section_elem = soup.find('a', text=re.compile(section_name, re.IGNORECASE))
            if section_elem:
                # Get the href to the section data
                section_href = section_elem.get('href', '')
                if section_href:
                    # Construct the full URL
                    section_url = section_href if section_href.startswith('http') else f"{TENNIS_ABSTRACT_BASE_URL}{section_href}"
                    
                    # Get the section data
                    section_response = self.session.get(section_url, timeout=10)
                    if section_response.status_code == 200:
                        section_soup = BeautifulSoup(section_response.content, 'html.parser')
                        
                        # Extract table data from the section
                        tables = section_soup.find_all('table')
                        serve_overview_mode = 'serve statistics overview' in section_name.lower()
                        
                        # For overview sections, prioritize overall match totals over individual set data
                        if 'overview' in section_name.lower():
                            # Process tables in order, but prioritize the first table (overall match totals)
                            for table_idx, table in enumerate(tables):
                                rows = table.find_all('tr')
                                current_table_name = None
                                current_table_data = []
                                
                                for row in rows:
                                    cells = row.find_all(['td', 'th'])
                                    if not cells:
                                        continue
                                    
                                    # Check if this row looks like a header row
                                    header_text = clean_unicode_text(cells[0].get_text(strip=True)) if cells else ''
                                    is_header_row = (
                                        cells[0].name == 'th' or 
                                        header_text.isupper() or 
                                        header_text.endswith('BASICS') or 
                                        header_text in {'DIRECTION', 'BREAKDOWN', 'OUTCOMES', 'DEPTH'} or
                                        any(keyword in header_text.upper() for keyword in ['BREAKDOWN', 'OVERVIEW', 'STATISTICS', 'KEY POINTS', 'OUTCOMES', 'DEPTH'])
                                    )
                                    
                                    if is_header_row:
                                        # If we have accumulated data from a previous table, emit it
                                        if current_table_name and current_table_data:
                                            for data_row in current_table_data:
                                                section_data[data_row['key']] = data_row['value']
                                        
                                        # Start new table
                                        current_table_name = header_text
                                        current_table_data = []
                                        
                                        # Emit the header row
                                        header_values = []
                                        for cell in cells:
                                            t = clean_unicode_text(cell.get_text(strip=True))
                                            if t:
                                                header_values.append(t)
                                        if len(header_values) > 1:
                                            section_data[f"{current_table_name}"] = ' | '.join(header_values[1:])
                                        continue
                                    
                                    # This is a data row
                                    if len(cells) >= 2 and current_table_name:
                                        key = clean_unicode_text(cells[0].get_text(strip=True))
                                        if key and len(key) < 80:
                                            values = []
                                            for cell in cells[1:]:
                                                cell_text = clean_unicode_text(cell.get_text(strip=True))
                                                if cell_text:
                                                    values.append(cell_text)
                                            if values:
                                                # For overview sections, only use the first table (overall match totals)
                                                # Skip individual set data (Set 1, Set 2, etc.)
                                                if table_idx == 0 or not any(set_indicator in key for set_indicator in ['SET 1', 'SET 2', 'SET 3', 'SET 4', 'SET 5']):
                                                    data_key = f"{key}"
                                                    data_value = ' | '.join(values)
                                                    
                                                    # Store for later emission (to avoid overwriting)
                                                    current_table_data.append({
                                                        'key': data_key,
                                                        'value': data_value
                                                    })
                                
                                # Emit any remaining data from the last table
                                if current_table_name and current_table_data:
                                    for data_row in current_table_data:
                                        section_data[data_row['key']] = data_row['value']
                                
                                # For overview sections, only process the first table (overall match totals)
                                if table_idx == 0:
                                    break
                        else:
                            # Process all tables in the section (non-overview sections)
                            for table_idx, table in enumerate(tables):
                                rows = table.find_all('tr')
                                current_table_name = None
                                current_table_data = []
                                
                                for row in rows:
                                    cells = row.find_all(['td', 'th'])
                                    if not cells:
                                        continue
                                    
                                    # Check if this row looks like a header row
                                    header_text = clean_unicode_text(cells[0].get_text(strip=True)) if cells else ''
                                    is_header_row = (
                                        cells[0].name == 'th' or 
                                        header_text.isupper() or 
                                        header_text.endswith('BASICS') or 
                                        header_text in {'DIRECTION', 'BREAKDOWN', 'OUTCOMES', 'DEPTH'} or
                                        any(keyword in header_text.upper() for keyword in ['BREAKDOWN', 'OVERVIEW', 'STATISTICS', 'KEY POINTS', 'OUTCOMES', 'DEPTH'])
                                    )
                                    
                                    if is_header_row:
                                        # If we have accumulated data from a previous table, emit it
                                        if current_table_name and current_table_data:
                                            for data_row in current_table_data:
                                                section_data[data_row['key']] = data_row['value']
                                        
                                        # Start new table
                                        current_table_name = header_text
                                        current_table_data = []
                                        
                                        # Emit the header row
                                        header_values = []
                                        for cell in cells:
                                            t = clean_unicode_text(cell.get_text(strip=True))
                                            if t:
                                                header_values.append(t)
                                        if len(header_values) > 1:
                                            section_data[f"{current_table_name}"] = ' | '.join(header_values[1:])
                                        continue
                                    
                                    # This is a data row
                                    if len(cells) >= 2 and current_table_name:
                                        key = clean_unicode_text(cells[0].get_text(strip=True))
                                        if key and len(key) < 80:
                                            values = []
                                            for cell in cells[1:]:
                                                cell_text = clean_unicode_text(cell.get_text(strip=True))
                                                if cell_text:
                                                    values.append(cell_text)
                                            if values:
                                                # Create table-specific key
                                                table_suffix = f"_table{table_idx+1}" if len(tables) > 1 else ""
                                                data_key = f"{key}{table_suffix}"
                                                data_value = ' | '.join(values)
                                                
                                                # Store for later emission (to avoid overwriting)
                                                current_table_data.append({
                                                    'key': data_key,
                                                    'value': data_value
                                                })
                                
                                # Emit any remaining data from the last table
                                if current_table_name and current_table_data:
                                    for data_row in current_table_data:
                                        section_data[data_row['key']] = data_row['value']
                        
                        # Also look for any text patterns in the section
                        section_texts = section_soup.find_all(text=True)
                        for text in section_texts:
                            text = text.strip()
                            if text and len(text) < 200:
                                # Look for patterns like "X: Y" or "X - Y"
                                if ':' in text and len(text.split(':')) == 2:
                                    key, value = text.split(':', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    if key and value and len(key) < 50:
                                        section_data[key] = value
                        
        except Exception as e:
            print(f"Error extracting section {section_name}: {e}")
        
        return section_data

    def _extract_match_result(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract the match result from the page
        """
        try:
            # Look for common patterns where match results are displayed
            # Pattern 1: Look for text with score format including tiebreaks (e.g., "6-4", "6-7(5)", "6-3")
            score_pattern = re.compile(r'\d+-\d+(?:\(\d+\))?(?:\s+\d+-\d+(?:\(\d+\))?)*')
            
            # Look in various elements where results might be displayed
            # Search in priority order: b/strong tags first (most likely to have full result), then others
            best_result = None
            max_sets = 0
            
            for element in soup.find_all(['b', 'strong', 'h1', 'h2', 'h3', 'h4', 'div', 'span', 'p']):
                text = clean_unicode_text(element.get_text(strip=True))
                if text:
                    # Look for score patterns
                    scores = score_pattern.findall(text)
                    if scores and len(text) > 20:  # Avoid short fragments
                        # Count actual number of sets in the matched score string
                        # Each set is "X-Y" or "X-Y(Z)", so count how many times we see "\d+-\d+"
                        set_count = len(re.findall(r'\d+-\d+', scores[0])) if scores else 0
                        
                        # Check if this looks like a match result
                        if any(keyword in text.lower() for keyword in ['won', 'defeated', 'beat', 'final', 'result', 'd.']):
                            # Keep the result with the MOST sets (most complete)
                            if set_count > max_sets:
                                best_result = text.strip()
                                max_sets = set_count
                        # If it's just scores, try to find context
                        elif set_count >= 3:  # At least 3 sets suggest a complete match result
                            if set_count > max_sets:
                                best_result = text.strip()
                                max_sets = set_count
            
            if best_result:
                return best_result
            
            # Pattern 2: Look for specific result text patterns with tiebreak support
            result_patterns = [
                r'(\w+\s+\w+)\s+d\.\s+(\w+\s+\w+)\s+((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)',  # Carlos Alcaraz d. Jannik Sinner 6-3 6-7(7) 6-7(0) 7-5 6-3
                r'(\w+\s+\w+)\s+(?:won|defeated|beat)\s+(\w+\s+\w+)\s+((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)',
                r'((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)\s+(?:won by|defeated by)\s+(\w+\s+\w+)',
                r'Final:\s+(\w+\s+\w+)\s+((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)',
                r'Result:\s+(\w+\s+\w+)\s+((?:\d+-\d+(?:\(\d+\))?(?:\s+|$))+)'
            ]
            
            for pattern in result_patterns:
                for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'div', 'span', 'p', 'b', 'strong']):
                    text = clean_unicode_text(element.get_text(strip=True))
                    if text:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            return text.strip()
            
            # Pattern 3: Look for the page title or main heading
            title = soup.find('title')
            if title:
                title_text = clean_unicode_text(title.get_text(strip=True))
                if score_pattern.search(title_text):
                    return title_text
            
            # Pattern 4: Look for any text that contains both player names and scores
            # This is a fallback for when the result is in a less obvious location
            for element in soup.find_all(['div', 'span', 'p']):
                text = clean_unicode_text(element.get_text(strip=True))
                if text and len(text) < 200:  # Reasonable length for a result
                    if score_pattern.search(text):
                        # Check if this looks like a complete result
                        if any(keyword in text.lower() for keyword in ['vs', 'defeated', 'won', 'final']):
                            return text.strip()
            
            return None
            
        except Exception as e:
            print(f"Error extracting match result: {e}")
            return None

def create_data_collection_agent():
    """
    Create the data collection agent
    """
    return Agent(
        role='Tennis Data Collector',
        goal='Collect comprehensive match data between two tennis players from TennisAbstract.com',
        backstory="""You are an expert tennis data analyst specializing in collecting and organizing 
        historical match data from TennisAbstract.com. You have extensive experience in web scraping 
        and data extraction, particularly from tennis statistics websites. Your expertise lies in 
        finding all matches between two specific players and extracting detailed match statistics.""",
        verbose=True,
        allow_delegation=False,
        tools=[
            {
                "name": "search_player_matches",
                "description": "Search for all matches between two tennis players on TennisAbstract.com",
                "function": lambda player1, player2: TennisDataCollector().search_player_matches(player1, player2)
            },
            {
                "name": "get_match_details",
                "description": "Get detailed statistics for a specific match",
                "function": lambda match_url: TennisDataCollector().get_match_details(match_url)
            }
        ]
    ) 