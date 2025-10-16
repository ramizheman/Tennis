#!/usr/bin/env python3

import json
import sys
import os
from datetime import datetime
from agents.data_collection_agent import TennisDataCollector

def extract_structured_tables_from_details_flat(details_flat: dict, player1: str, player2: str) -> list:
    """
    Extract structured table data from the flat details using the working method.
    This creates the same details_tables structure that was working for Pegula vs Swiatek.
    """
    # Group data by section (same as working method)
    sections = {}
    for k, v in details_flat.items():
        if isinstance(v, str) and '<table' in v.lower():
            continue  # Skip raw HTML tables
        sec = k.split(" - ")[0] if " - " in k else "Other Data"
        sections.setdefault(sec, []).append((k, v))
    
    # Create details_tables structure (same as working method)
    details_tables_json = []
    for raw_section, fields in sections.items():
        if raw_section.lower() in {"js", "pointlog_js"}:
            continue
        if raw_section.lower() == "point_log":
            continue
        # Skip unstructured Point-by-point table (redundant with pointlog_rows)
        if raw_section.lower() == "point-by-point":
            continue
        
        table = {"name": ("Point-by-point Stats" if raw_section == "point_by_point" else raw_section), "rows": []}
        for k, v in fields:
            after = k.split(" - ", 1)[1] if " - " in k else k
            if raw_section == "point_by_point":
                if any(n in after for n in ['Roger', 'Rafael', 'Iga', 'Jessica', 'Swiatek', 'Pegula', 'Jannik', 'Carlos', 'Sinner', 'Alcaraz']) and any(t in v.lower() for t in ['serve', 'winner', 'unforced', 'forced', 'rally']):
                    continue
            toks = [t.strip() for t in v.split('|')]
            row = {"label": after, "values": toks}
            has_numbers = any(any(ch.isdigit() for ch in t) for t in toks)
            if toks and not has_numbers:
                row["is_header"] = True
            table["rows"].append(row)
        details_tables_json.append(table)
    
    return details_tables_json

def main(player1=None, player2=None, match_index=0):
    print("=== RUNNING DATA COLLECTION AGENT ===")
    print()
    
    # Initialize the data collection agent
    collector = TennisDataCollector()
    
    # Require player names to be provided
    if not player1 or not player2:
        print("Error: Both player names must be provided.")
        print("Usage: python run_data_collection_agent.py 'Player1 Name' 'Player2 Name' [match_index]")
        return None
    
    print(f"Searching for matches between {player1} and {player2}...")
    
    # Search for matches
    matches = collector.search_player_matches(player1, player2)
    
    if not matches:
        print(f"No matches found between {player1} and {player2}.")
        return None
    
    print(f"\nFound {len(matches)} matches:")
    print()
    
    for i, match in enumerate(matches):
        date = match.get('date', 'N/A')
        tournament = match.get('tournament', 'N/A')
        p1 = match.get('player1', 'N/A')
        p2 = match.get('player2', 'N/A')
        tour = match.get('tour', 'N/A')
        url = match.get('url', 'N/A')
        
        print(f"{i}: {date} - {tournament}")
        print(f"   {p1} vs {p2} ({tour})")
        print(f"   URL: {url}")
        print()
    
    # Get match details for the selected match
    if match_index >= len(matches):
        print(f"Error: Match index {match_index} is out of range. Found {len(matches)} matches.")
        return None
        
    selected_match = matches[match_index]
    print(f"Getting details for: {selected_match['date']} - {selected_match['tournament']}")
    print(f"{selected_match['player1']} vs {selected_match['player2']}")
    
    # Collect match data
    print(f"\nCollecting detailed match data...")
    match_details = collector.get_match_details(selected_match['url'])
    
    if not match_details:
        print("Error: Could not collect match data.")
        return None
    
    # Extract point-by-point data from the collected details
    pointlog_rows = []
    if isinstance(match_details, dict):
        # Look for point-by-point data in the collected details
        for key, value in match_details.items():
            if key.startswith('Point-by-point -'):
                # Extract the point data from the key format: "Point-by-point - {server} {sets} {games} {points}"
                parts = key.replace('Point-by-point - ', '').split(' ')
                if len(parts) >= 4:
                    # Create the exact format needed for the narrative
                    point_number = len(pointlog_rows) + 1  # Sequential numbering
                    point_description = value
                    server = parts[0]  # Extract server name
                    sets = parts[1]   # Extract sets score
                    games = parts[2]  # Extract games score
                    points = parts[3] # Extract points score
                    
                    # Format: "Point {number}: {description}"
                    formatted_point = f"Point {point_number}: {point_description}"
                    
                    pointlog_rows.append({
                        'point_number': point_number,
                        'server': server,
                        'sets': sets,
                        'games': games,
                        'points': points,
                        'description': point_description,
                        'formatted': formatted_point
                    })
    
    # If no point-by-point data was found, try to extract from other sources
    if not pointlog_rows:
        # Look for any data that might contain point descriptions
        for key, value in match_details.items():
            if 'point' in key.lower() and isinstance(value, str) and len(value) > 20:
                # Try to parse this as point data
                if any(keyword in value.lower() for keyword in ['serve', 'return', 'winner', 'error', 'ace', 'fault']):
                    point_number = len(pointlog_rows) + 1
                    pointlog_rows.append({
                        'point_number': point_number,
                        'description': value,
                        'formatted': f"Point {point_number}: {value}"
                    })
    
    # Extract structured table data from the flat details
    print(f"\nExtracting structured table data from collected details...")
    details_tables = extract_structured_tables_from_details_flat(match_details, player1, player2)
    
    # Create the same structure as the working JSON
    match_data = {
        'basic': {
            'date': selected_match.get('date', '2025-06-28'),
            'tournament': selected_match.get('tournament', 'Bad Homburg F'),
            'player1': selected_match.get('player1', player1),
            'player2': selected_match.get('player2', player2),
            'tour': selected_match.get('tour', 'WTA'),
            'url': selected_match.get('url', ''),
            'raw_text': f"{selected_match.get('date', '2025-06-28')} {selected_match.get('tournament', 'Bad Homburg F')}: {selected_match.get('player1', player1)} vs {selected_match.get('player2', player2)} ({selected_match.get('tour', 'WTA')})"
        },
        'details_flat': match_details,  # Keep original flat data for reference
        'details_tables': details_tables,  # Now properly structured
        'pointlog_rows': pointlog_rows  # Use the extracted point-by-point data
    }
    
    # Save to JSON
    output_data = {
        'matches': [match_data]
    }
    
    # Create filename based on players and actual match date
    player1_clean = player1.replace(' ', '_').replace('.', '')
    player2_clean = player2.replace(' ', '_').replace('.', '')
    
    # Get the actual match date from the selected match
    match_date = selected_match.get('date', '2025-06-28')
    date_clean = match_date.replace('-', '')  # Convert 2025-06-28 to 20250628
    
    # Create dynamic filename based on actual player names and match date
    filename = f"{player1_clean}_{player2_clean}_{date_clean}.json"
    
    # DEBUG: Check what Match Result is before saving
    print(f"DEBUG: Match Result before saving to JSON: {repr(match_details.get('Match Result', 'NOT FOUND'))}")
    
    # Save the JSON file in the root directory (where chat interface expects it)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Data collection completed and saved to: {filename}")
    
    # Show what was collected
    print(f"\n=== COLLECTED DATA SUMMARY ===")
    print(f"Total keys in match_details: {len(match_details)}")
    
    # Show some sample keys
    print(f"\nSample keys collected:")
    for i, key in enumerate(list(match_details.keys())[:10]):
        print(f"  {i+1}: {key}")
    
    if len(match_details) > 10:
        print(f"  ... and {len(match_details) - 10} more keys")
    
    # Check for pointlog data specifically
    pointlog_keys = [key for key in match_details.keys() if 'pointlog' in key.lower()]
    print(f"\nPointlog-related keys found: {len(pointlog_keys)}")
    for key in pointlog_keys:
        print(f"  - {key}")
    
    return filename

if __name__ == "__main__":
    # Require command line arguments for player names
    if len(sys.argv) < 3:
        print("Error: Both player names must be provided.")
        print("Usage: python run_data_collection_agent.py 'Player1 Name' 'Player2 Name' [match_index]")
        print("Example: python run_data_collection_agent.py 'Iga Swiatek' 'Jessica Pegula' 0")
        sys.exit(1)
    
    player1 = sys.argv[1]
    player2 = sys.argv[2]
    match_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(player1, player2, match_index)
