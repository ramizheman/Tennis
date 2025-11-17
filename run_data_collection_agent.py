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
        # Skip internal metadata keys (not meant for table extraction)
        if k == '_pointlog_ordered_list':
            continue
        # Skip non-string values (lists, dicts, etc.)
        if not isinstance(v, str):
            continue
        if '<table' in v.lower():
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
                # Skip rows that contain player names and look like point descriptions (not stats)
                # Use actual player names instead of hardcoded list
                player_name_parts = []
                for player in [player1, player2]:
                    player_name_parts.extend(player.split())
                
                if any(part in after for part in player_name_parts) and any(t in v.lower() for t in ['serve', 'winner', 'unforced', 'forced', 'rally']):
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
    
    # DEBUG: Check if _pointlog_ordered_list exists
    print(f"\n[DEBUG] '_pointlog_ordered_list' exists: {'_pointlog_ordered_list' in match_details}")
    if '_pointlog_ordered_list' in match_details:
        print(f"[DEBUG] Number of points in ordered list: {len(match_details['_pointlog_ordered_list'])}")
    else:
        print(f"[DEBUG] Available keys with 'point': {[k for k in match_details.keys() if 'point' in k.lower()][:10]}")
    
    # Extract point-by-point data from the collected details
    pointlog_rows = []
    if isinstance(match_details, dict):
        # CRITICAL: Use ordered list from scraper if available (preserves HTML table order)
        # This is the most reliable source as it maintains the exact order from the source HTML
        if '_pointlog_ordered_list' in match_details:
            point_entries = match_details['_pointlog_ordered_list']
            print(f"[OK] Using ordered point list from scraper ({len(point_entries)} points)")
            # Debug: Show first few and last few points to verify extraction
            if len(point_entries) > 0:
                print(f"[DEBUG] First point: {point_entries[0]}")
                if len(point_entries) > 1:
                    print(f"[DEBUG] Last point: {point_entries[-1]}")
        else:
            # Fallback: collect from dictionary keys (may not be in correct order)
            point_entries = []
            for key, value in match_details.items():
                if key.startswith('Point-by-point -'):
                    # Extract the point data from the key format: "Point-by-point - {server} {sets} {games} {points}"
                    parts = key.replace('Point-by-point - ', '').split(' ')
                    if len(parts) >= 4:
                        server = parts[0]  # Extract server name
                        sets = parts[1]   # Extract sets score
                        games = parts[2]  # Extract games score
                        points = parts[3] # Extract points score
                        point_description = value
                        
                        point_entries.append({
                            'server': server,
                            'sets': sets,
                            'games': games,
                            'points': points,
                            'description': point_description
                        })
            
            # CRITICAL: Sort points chronologically by score (sets, games, points)
            # This ensures correct point numbering regardless of dictionary iteration order
            def sort_key(entry):
                """Sort by sets, then games, then points score"""
                sets_parts = entry['sets'].split('-')
                games_parts = entry['games'].split('-')
                points_parts = entry['points'].split('-')
                
                # Convert to integers for proper sorting (handle special scores like "AD-40")
                try:
                    set1 = int(sets_parts[0]) if sets_parts[0] else 0
                    set2 = int(sets_parts[1]) if len(sets_parts) > 1 and sets_parts[1] else 0
                    game1 = int(games_parts[0]) if games_parts[0] else 0
                    game2 = int(games_parts[1]) if len(games_parts) > 1 and games_parts[1] else 0
                    
                    # Handle point scores (can be "0-0", "15-0", "40-30", "AD-40", etc.)
                    if len(points_parts) == 2:
                        p1_str, p2_str = points_parts[0].strip(), points_parts[1].strip()
                        # Convert point scores to sortable values
                        point_values = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4, 'ad': 4}
                        p1_val = point_values.get(p1_str, 0)
                        p2_val = point_values.get(p2_str, 0)
                    else:
                        p1_val, p2_val = 0, 0
                    
                    return (set1, set2, game1, game2, p1_val, p2_val)
                except (ValueError, IndexError):
                    # Fallback for malformed scores
                    return (0, 0, 0, 0, 0, 0)
            
            point_entries.sort(key=sort_key)
            print(f"[OK] Collected and sorted {len(point_entries)} points from dictionary keys")
        
        # CRITICAL: TennisAbstract's HTML table only contains SELECTED points, not all points
        # The table shows key points or points with descriptions, but skips many routine points
        # We assign sequential numbers based on the order in the HTML table, but this creates gaps
        # Example: If HTML has points at scores 0-0 1-2 30-15, 0-0 1-2 40-15, 0-0 2-2 0-0,
        # we number them 1, 2, 3, but there may be missing points between them
        
        # Assign sequential point numbers in the order they appear (already sorted if from ordered list)
        for i, entry in enumerate(point_entries, 1):
            point_number = i
            formatted_point = f"Point {point_number}: {entry['description']}"
            
            pointlog_rows.append({
                'point_number': point_number,
                'server': entry['server'],
                'sets': entry['sets'],
                'games': entry['games'],
                'points': entry['points'],
                'description': entry['description'],
                'formatted': formatted_point
            })
        
        # Warn if we detect large gaps in score progression (indicates missing points)
        if len(point_entries) > 1:
            # Check for score jumps that suggest missing points
            prev_entry = point_entries[0]
            large_gaps = []
            for i, entry in enumerate(point_entries[1:], 1):
                # If sets/games changed significantly, there might be missing points
                prev_sets = prev_entry['sets'].split('-')
                curr_sets = entry['sets'].split('-')
                prev_games = prev_entry['games'].split('-')
                curr_games = entry['games'].split('-')
                
                try:
                    if (prev_sets[0] != curr_sets[0] or prev_sets[1] != curr_sets[1] or
                        abs(int(prev_games[0]) - int(curr_games[0])) > 1 or
                        abs(int(prev_games[1]) - int(curr_games[1])) > 1):
                        large_gaps.append(i)
                except (ValueError, IndexError):
                    pass
                
                prev_entry = entry
            
            if large_gaps:
                print(f"[WARN] Detected {len(large_gaps)} potential gaps in point sequence. "
                      f"TennisAbstract HTML table only shows selected points, not all points.")
                print(f"[WARN] This is expected - the table contains key points, not every point in the match.")
    
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
    
    print(f"[OK] Data collection completed and saved to: {filename}")
    
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
