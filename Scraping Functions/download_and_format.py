#!/usr/bin/env python3

import requests
import os
import json
import random
import time
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import re

BASE_URL = "https://replay.pokemonshowdown.com/"
JSON_ENDPOINT = "search.json"

def get_listing_url(format_name, sort_mode="newest"):
    # sort_mode: "newest" (default) or "rating"
    sort_param = ""
    if sort_mode == "rating":
        sort_param = "&sort=rating"
    return f"{BASE_URL}{JSON_ENDPOINT}?format={format_name}{sort_param}"

def get_page_json(session, listing_url, page_number):
    url = f"{listing_url}&page={page_number}" if page_number > 1 else listing_url
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch page {page_number}: {e}")
        return []

def save_new_battles(session, page_json, download_dir, already_downloaded_ids, sleep_min=0, sleep_max=3, early_stop_threshold=10):
    new_battles = []
    consecutive_seen = 0
    for battle in tqdm(page_json, desc="Downloading battles", unit="battle"):
        battle_id = battle.get("id")
        if not battle_id or battle_id in already_downloaded_ids:
            consecutive_seen += 1
            if consecutive_seen >= early_stop_threshold:
                tqdm.write(f"[INFO] Early stopping: {consecutive_seen} already-seen battles in a row.")
                break
            continue
        consecutive_seen = 0  # reset if a new one is found
        filename = download_dir / f"{battle_id}.json"
        url = f"{BASE_URL}{battle_id}.json"
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            battle_json = r.json()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(battle_json, f)
            new_battles.append(filename)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to download {battle_id}: {e}")
            continue
        time.sleep(random.uniform(sleep_min, sleep_max))
    return new_battles, (consecutive_seen >= early_stop_threshold)

def parse_showteam(log, player_tag):
    for line in log.split('\n'):
        if line.startswith(f'|showteam|{player_tag}|'):
            team_str = line.split('|', 3)[-1]
            pokes = [p.strip('[]') for p in team_str.split(']') if p.strip()]
            team = []
            for poke in pokes:
                sections = poke.split('|')
                name = sections[0]
                item = sections[2] if len(sections) > 2 else None
                ability = sections[3] if len(sections) > 3 else None
                moves = sections[4].split(',') if len(sections) > 4 else []
                moves = [m.strip() for m in moves if m.strip()]
                team.append({
                    'species': name.strip() if name else None,
                    'item': item.strip() if item else None,
                    'ability': ability.strip() if ability else None,
                    'moves': moves
                })
            return team
    return []

def parse_winner(log, player1, player2):
    for line in log.split('\n'):
        if line.startswith('|win|'):
            winner = line.split('|')[2].strip()
            if winner == player1:
                return 1
            elif winner == player2:
                return 0
    return None

def canonicalize_team(team, max_pokemon=6, max_moves=4):
    sorted_team = sorted(
        [poke for poke in team if poke['species']],
        key=lambda poke: poke['species'] or ""
    )
    while len(sorted_team) < max_pokemon:
        sorted_team.append({'species': None, 'item': None, 'ability': None, 'moves': [None]*max_moves})
    for poke in sorted_team:
        moves = poke.get('moves', [])
        poke['moves'] = moves + [None]*(max_moves - len(moves))
        poke['moves'] = poke['moves'][:max_moves]
    return sorted_team

def team_to_flat_features(team, prefix, max_pokemon=6, max_moves=4):
    features = {}
    for i, poke in enumerate(team):
        idx = i + 1
        features[f'{prefix}_species_{idx}'] = poke['species']
        features[f'{prefix}_item_{idx}'] = poke['item']
        features[f'{prefix}_ability_{idx}'] = poke['ability']
        for m, move in enumerate(poke['moves']):
            features[f'{prefix}_move_{idx}_{m+1}'] = move
    return features

def extract_leads(log, player_tag):
    leads = []
    for line in log.split('\n'):
        if line.startswith(f'|switch|{player_tag}'):
            m = re.match(r'\|switch\|\w+: [^|]*\|([^,|]+),', line)
            if m:
                species = m.group(1).strip()
                leads.append(species)
            if len(leads) == 2:
                break
    while len(leads) < 2:
        leads.append(None)
    return sorted(leads, key=lambda x: (x is None, str(x)))

def extract_rating(data):
    return data.get("rating", None)

def detect_disconnect(log):
    for line in log.split('\n'):
        if line.startswith('|l|') or 'disconnected' in line or 'has left' in line:
            return 1
    return 0

def detect_forfeit(log):
    for line in log.split('\n'):
        if 'forfeit' in line.lower():
            return 1
    return 0

def count_turns(log):
    return sum(1 for line in log.split('\n') if line.startswith('|turn|'))

def process_file(json_path, max_pokemon=6, max_moves=4):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log = data['log']
    players = data['players']
    if len(players) != 2:
        return None
    p1_team = canonicalize_team(parse_showteam(log, 'p1'), max_pokemon=max_pokemon, max_moves=max_moves)
    p2_team = canonicalize_team(parse_showteam(log, 'p2'), max_pokemon=max_pokemon, max_moves=max_moves)
    outcome = parse_winner(log, players[0], players[1])
    p1_leads = extract_leads(log, 'p1')
    p2_leads = extract_leads(log, 'p2')
    rating = extract_rating(data)
    disconnect = detect_disconnect(log)
    forfeit = detect_forfeit(log)
    turn_count = count_turns(log)

    return {
        'p1_team': p1_team,
        'p2_team': p2_team,
        'p1_player': players[0],
        'p2_player': players[1],
        'p1_win': outcome,
        'p1_leads': p1_leads,
        'p2_leads': p2_leads,
        'rating': rating,
        'disconnect': disconnect,
        'forfeit': forfeit,
        'turn_count': turn_count,
    }

def build_dataset_from_jsons(directory, max_pokemon=6, max_moves=4):
    data_rows = []
    for file in glob.glob(os.path.join(directory, '*.json')):
        game = process_file(file, max_pokemon=max_pokemon, max_moves=max_moves)
        if game is None:
            print(f"[SKIP] Could not process: {file}")
            continue
        row = {}
        row.update(team_to_flat_features(game['p1_team'], 'p1', max_pokemon=max_pokemon, max_moves=max_moves))
        row.update(team_to_flat_features(game['p2_team'], 'p2', max_pokemon=max_pokemon, max_moves=max_moves))
        row['p1_win'] = game['p1_win']
        row['p1_player'] = game['p1_player']
        row['p2_player'] = game['p2_player']
        row['p1_lead_1'] = game['p1_leads'][0]
        row['p1_lead_2'] = game['p1_leads'][1]
        row['p2_lead_1'] = game['p2_leads'][0]
        row['p2_lead_2'] = game['p2_leads'][1]
        row['rating'] = game['rating']
        row['disconnect'] = game['disconnect']
        row['forfeit'] = game['forfeit']
        row['turn_count'] = game['turn_count']
        data_rows.append(row)
    df = pd.DataFrame(data_rows)
    return df

def get_existing_ids(directory):
    return set(f.stem for f in Path(directory).glob("*.json"))

def main(format_name, output_dir, csv_path, sort_mode="newest", only_new=True):
    download_dir = Path(output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    listing_url = get_listing_url(format_name, sort_mode=sort_mode)
    session = requests.Session()
    page_number = 1
    total_downloaded = 0

    already_downloaded_ids = get_existing_ids(download_dir)

    print(f"Starting extraction for format: {format_name} (sort: {sort_mode})")
    while True:
        print(f"[INFO] Fetching results page {page_number}...")
        page_json = get_page_json(session, listing_url, page_number)
        if not page_json:
            print("[INFO] No data retrieved, finishing.")
            break

        new_files, early_stopped = save_new_battles(
            session, page_json, download_dir, already_downloaded_ids,
            early_stop_threshold=10
        )
        total_downloaded += len(new_files)
        already_downloaded_ids.update([f.stem for f in [Path(f) for f in new_files]])

        if early_stopped:
            print(f"[INFO] Early stop triggered: {total_downloaded} new battles downloaded this run.")
            break

        if len(page_json) < 51:
            print(f"[INFO] Extraction complete. Total new battles downloaded: {total_downloaded}")
            break
        page_number += 1

    print("[INFO] Building dataset...")
    df = build_dataset_from_jsons(output_dir)
    print(f"[INFO] Writing {len(df)} rows to {csv_path}")
    df.to_csv(csv_path, index=False)
    print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Pokémon Showdown battle replays to CSV.")
    parser.add_argument("--format", type=str, default="gen9vgc2025regibo3", help="Format name (e.g., gen9vgc2025regibo3)")
    parser.add_argument("--output", type=str, default="replays", help="Directory to save replays")
    parser.add_argument("--csv", type=str, default="battle_data.csv", help="CSV output file path")
    parser.add_argument("--sort", type=str, choices=["newest", "rating"], default="newest", help="Sort battles by 'newest' (default) or 'rating'")
    args = parser.parse_args()
    main(args.format, args.output, args.csv, sort_mode=args.sort)