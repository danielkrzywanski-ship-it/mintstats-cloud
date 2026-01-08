import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import difflib
import random
import requests
from scipy.stats import poisson
from PIL import Image
import pytesseract
import re
import io
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta

# --- KONFIGURACJA ---
st.set_page_config(page_title="MintStats v26.2 Missing Link", layout="wide", page_icon="🔗")
FIXTURES_DB_FILE = "my_fixtures.csv"
COUPONS_DB_FILE = "my_coupons.csv"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .stButton>button { background-color: #00C896; color: white; border-radius: 8px; border: none; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stButton>button:hover { background-color: #00A87E; color: white; }
    h1, h2, h3 { color: #008F7A !important; }
    div[data-testid="stMetricValue"] { color: #008F7A; }
    .streamlit-expanderHeader { background-color: #FFFFFF; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SŁOWNIK ALIASÓW ---
TEAM_ALIASES = {
    "avs": "AFS", "avs futebol": "AFS", "afs": "AFS", "brag": "Sp Braga", "braga": "Sp Braga",
    "sc braga": "Sp Braga", "sporting": "Sp Lisbon", "sporting cp": "Sp Lisbon", "fc porto": "Porto",
    "man utd": "Man United", "man city": "Man City", "leeds": "Leeds", "arsenal": "Arsenal",
    "chelsea": "Chelsea", "liverpool": "Liverpool", "real madrid": "Real Madrid", "barcelona": "Barcelona",
    "bayern": "Bayern Munich", "psg": "Paris SG", "juventus": "Juventus", "inter": "Inter Milan",
    "milan": "Milan", "napoli": "Napoli", "roma": "Roma", "ajax": "Ajax", "feyenoord": "Feyenoord",
    "benfica": "Benfica", "porto": "Porto", "celtic": "Celtic", "rangers": "Rangers",
    "valiadolia": "Valladolid", "betis": "Real Betis", "celta": "Celta", "monchengladbach": "M'gladbach",
    "mainz": "Mainz 05", "frankfurt": "Ein Frankfurt", "parc": "Pau FC", "b tyon": "Lyon",
    "young boys": "Young Boys", "servette": "Servette", "lugano": "Lugano", "basel": "Basel",
    "malmo": "Malmo FF", "aik": "AIK", "djurgarden": "Djurgarden", "rosenborg": "Rosenborg", "bodo/glimt": "Bodo Glimt",
    "legia": "Legia Warsaw", "lech": "Lech Poznan", "rakow": "Rakow Czestochowa"
}

# --- SŁOWNIK LIG ---
LEAGUE_NAMES = {
    'E0': '🇬🇧 Anglia - Premier League', 'E1': '🇬🇧 Anglia - Championship', 'E2': '🇬🇧 Anglia - League One', 'E3': '🇬🇧 Anglia - League Two', 'EC': '🇬🇧 Anglia - Conference',
    'D1': '🇩🇪 Niemcy - Bundesliga', 'D2': '🇩🇪 Niemcy - 2. Bundesliga',
    'I1': '🇮🇹 Włochy - Serie A', 'I2': '🇮🇹 Włochy - Serie B',
    'SP1': '🇪🇸 Hiszpania - La Liga', 'SP2': '🇪🇸 Hiszpania - La Liga 2',
    'F1': '🇫🇷 Francja - Ligue 1', 'F2': '🇫🇷 Francja - Ligue 2',
    'N1': '🇳🇱 Holandia - Eredivisie', 'P1': '🇵🇹 Portugalia - Liga Portugal', 'B1': '🇧🇪 Belgia - Jupiler League', 
    'T1': '🇹🇷 Turcja - Super Lig', 'G1': '🇬🇷 Grecja - Super League',
    'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Szkocja - Premiership', 'SC1': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Szkocja - Championship', 'SC2': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Szkocja - League One', 'SC3': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Szkocja - League Two',
    'SWZ': '🇨🇭 Szwajcaria - Super League', 'S1': '🇨🇭 Szwajcaria - Super League',
    'SWE': '🇸🇪 Szwecja - Allsvenskan', 'SE1': '🇸🇪 Szwecja - Allsvenskan',
    'ROU': '🇷🇴 Rumunia - Liga I', 'R1': '🇷🇴 Rumunia - Liga I',
    'POL': '🇵🇱 Polska - Ekstraklasa', 'PL': '🇵🇱 Polska - Ekstraklasa',
    'NOR': '🇳🇴 Norwegia - Eliteserien', 'NO1': '🇳🇴 Norwegia - Eliteserien',
    'IRL': '🇮🇪 Irlandia - Premier Division',
    'FIN': '🇫🇮 Finlandia - Veikkausliiga', 'FI1': '🇫🇮 Finlandia - Veikkausliiga',
    'DNK': '🇩🇰 Dania - Superliga', 'DK1': '🇩🇰 Dania - Superliga',
    'AUT': '🇦🇹 Austria - Bundesliga', 'A1': '🇦🇹 Austria - Bundesliga',
    'USA': '🇺🇸 USA - MLS',
    'MEX': '🇲🇽 Meksyk - Liga MX',
    'JPN': '🇯🇵 Japonia - J-League',
    'CHN': '🇨🇳 Chiny - Super League',
    'BRA': '🇧🇷 Brazylia - Serie A', 'BR1': '🇧🇷 Brazylia - Serie A',
    'ARG': '🇦🇷 Argentyna - Primera Division'
}

# --- FUNKCJE I KLASY (WŁAŚCIWA KOLEJNOŚĆ) ---

def get_leagues_list():
    try:
        conn = sqlite3.connect("mintstats.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='all_leagues';")
        if cursor.fetchone() is None: return []
        df = pd.read_sql("SELECT DISTINCT LeagueName FROM all_leagues ORDER BY LeagueName", conn)
        conn.close()
        return df['LeagueName'].tolist()
    except: return []

def get_all_data():
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT * FROM all_leagues", conn)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        conn.close()
        return df
    except: return pd.DataFrame()

def get_data_for_league(league_name):
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT * FROM all_leagues WHERE LeagueName = ?", conn, params=(league_name,))
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        conn.close()
        return df
    except: return pd.DataFrame()

def load_fixture_pool():
    if os.path.exists(FIXTURES_DB_FILE):
        try: return pd.read_csv(FIXTURES_DB_FILE).to_dict('records')
        except: return []
    return []

def save_fixture_pool(pool_data):
    if pool_data: pd.DataFrame(pool_data).to_csv(FIXTURES_DB_FILE, index=False)
    else:
        if os.path.exists(FIXTURES_DB_FILE): os.remove(FIXTURES_DB_FILE)

def load_saved_coupons():
    if os.path.exists(COUPONS_DB_FILE):
        try: 
            df = pd.read_csv(COUPONS_DB_FILE)
            coupons = []
            for _, row in df.iterrows():
                try:
                    data_json = row['Data'].replace("'", '"')
                    coupon_data = json.loads(data_json)
                    coupons.append({
                        'id': row['ID'], 'name': row['Name'],
                        'date_created': row['DateCreated'], 'data': coupon_data
                    })
                except: continue
            return coupons
        except: return []
    return []

def save_new_coupon(name, coupon_data):
    coupons = load_saved_coupons()
    new_id = len(coupons) + 1
    simplified_data = []
    for bet in coupon_data:
        simplified_data.append({
            'Mecz': bet['Mecz'], 'Home': bet['Mecz'].split(' - ')[0], 'Away': bet['Mecz'].split(' - ')[1],
            'Typ': bet['Typ'], 'Date': bet.get('Date', 'N/A'), 'Pewność': bet['Pewność'],
            'Result': '?', 'Forma': bet.get('Forma', ''), 'Stabilność': bet.get('Stabilność', ''),
            'Wynik': bet.get('Wynik', ''), 'Verdict': bet.get('Verdict', '')
        })
    new_entry = {
        'ID': new_id, 'Name': name, 'DateCreated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Data': json.dumps(simplified_data)
    }
    df_new = pd.DataFrame([new_entry])
    if os.path.exists(COUPONS_DB_FILE): df_new.to_csv(COUPONS_DB_FILE, mode='a', header=False, index=False)
    else: df_new.to_csv(COUPONS_DB_FILE, index=False)

def check_results_for_coupons():
    coupons = load_saved_coupons()
    if not coupons: return []
    conn = sqlite3.connect("mintstats.db")
    df_history = pd.read_sql("SELECT * FROM all_leagues", conn)
    conn.close()
    updated_coupons = []
    for coupon in coupons:
        bets = coupon['data']
        processed_bets = []
        for bet in bets:
            status = bet.get('Result', '?')
            if status in ['✅', '❌']: processed_bets.append(bet); continue
            h, a = bet['Home'], bet['Away']
            match = df_history[(df_history['HomeTeam'] == h) & (df_history['AwayTeam'] == a)]
            if not match.empty:
                row = match.iloc[0]
                res = evaluate_bet(bet['Typ'], row)
                bet['Result'] = '✅' if res else '❌'
                bet['Score'] = f"{int(row['FTHG'])}:{int(row['FTAG'])}"
            processed_bets.append(bet)
        coupon['data'] = processed_bets
        updated_coupons.append(coupon)
    df_save = pd.DataFrame([{
        'ID': c['id'], 'Name': c['name'], 'DateCreated': c['date_created'], 'Data': json.dumps(c['data'])
    } for c in updated_coupons])
    df_save.to_csv(COUPONS_DB_FILE, index=False)
    return updated_coupons

def evaluate_bet(bet_type, row):
    fthg, ftag = row['FTHG'], row['FTAG']; goals = fthg + ftag
    try:
        bet_type_clean = bet_type.split('(')[0].strip()
        if bet_type_clean.startswith("Win"):
            if "Win " + row['HomeTeam'] == bet_type_clean: return fthg > ftag
            if "Win " + row['AwayTeam'] == bet_type_clean: return ftag > fthg
        if bet_type_clean == "Over 2.5": return goals > 2.5
        if bet_type_clean == "Over 1.5": return goals > 1.5
        if bet_type_clean == "Over 0.5": return goals > 0.5
        if bet_type_clean == "Under 4.5": return goals <= 4.5
        if bet_type_clean == "Under 3.5": return goals <= 3.5
        if bet_type_clean == "Under 2.5": return goals <= 2.5
        if bet_type_clean == "BTS": return fthg > 0 and ftag > 0
        if bet_type_clean == "BTS NO": return not (fthg > 0 and ftag > 0)
        if bet_type_clean == "1X": return fthg >= ftag
        if bet_type_clean == "X2": return ftag >= fthg
        if bet_type_clean == "12": return fthg != ftag
        if "nie strzeli" in bet_type_clean:
            if row['HomeTeam'] in bet_type_clean: return fthg == 0
            if row['AwayTeam'] in bet_type_clean: return ftag == 0
        elif "strzeli" in bet_type_clean:
            if row['HomeTeam'] in bet_type_clean: return fthg > 0
            if row['AwayTeam'] in bet_type_clean: return ftag > 0
        if "HT Over 1.5" in bet_type_clean:
            if 'HTHG' in row and 'HTAG' in row: return (row['HTHG'] + row['HTAG']) > 1.5
            return False
    except: return False
    return False

def check_team_conflict(home, away, pool):
    for m in pool:
        if m['Home'] == home and m['Away'] == away: return f"⛔ Mecz {home} vs {away} jest już na liście!"
    return None

def clean_expired_matches(pool):
    today_str = datetime.today().strftime('%Y-%m-%d'); new_pool = []; removed = 0
    for m in pool:
        if 'Date' not in m or not m['Date'] or str(m['Date']) == 'nan': new_pool.append(m); continue
        try:
            if str(m['Date']) >= today_str: new_pool.append(m)
            else: removed += 1
        except: new_pool.append(m)
    return new_pool, removed

def process_uploaded_history(files):
    all_data = []
    detected_codes = set()
    unknown_codes = set()
    for uploaded_file in files:
        try:
            bytes_data = uploaded_file.getvalue()
            try: df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8')
            except: 
                try: df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1')
                except: df = pd.read_csv(io.BytesIO(bytes_data), sep=';', encoding='latin1')
            if len(df.columns) < 2: continue
            df.columns = [c.strip() for c in df.columns]
            renames = {'Home': 'HomeTeam', 'Away': 'AwayTeam', 'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'}
            df.rename(columns=renames, inplace=True)
            if 'Div' not in df.columns:
                file_code = uploaded_file.name.replace('.csv', '').upper()
                df['Div'] = file_code
            unique_divs = df['Div'].unique()
            for div in unique_divs:
                detected_codes.add(div)
                if div not in LEAGUE_NAMES: unknown_codes.add(div)
            base_req = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if not all(col in df.columns for col in base_req):
                missing = [c for c in base_req if c not in df.columns]
                st.error(f"❌ Plik '{uploaded_file.name}' odrzucony. Brakuje: {missing}")
                continue
            cols = base_req + ['FTR'] if 'FTR' in df.columns else base_req
            if 'HTHG' in df.columns and 'HTAG' in df.columns: cols.extend(['HTHG', 'HTAG'])
            df_cl = df[cols].copy().dropna(subset=['HomeTeam', 'FTHG'])
            df_cl['Date'] = pd.to_datetime(df_cl['Date'], dayfirst=True, errors='coerce')
            df_cl['LeagueName'] = df_cl['Div'].map(LEAGUE_NAMES).fillna(df_cl['Div'])
            all_data.append(df_cl)
        except Exception as e: st.error(f"Krytyczny błąd pliku {uploaded_file.name}: {e}")
    if all_data:
        master = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect("mintstats.db")
        master.to_sql('all_leagues', conn, if_exists='replace', index=False)
        conn.close()
        return len(master), list(detected_codes), list(unknown_codes)
    return 0, [], []

def clean_ocr_text_debug(text):
    lines = text.split('\n'); cleaned = []
    for line in lines:
        normalized = re.sub(r'[^a-zA-Z0-9 ]', ' ', line).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        if "liga" in normalized.lower() or "serie" in normalized.lower(): continue
        if len(normalized) > 2: cleaned.append(normalized)
    return cleaned

def extract_text_from_image(uploaded_file):
    try: image = Image.open(uploaded_file); return pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    except Exception as e: return f"Error OCR: {e}"

def resolve_team_name(raw_name, available_teams):
    cur = raw_name.lower().strip()
    for alias, db_name in TEAM_ALIASES.items():
        if alias == cur: return db_name
        if len(alias) > 3 and alias in cur: return db_name
    match = difflib.get_close_matches(cur, [t.lower() for t in available_teams], n=1, cutoff=0.7)
    if match:
        for real_name in available_teams:
            if real_name.lower() == match[0]: return real_name
    return None

def parse_raw_text(text_input, available_teams):
    lines = text_input.split('\n'); found_matches = []; today_str = datetime.today().strftime('%Y-%m-%d')
    for line in lines:
        line = line.strip()
        if not line: continue
        line = re.sub(r'\d{2}:\d{2}', '', line)
        parts = []
        if " - " in line: parts = line.split(" - ")
        elif " vs " in line: parts = line.split(" vs ")
        if len(parts) >= 2:
            raw_home = parts[0]; raw_away_chunk = parts[1]; raw_away = re.split(r'[\d\.]+', raw_away_chunk)[0]
            home_team = resolve_team_name(raw_home, available_teams); away_team = resolve_team_name(raw_away, available_teams)
            if home_team and away_team and home_team != away_team:
                found_matches.append({'Home': home_team, 'Away': away_team, 'League': 'Text Import', 'Date': today_str})
    return found_matches

def smart_parse_matches_v3(text_input, available_teams):
    cleaned_lines = clean_ocr_text_debug(text_input); found_teams = []; debug_log = []; today_str = datetime.today().strftime('%Y-%m-%d')
    for line in cleaned_lines:
        cur = line.lower().strip(); matched = resolve_team_name(cur, available_teams)
        if matched:
            if not found_teams or found_teams[-1] != matched: found_teams.append(matched)
            debug_log.append(f"✅ '{cur}' -> '{matched}'")
        else: debug_log.append(f"❌ '{cur}'")
    matches = [{'Home': found_teams[i], 'Away': found_teams[i+1], 'League': 'OCR Import', 'Date': today_str} for i in range(0, len(found_teams) - 1, 2)]
    return matches, debug_log, cleaned_lines

def parse_fixtures_csv(file):
    try:
        df = pd.read_csv(file)
        if not {'Div', 'HomeTeam', 'AwayTeam'}.issubset(df.columns): return [], "Brak kolumn Div/HomeTeam/AwayTeam"
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        else: df['Date'] = datetime.today().strftime('%Y-%m-%d')
        matches = []
        for _, row in df.iterrows(): matches.append({'Home': row['HomeTeam'], 'Away': row['AwayTeam'], 'League': row['Div'], 'Date': row['Date']})
        return matches, None
    except Exception as e: return [], str(e)

# --- AUTOMATYCZNA AKTUALIZACJA ---
def get_current_season_string():
    today = datetime.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}"

def download_and_update_db(league_codes):
    season = get_current_season_string()
    base_url_main = f"https://www.football-data.co.uk/mmz4281/{season}/"
    base_url_extra = "https://www.football-data.co.uk/new/"
    success_count = 0
    total_rows = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_dfs = []
    codes_to_check = list(set([k for k in LEAGUE_NAMES.keys() if len(k) <= 4]))
    
    for i, code in enumerate(codes_to_check):
        status_text.text(f"Sprawdzam: {code}...")
        progress_bar.progress((i + 1) / len(codes_to_check))
        url = f"{base_url_main}{code}.csv"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                url = f"{base_url_extra}{code}.csv"
                response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                try:
                    df = pd.read_csv(io.StringIO(response.text))
                    renames = {'Home': 'HomeTeam', 'Away': 'AwayTeam', 'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'}
                    df.rename(columns=renames, inplace=True)
                    if 'Div' not in df.columns: df['Div'] = code
                    req_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                    if all(c in df.columns for c in req_cols):
                        cols = ['Div'] + req_cols
                        if 'HTHG' in df.columns and 'HTAG' in df.columns: cols.extend(['HTHG', 'HTAG'])
                        df_cl = df[cols].copy().dropna(subset=['HomeTeam', 'FTHG'])
                        df_cl['Date'] = pd.to_datetime(df_cl['Date'], dayfirst=True, errors='coerce')
                        df_cl['LeagueName'] = df_cl['Div'].map(LEAGUE_NAMES).fillna(df_cl['Div'])
                        all_dfs.append(df_cl)
                        success_count += 1
                        total_rows += len(df_cl)
                except: continue
        except: continue
            
    status_text.text("Zapisywanie do bazy...")
    if all_dfs:
        new_data = pd.concat(all_dfs, ignore_index=True)
        conn = sqlite3.connect("mintstats.db")
        try:
            old_data = pd.read_sql("SELECT * FROM all_leagues", conn)
            old_data['Date'] = pd.to_datetime(old_data['Date'])
            current_season_start = pd.to_datetime(f"{get_current_season_string()[:2]}-07-01", format='%y-%m-%d')
            history_keeper = old_data[old_data['Date'] < current_season_start]
            final_db = pd.concat([history_keeper, new_data], ignore_index=True)
            final_db.to_sql('all_leagues', conn, if_exists='replace', index=False)
        except:
            new_data.to_sql('all_leagues', conn, if_exists='replace', index=False)
        conn.close()
        st.cache_data.clear() # CLEAR CACHE
        return success_count, total_rows
    return 0, 0

def get_db_status():
    try:
        conn = sqlite3.connect("mintstats.db")
        query = "SELECT LeagueName as Liga, COUNT(*) as Mecze, MAX(Date) as Ostatni_Mecz FROM all_leagues GROUP BY LeagueName ORDER BY Ostatni_Mecz DESC"
        df = pd.read_sql(query, conn)
        df['Ostatni_Mecz'] = pd.to_datetime(df['Ostatni_Mecz']).dt.strftime('%Y-%m-%d')
        conn.close()
        return df
    except:
        return pd.DataFrame()

# --- MODEL POISSONA ---
class PoissonModel:
    def __init__(self, data):
        self.data = data
        self.team_stats_ft = {}
        self.team_stats_ht = {}
        self.team_form = {} 
        self.team_chaos = {} 
        self.league_avg_ft = 1.0
        self.league_avg_ht = 1.0
        self.home_adv_factor = 1.15
        
        if not data.empty:
            self.data = self.data.sort_values(by='Date')
            self._calculate_strength()
            self._calculate_form_and_chaos()

    def _calculate_strength(self):
        lg_ft = self.data['FTHG'].sum() + self.data['FTAG'].sum()
        matches = len(self.data) * 2
        self.league_avg_ft = lg_ft / matches if matches > 0 else 1.0
        home_goals = self.data['FTHG'].sum(); away_goals = self.data['FTAG'].sum()
        if away_goals > 0: self.home_adv_factor = home_goals / away_goals
        
        has_ht = 'HTHG' in self.data.columns and 'HTAG' in self.data.columns
        if has_ht: 
            lg_ht = self.data['HTHG'].sum() + self.data['HTAG'].sum()
            self.league_avg_ht = lg_ht / matches if matches > 0 and lg_ht > 0 else 1.0
        else:
            self.league_avg_ht = 1.0

        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for team in teams:
            home = self.data[self.data['HomeTeam'] == team]; away = self.data[self.data['AwayTeam'] == team]
            scored_ft = home['FTHG'].sum() + away['FTAG'].sum(); conceded_ft = home['FTAG'].sum() + away['FTHG'].sum()
            played = len(home) + len(away)
            if played > 0:
                la_ft = self.league_avg_ft if self.league_avg_ft > 0 else 1.0
                self.team_stats_ft[team] = {'att': (scored_ft/played)/la_ft, 'def': (conceded_ft/played)/la_ft}
                
                if has_ht:
                    scored_ht = home['HTHG'].sum() + away['HTAG'].sum(); conceded_ht = home['HTAG'].sum() + away['HTHG'].sum()
                    la_ht = self.league_avg_ht if self.league_avg_ht > 0 else 1.0
                    self.team_stats_ht[team] = {'attack': (scored_ht/played)/la_ht, 'defense': (conceded_ht/played)/la_ht}

    def _calculate_form_and_chaos(self):
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for team in teams:
            matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)].tail(5)
            form_icons = []; scored_recent = 0
            for _, row in matches.iterrows():
                is_home = row['HomeTeam'] == team
                gf = row['FTHG'] if is_home else row['FTAG']
                ga = row['FTAG'] if is_home else row['FTHG']
                scored_recent += gf
                if gf > ga: form_icons.append("🟢")
                elif gf == ga: form_icons.append("🤝")
                else: form_icons.append("🔴")
            
            att_boost = 1.0; form_score = 50
            if len(matches) > 0:
                avg_scored = scored_recent / len(matches)
                att_boost = 1.0 + (avg_scored * 0.05)
                form_score = form_icons.count("🟢")*20 + form_icons.count("🤝")*10
            
            self.team_form[team] = {'icons': "".join(reversed(form_icons)), 'att_boost': att_boost, 'score': form_score}

            all_matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)]
            goals_sequence = []
            for _, row in all_matches.iterrows():
                gf = row['FTHG'] if row['HomeTeam'] == team else row['FTAG']
                goals_sequence.append(gf)
            
            chaos_score = 50
            if len(goals_sequence) > 3:
                std_dev = np.std(goals_sequence)
                chaos_rating = "Stabilny 🧊" if std_dev < 0.9 else ("Chaos 🌪️" if std_dev > 1.4 else "Norma")
                chaos_penalty = 0.9 if std_dev > 1.4 else (1.05 if std_dev < 0.9 else 1.0)
                chaos_score = max(0, min(100, 100 - (std_dev * 30)))
                self.team_chaos[team] = {'rating': chaos_rating, 'factor': chaos_penalty, 'score': chaos_score}
            else: self.team_chaos[team] = {'rating': "-", 'factor': 1.0, 'score': 50}

    def get_h2h_analysis(self, home, away):
        mask = ((self.data['HomeTeam'] == home) & (self.data['AwayTeam'] == away)) | ((self.data['HomeTeam'] == away) & (self.data['AwayTeam'] == home))
        h2h_matches = self.data[mask].sort_values(by='Date', ascending=False).head(5)
        if h2h_matches.empty: return None
        home_wins = 0
        for _, row in h2h_matches.iterrows():
            if (row['HomeTeam'] == home and row['FTHG'] > row['FTAG']) or (row['AwayTeam'] == home and row['FTAG'] > row['FTHG']): home_wins += 1
        if len(h2h_matches) >= 3 and home_wins == 0: return "⚠️ H2H (Kryptonit!)"
        return None

    def simulate_match_monte_carlo(self, xg_h, xg_a, n=1000):
        home_wins, draws, away_wins, over_2_5, over_1_5, bts_yes = 0,0,0,0,0,0
        for _ in range(n):
            adj_h = xg_h * np.random.uniform(0.8, 1.2); adj_a = xg_a * np.random.uniform(0.8, 1.2)
            sim_h = np.random.poisson(adj_h); sim_a = np.random.poisson(adj_a)
            if sim_h > sim_a: home_wins += 1
            elif sim_h == sim_a: draws += 1
            else: away_wins += 1
            if (sim_h + sim_a) > 2.5: over_2_5 += 1
            if (sim_h + sim_a) > 1.5: over_1_5 += 1
            if sim_h > 0 and sim_a > 0: bts_yes += 1
        return {'1': (home_wins/n)*100, 'X': (draws/n)*100, '2': (away_wins/n)*100, 'Over 2.5': (over_2_5/n)*100, 'Over 1.5': (over_1_5/n)*100, 'BTS': (bts_yes/n)*100}

    def predict(self, home, away):
        if home not in self.team_stats_ft or away not in self.team_stats_ft: return None, None, None, None
        h_att = self.team_stats_ft[home]['att']; h_def = self.team_stats_ft[home]['def']
        a_att = self.team_stats_ft[away]['att']; a_def = self.team_stats_ft[away]['def']
        if home in self.team_form: h_att *= self.team_form[home]['att_boost']
        if away in self.team_form: a_att *= self.team_form[away]['att_boost']
        xg_h_ft = h_att * a_def * self.league_avg_ft * self.home_adv_factor
        xg_a_ft = a_att * h_def * self.league_avg_ft
        xg_h_ht, xg_a_ht = 0.0, 0.0
        if home in self.team_stats_ht and away in self.team_stats_ht:
            xg_h_ht = self.team_stats_ht[home]['attack'] * self.team_stats_ht[away]['defense'] * self.league_avg_ht * 1.05
            xg_a_ht = self.team_stats_ht[away]['attack'] * self.team_stats_ht[home]['defense'] * self.league_avg_ht
        return xg_h_ft, xg_a_ft, xg_h_ht, xg_a_ht

    def calculate_probs(self, xg_h_ft, xg_a_ft, xg_h_ht, xg_a_ht):
        max_goals = 8
        mat_ft = np.array([[poisson.pmf(i, xg_h_ft) * poisson.pmf(j, xg_a_ft) for j in range(max_goals)] for i in range(max_goals)])
        mat_ht = np.array([[poisson.pmf(i, xg_h_ht) * poisson.pmf(j, xg_a_ht) for j in range(max_goals)] for i in range(max_goals)])
        prob_1 = np.sum(np.tril(mat_ft, -1)); prob_x = np.sum(np.diag(mat_ft)); prob_2 = np.sum(np.triu(mat_ft, 1))
        
        prob_home_0 = poisson.pmf(0, xg_h_ft)
        prob_away_0 = poisson.pmf(0, xg_a_ft)
        prob_0_0 = prob_home_0 * prob_away_0
        
        max_prob_index = np.unravel_index(mat_ft.argmax(), mat_ft.shape)
        most_likely_score = f"{max_prob_index[0]}:{max_prob_index[1]}"

        return {
            "1": prob_1, "X": prob_x, "2": prob_2, "1X": prob_1+prob_x, "X2": prob_x+prob_2, "12": prob_1+prob_2,
            "BTS_Yes": np.sum(mat_ft[1:, 1:]), "BTS_No": 1.0-np.sum(mat_ft[1:, 1:]),
            "Over_1.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Over_2.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 2.5]),
            "Over_0.5_FT": 1.0 - prob_0_0,
            "Under_2.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 2.5]),
            "Under_3.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 3.5]),
            "Under_4.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 4.5]),
            "Over_1.5_HT": np.sum([mat_ht[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Home_Yes": 1.0 - prob_home_0, "Away_Yes": 1.0 - prob_away_0,
            "Exact_Score": most_likely_score
        }
    
    def get_team_info(self, team):
        form = self.team_form.get(team, {'icons': '⚪', 'att_boost': 1.0, 'score': 50})
        chaos = self.team_chaos.get(team, {'rating': '-', 'factor': 1.0, 'score': 50})
        stats = self.team_stats_ft.get(team, {'att':1.0, 'def':1.0})
        combined = {**stats, 'form_score': form['score'], 'chaos_score': chaos['score']}
        return form['icons'], chaos, combined

    def generate_narrative(self, xg_h, xg_a, chaos_h, chaos_a, home_adv):
        texts = []
        if xg_h > xg_a * 1.6: texts.append("🔥 Gospodarz jest wyraźnym faworytem (Twierdza).")
        elif xg_a > xg_h * 1.4: texts.append("🔥 Goście dominują analitycznie.")
        else: texts.append("⚖️ Mecz wyrównany (50/50).")
        total_xg = xg_h + xg_a
        if total_xg > 3.0: texts.append("🍿 Spodziewany grad goli (Wysokie xG).")
        elif total_xg < 2.0: texts.append("🧱 Zapowiada się defensywne szachy (Niskie xG).")
        if chaos_h['factor'] < 0.95 or chaos_a['factor'] < 0.95: texts.append("🌪️ Ostrzeżenie: Przynajmniej jedna drużyna jest nieprzewidywalna (Chaos).")
        return " ".join(texts)

# --- WYKRESY (TERAZ TUTAJ!) ---
def create_radar_chart(h_stats, a_stats, h_name, a_name):
    def norm_att(val): return min(val * 50, 100)
    def norm_def(val): return min((2.0 - val) * 50, 100)
    categories = ['Atak', 'Obrona (Szczelność)', 'Forma', 'Stabilność']
    h_vals = [norm_att(h_stats['att']), norm_def(h_stats['def']), h_stats.get('form_score', 50), h_stats.get('chaos_score', 50)]
    a_vals = [norm_att(a_stats['att']), norm_def(a_stats['def']), a_stats.get('form_score', 50), a_stats.get('chaos_score', 50)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories, fill='toself', name=h_name, line_color='#00C896'))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a_name, line_color='#FF4B4B'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=True), angularaxis=dict(tickfont=dict(color='#333', size=12))),
        showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#333'), margin=dict(l=30, r=30, t=30, b=30), height=250, legend=dict(font=dict(color='#333'))
    )
    return fig

def create_score_heatmap(xg_h, xg_a):
    max_g = 6
    matrix = [[poisson.pmf(h, xg_h) * poisson.pmf(a, xg_a) for a in range(max_g)] for h in range(max_g)]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=[str(i) for i in range(max_g)], y=[str(i) for i in range(max_g)],
        colorscale='Mint', texttemplate="%{z:.1%}"
    ))
    fig.update_layout(
        title="Prawdopodobieństwo Wyniku", xaxis_title="Gole Gości", yaxis_title="Gole Gospodarzy",
        height=300, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_goal_distribution(xg_h, xg_a, h_name, a_name):
    max_g = 5
    h_probs = [poisson.pmf(i, xg_h)*100 for i in range(max_g)]
    a_probs = [poisson.pmf(i, xg_a)*100 for i in range(max_g)]
    x_labels = [str(i) for i in range(max_g-1)] + ["4+"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=h_name, x=x_labels, y=h_probs, marker_color='#00C896'))
    fig.add_trace(go.Bar(name=a_name, x=x_labels, y=a_probs, marker_color='#FF4B4B'))
    fig.update_layout(
        title="Rozkład Goli", barmode='group', xaxis_title="Liczba Goli", yaxis_title="Szansa (%)",
        height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_league_scatter(df_league):
    model = PoissonModel(df_league)
    teams = []; att = []; defn = []
    for team, stats in model.team_stats_ft.items():
        teams.append(team); att.append(stats['att']); defn.append(stats['def'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=att, y=defn, mode='markers+text', text=teams, textposition='top center',
        marker=dict(size=12, color='#00C896', line=dict(width=2, color='DarkSlateGrey'))
    ))
    fig.add_vline(x=1.0, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="grey")
    fig.add_annotation(x=1.5, y=0.5, text="👑 DOMINATORZY", showarrow=False, font=dict(size=14, color="green"))
    fig.add_annotation(x=0.5, y=1.5, text="💀 DO BICIA", showarrow=False, font=dict(size=14, color="red"))
    fig.add_annotation(x=1.5, y=1.5, text="🍿 WESOŁY FUTBOL", showarrow=False, font=dict(size=12, color="orange"))
    fig.add_annotation(x=0.5, y=0.5, text="🧱 MURARZE", showarrow=False, font=dict(size=12, color="blue"))
    fig.update_layout(
        title="Mapa Siły Ligowej", xaxis_title="Siła Ataku (>1.0 Dobrze)", yaxis_title="Dziurawość Obrony (>1.0 Źle)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#333'),
        yaxis=dict(autorange="reversed")
    )
    return fig

def get_global_stats(df_all):
    stats = []
    leagues = df_all['LeagueName'].unique()
    for lg in leagues:
        d = df_all[df_all['LeagueName'] == lg]
        total = len(d)
        if total < 10: continue
        goals = d['FTHG'].sum() + d['FTAG'].sum()
        avg_goals = goals / total
        home_wins = len(d[d['FTHG'] > d['FTAG']])
        draws = len(d[d['FTHG'] == d['FTAG']])
        away_wins = len(d[d['FTHG'] < d['FTAG']])
        bts = len(d[(d['FTHG'] > 0) & (d['FTAG'] > 0)])
        over25 = len(d[(d['FTHG'] + d['FTAG']) > 2.5])
        stats.append({
            'Liga': lg, 'Mecze': total, 'Śr. Goli': round(avg_goals, 2),
            '1 (%)': round((home_wins/total)*100, 1),
            'X (%)': round((draws/total)*100, 1),
            '2 (%)': round((away_wins/total)*100, 1),
            'BTS (%)': round((bts/total)*100, 1),
            'Over 2.5 (%)': round((over25/total)*100, 1)
        })
    return pd.DataFrame(stats)

# --- BUFOROWANIE DANYCH (CACHE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_filter_data(cutoff_date):
    """Wczytuje dane z SQL i filtruje po dacie. Wynik jest buforowany."""
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT * FROM all_leagues", conn)
        conn.close()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Filtrowanie globalne Chronos
        df = df[df['Date'] >= cutoff_date]
        return df
    except:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def get_cached_model(data_hash_key, _df):
    """Tworzy model Poissona tylko gdy dane się zmienią."""
    if _df.empty: return None
    return PoissonModel(_df)

# --- INIT ---
# Global variables moved to top for scope safety
if 'fixture_pool' not in st.session_state: st.session_state.fixture_pool = load_fixture_pool()
if 'generated_coupons' not in st.session_state: st.session_state.generated_coupons = [] 
if 'last_ocr_debug' not in st.session_state: st.session_state.last_ocr_debug = None

# --- INTERFEJS ---
st.title("☁️ MintStats v26.2: Missing Link")

st.sidebar.header("Panel Sterowania")
mode = st.sidebar.radio("Wybierz moduł:", ["1. 🛠️ ADMIN (Baza Danych)", "2. 🚀 GENERATOR KUPONÓW", "3. 📜 MOJE KUPONY", "4. 🧪 LABORATORIUM"])

# --- SUWAK HORYZONTU CZASOWEGO (GLOBALNY) ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Ustawienia Modelu")
years_back = st.sidebar.slider("Horyzont Czasowy (Lata)", 1, 10, 2, help="Ile lat wstecz analizować? Mniej = świeża forma.")
# Obliczamy datę odcięcia RAZ dla całej aplikacji
cutoff_date = pd.to_datetime('today') - pd.DateOffset(years=years_back)

# --- SEKCJA BACKUP DLA PODRÓŻNIKÓW ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Kopia Zapasowa (Praca <-> Dom)")

if st.sidebar.button("📦 Przygotuj Paczkę (Export)"):
    save_fixture_pool(st.session_state.fixture_pool)
    try:
        with open(FIXTURES_DB_FILE, "rb") as f:
            st.sidebar.download_button("⬇️ Pobierz Terminarz", f, file_name="terminarz_backup.csv", mime="text/csv")
    except: st.sidebar.warning("Brak terminarza.")
    try:
        with open(COUPONS_DB_FILE, "rb") as f:
            st.sidebar.download_button("⬇️ Pobierz Kupony", f, file_name="kupony_backup.csv", mime="text/csv")
    except: st.sidebar.warning("Brak kuponów.")

st.sidebar.markdown("---")
uploaded_backup_fix = st.sidebar.file_uploader("Wgraj Terminarz (CSV)", type=['csv'])
if uploaded_backup_fix:
    if st.sidebar.button("♻️ Przywróć Terminarz"):
        try:
            df = pd.read_csv(uploaded_backup_fix)
            if 'Date' not in df.columns: df['Date'] = datetime.today().strftime('%Y-%m-%d')
            st.session_state.fixture_pool = df.to_dict('records')
            save_fixture_pool(st.session_state.fixture_pool)
            st.sidebar.success("Terminarz przywrócony!")
            st.rerun()
        except Exception as e: st.sidebar.error(f"Błąd: {e}")

uploaded_backup_coup = st.sidebar.file_uploader("Wgraj Kupony (CSV)", type=['csv'])
if uploaded_backup_coup:
    if st.sidebar.button("♻️ Przywróć Kupony"):
        try:
            with open(COUPONS_DB_FILE, "wb") as f: f.write(uploaded_backup_coup.getbuffer())
            st.sidebar.success("Kupony przywrócone!")
            st.rerun()
        except Exception as e: st.sidebar.error(f"Błąd: {e}")

if mode == "1. 🛠️ ADMIN (Baza Danych)":
    st.subheader("🛠️ Zarządzanie Bazą Danych")
    
    st.markdown("### 🌐 Aktualizacja Online (Football-Data)")
    if st.button("🔄 Sprawdź i Pobierz Aktualizacje"):
        with st.spinner("Łączenie z serwerem Football-Data..."):
            s, t = download_and_update_db(LEAGUE_NAMES)
            if s > 0: st.success(f"✅ Pomyślnie zaktualizowano {s} lig ({t} meczów).")
            else: st.warning("Brak nowych danych lub błąd połączenia.")
            
    st.divider()
    st.markdown("### 📂 Wgrywanie Ręczne (Pliki CSV)")
    
    uploaded_history = st.file_uploader("Wgraj pliki ligowe (Historia)", type=['csv'], accept_multiple_files=True)
    if uploaded_history and st.button("Aktualizuj Bazę Danych"):
        with st.spinner("Przetwarzanie..."):
            count, leagues_found, unknown_codes = process_uploaded_history(uploaded_history)
            if count > 0: 
                # CLEAR CACHE
                st.cache_data.clear()
                st.success(f"✅ Baza zaktualizowana ({count} meczów).")
                st.info(f"🆕 Wykryte kody lig: {', '.join(map(str, leagues_found))}")
                
                if unknown_codes:
                    st.warning(f"⚠️ Kody nierozpoznane (nieznane dla MintStats): {', '.join(map(str, unknown_codes))}")
                    st.caption("ℹ️ Te ligi zostały wgrane, ale będą widoczne pod surowym kodem (np. 'X1') zamiast pełnej nazwy.")
            else: st.error("Błąd importu. Sprawdź komunikaty powyżej.")
    
    # --- AUDYTOR BAZY DANYCH (NOWY PANEL) ---
    st.divider()
    st.markdown("### 🔍 Status Bazy Danych (Audytor)")
    db_status = get_db_status()
    if not db_status.empty:
        st.caption("Poniższa tabela pokazuje, jakie dane masz w systemie i kiedy był ostatni mecz.")
        st.dataframe(db_status, use_container_width=True)
    else:
        st.info("Baza danych jest pusta.")

elif mode == "2. 🚀 GENERATOR KUPONÓW":
    leagues = get_leagues_list()
    if not leagues: st.error("⛔ Baza pusta!"); st.stop()
    
    # --- TURBO CACHING ---
    df_all = load_and_filter_data(cutoff_date)
    
    if df_all.empty:
        st.warning("Brak danych po filtrowaniu (zmień horyzont czasowy lub wgraj nowsze pliki).")
        st.stop()

    # Using cache_resource because model is an object
    model = get_cached_model(str(df_all.shape), df_all)
    if not model: st.stop()
    
    gen = CouponGenerator(model)
    all_teams_list = pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).unique()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dodaj Mecze")
    tab_manual, tab_ocr, tab_text, tab_csv = st.sidebar.tabs(["Ręczny", "📸 Zdjęcie", "📝 Wklej Tekst", "📁 CSV"])
    
    new_items = []
    with tab_manual:
        sel_league = st.selectbox("Liga:", leagues)
        # Filter league data as well using the cached DF
        df_l = df_all[df_all['LeagueName'] == sel_league]
        
        teams = sorted(pd.concat([df_l['HomeTeam'], df_l['AwayTeam']]).unique())
        with st.form("manual_add"):
            col_date, col_h, col_a = st.columns([1,2,2])
            with col_date: date_input = st.date_input("Data", datetime.today())
            with col_h: h = st.selectbox("Dom", teams)
            with col_a: a = st.selectbox("Wyjazd", teams)
            if st.form_submit_button("➕ Dodaj") and h!=a: new_items.append({'Home':h, 'Away':a, 'League':sel_league, 'Date': str(date_input)})
    
    with tab_ocr:
        uploaded_img = st.file_uploader("Screen Flashscore", type=['png', 'jpg', 'jpeg'])
        if uploaded_img and st.button("Skanuj"):
            with st.spinner("OCR Analiza..."):
                txt = extract_text_from_image(uploaded_img)
                m_list, debug_logs, raw_lines = smart_parse_matches_v3(txt, all_teams_list)
                st.session_state.last_ocr_debug = {'raw': raw_lines, 'logs': debug_logs}
                if m_list: new_items.extend(m_list); st.success(f"Wykryto {len(m_list)} meczów")
                else: st.warning("Brak dopasowań.")
        if st.session_state.last_ocr_debug:
            with st.expander("🕵️ DEBUG OCR"):
                st.text("\n".join(st.session_state.last_ocr_debug['raw']))
                for log in st.session_state.last_ocr_debug['logs']:
                    if "✅" in log: st.success(log)
                    elif "🔹" in log: st.info(log)
                    else: st.error(log)
            if st.button("Wyczyść Debug"): st.session_state.last_ocr_debug = None; st.rerun()

    with tab_text:
        st.info("💡 Skopiuj listę meczów z Flashscore (Ctrl+C) i wklej tutaj.")
        raw_text_input = st.text_area("Wklej mecze", height=150)
        if st.button("🔍 Analizuj Tekst"):
            parsed = parse_raw_text(raw_text_input, all_teams_list)
            if parsed: new_items.extend(parsed); st.success(f"✅ Znaleziono {len(parsed)} par!")
            else: st.error("Brak par.")

    with tab_csv:
        uploaded_fix = st.file_uploader("fixtures.csv", type=['csv'])
        if uploaded_fix and st.button("📥 Import"):
            m_list, err = parse_fixtures_csv(uploaded_fix)
            if not err: new_items.extend(m_list); st.success(f"Import {len(m_list)}")
            else: st.error(err)

    if new_items:
        added_count = 0; errors = []
        for item in new_items:
            conflict_msg = check_team_conflict(item['Home'], item['Away'], st.session_state.fixture_pool)
            if conflict_msg: errors.append(conflict_msg)
            else: st.session_state.fixture_pool.append(item); added_count += 1
        save_fixture_pool(st.session_state.fixture_pool)
        if added_count > 0: st.toast(f"Dodano {added_count} meczów!")
        if errors:
            with st.expander("⚠️ Pominięto (Duplikaty)", expanded=True):
                for e in errors: st.warning(e)
        st.rerun()

    with st.expander("📊 Mapa Siły Ligowej (Scatter Plot)", expanded=False):
        sel_scatter_league = st.selectbox("Wybierz Ligę do Analizy:", leagues, key="scatter_league")
        df_scatter = df_all[df_all['LeagueName'] == sel_scatter_league]
        
        if not df_scatter.empty:
            fig_scatter = create_league_scatter(df_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else: st.warning("Brak danych dla tej ligi w wybranym okresie.")

    st.subheader("📋 Terminarz")
    col_clean, col_clear = st.columns(2)
    with col_clean:
        if st.button("🧹 Usuń przeterminowane mecze"):
            st.session_state.fixture_pool, removed = clean_expired_matches(st.session_state.fixture_pool)
            save_fixture_pool(st.session_state.fixture_pool); st.success(f"Usunięto {removed}."); st.rerun()
    with col_clear:
        if st.button("🗑️ Wyczyść WSZYSTKO"): 
            st.session_state.fixture_pool = []; save_fixture_pool([]); st.rerun()

    if st.session_state.fixture_pool:
        df_pool = pd.DataFrame(st.session_state.fixture_pool)
        if 'Date' not in df_pool.columns: df_pool['Date'] = datetime.today().strftime('%Y-%m-%d')
        
        edited_df = st.data_editor(df_pool, num_rows="dynamic", use_container_width=True, key="fixture_editor")
        formatted_records = []
        for r in edited_df.to_dict('records'):
            if isinstance(r['Date'], (datetime, date)): r['Date'] = r['Date'].strftime('%Y-%m-%d')
            formatted_records.append(r)

        if formatted_records != st.session_state.fixture_pool:
            st.session_state.fixture_pool = formatted_records
            save_fixture_pool(st.session_state.fixture_pool)

        st.divider()
        st.header("🎲 Generator Kuponów")
        c1, c2, c3 = st.columns(3)
        with c1: gen_mode = st.radio("Tryb:", ["Jeden Pewny Kupon", "System Rozpisowy"])
        with c2: strat = st.selectbox("Strategia", [
            "Mix Bezpieczny (1X, X2, U4.5, O0.5, Gole)", 
            "Podwójna Szansa (1X, X2, 12)",
            "Gole Agresywne (BTS, O2.5)",
            "Do Przerwy (HT O1.5)",
            "Twierdza (Home Win)",
            "Mur Obronny (Under 2.5/3.5)",
            "Złoty Środek (Over 1.5)",
            "Wszystkie Zdarzenia (Max Pewność)",
            "Obie strzelą (TAK)",
            "Obie strzelą (NIE)",
            "1 drużyna strzeli (TAK)",
            "1 drużyna strzeli (NIE)",
            "2 drużyna strzeli (TAK)",
            "2 drużyna strzeli (NIE)"
        ])
        with c3:
            if gen_mode == "Jeden Pewny Kupon": coupon_len = st.number_input("Długość", 1, 50, 12)
            else:
                num_coupons = st.number_input("Ile kuponów?", 1, 10, 3)
                events_per_coupon = st.number_input("Mecze na kupon?", 1, 20, 5)
                chaos_factor = st.slider("Pula (Top X)", 10, 100, 30)
        
        # --- DATE FILTER ---
        available_dates = sorted(list(set([m['Date'] for m in st.session_state.fixture_pool])))
        selected_dates = st.multiselect("📅 Wybierz daty meczów do analizy:", available_dates, default=available_dates)

        if st.button("🚀 GENERUJ", type="primary"):
            # Filter pool by dates
            filtered_pool = [m for m in st.session_state.fixture_pool if m['Date'] in selected_dates]
            
            if not filtered_pool:
                st.warning("Brak meczów w wybranych datach.")
            else:
                analyzed_pool = gen.analyze_pool(filtered_pool, strat)
                if "Mix Bezpieczny" in strat:
                    cat_dc = sorted([x for x in analyzed_pool if x['Kategoria'] == 'DC'], key=lambda x: x['Pewność'], reverse=True)
                    cat_uo = sorted([x for x in analyzed_pool if x['Kategoria'] == 'U/O'], key=lambda x: x['Pewność'], reverse=True)
                    cat_team = sorted([x for x in analyzed_pool if x['Kategoria'] == 'TEAM'], key=lambda x: x['Pewność'], reverse=True)
                    mixed_list = []
                    max_len = max(len(cat_dc), len(cat_uo), len(cat_team))
                    for i in range(max_len):
                        if i < len(cat_dc): mixed_list.append(cat_dc[i])
                        if i < len(cat_uo): mixed_list.append(cat_uo[i])
                        if i < len(cat_team): mixed_list.append(cat_team[i])
                    final_pool = mixed_list
                else: final_pool = sorted(analyzed_pool, key=lambda x: x['Pewność'], reverse=True)

                st.session_state.generated_coupons = [] 
                if gen_mode == "Jeden Pewny Kupon":
                    st.session_state.generated_coupons.append({"name": f"Top {strat}", "data": final_pool[:coupon_len]})
                else: 
                    candidate_pool = final_pool[:chaos_factor]
                    if len(candidate_pool) < events_per_coupon: st.error("Za mało meczów w puli!")
                    else:
                        for i in range(num_coupons):
                            random_selection = random.sample(candidate_pool, min(len(candidate_pool), events_per_coupon))
                            st.session_state.generated_coupons.append({"name": f"Kupon Losowy #{i+1}", "data": random_selection})

        if st.session_state.generated_coupons:
            st.write("---")
            if st.button("💾 Zapisz Wygenerowane Kupony"):
                for kupon in st.session_state.generated_coupons: save_new_coupon(kupon['name'], kupon['data'])
                st.success("Zapisano w Historii!")
            
            for kupon in st.session_state.generated_coupons:
                with st.container():
                    st.subheader(f"🎫 {kupon['name']}")
                    df_k = pd.DataFrame(kupon['data'])
                    if not df_k.empty:
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Średnia Pewność", f"{df_k['Pewność'].mean()*100:.1f}%")
                        k2.metric("Liczba Zdarzeń", len(df_k))
                        best_bet = df_k.iloc[0]
                        k3.metric("Gwiazda Kuponu", f"{best_bet['Mecz']}", delta=best_bet['Typ'])
                        disp_cols = ['Date', 'Mecz', 'Forma', 'Stabilność', 'Liga', 'Typ', 'Pewność', 'xG']
                        st.dataframe(df_k[disp_cols].style.background_gradient(subset=['Pewność'], cmap="RdYlGn", vmin=0.4, vmax=0.9).format({'Pewność':'{:.1%}'}), use_container_width=True)
                        with st.expander("🔍 WAR ROOM (Szczegóły Meczowe)"):
                            for idx, row in df_k.iterrows():
                                st.markdown(f"### {row['Mecz']}")
                                st.info(f"💡 AI Verdict: {row.get('Verdict', 'Brak danych')}")
                                c1, c2, c3 = st.columns([1, 1, 1])
                                h_stats = row.get('HomeStats')
                                a_stats = row.get('AwayStats')
                                xg = row.get('xG', "0:0").split(':')
                                xg_h, xg_a = float(xg[0]), float(xg[1])
                                if h_stats and a_stats:
                                    with c1:
                                        st.caption("Radar Siły")
                                        fig_rad = create_radar_chart(h_stats, a_stats, row['Mecz'].split(' - ')[0], row['Mecz'].split(' - ')[1])
                                        st.plotly_chart(fig_rad, use_container_width=True, key=f"rad_{idx}")
                                    with c2:
                                        st.caption("Heatmapa Wyników")
                                        fig_heat = create_score_heatmap(xg_h, xg_a)
                                        st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{idx}")
                                    with c3:
                                        st.caption("Rozkład Goli")
                                        fig_bar = create_goal_distribution(xg_h, xg_a, row['Mecz'].split(' - ')[0], row['Mecz'].split(' - ')[1])
                                        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}")
                                st.divider()
                    else: st.warning("Brak typów.")
                    st.write("---")
    else: st.info("Pula pusta.")

elif mode == "3. 📜 MOJE KUPONY":
    st.title("📜 Historia Kuponów")
    st.info("ℹ️ Aby system rozliczył kupony, wejdź w ADMIN i wgraj aktualne pliki CSV.")
    if st.button("🔄 Sprawdź Wyniki (Rozlicz Kupony)"):
        with st.spinner("Sędzia sprawdza wyniki..."):
            updated = check_results_for_coupons()
            if updated: st.success("Zaktualizowano statusy!")
            else: st.warning("Brak kuponów do sprawdzenia lub brak danych w bazie.")
    coupons = load_saved_coupons()
    if coupons:
        for c in reversed(coupons):
            with st.expander(f"🎫 {c['name']} (Utworzono: {c['date_created']})", expanded=False):
                df_c = pd.DataFrame(c['data'])
                def highlight_result(val):
                    color = 'transparent'
                    if val == '✅': color = '#C8E6C9' # Light Green
                    elif val == '❌': color = '#FFCDD2' # Light Red
                    return f'background-color: {color}'
                st.dataframe(df_c.style.applymap(highlight_result, subset=['Result']), use_container_width=True)
                wins = len(df_c[df_c['Result'] == '✅']); losses = len(df_c[df_c['Result'] == '❌']); pending = len(df_c[df_c['Result'] == '?'])
                st.caption(f"✅ Trafione: {wins} | ❌ Pudła: {losses} | ⏳ Oczekujące: {pending}")
        if st.button("🗑️ Wyczyść Historię"):
            if os.path.exists(COUPONS_DB_FILE): os.remove(COUPONS_DB_FILE); st.rerun()
    else: st.info("Brak kuponów.")

elif mode == "4. 🧪 LABORATORIUM":
    st.title("🧪 Laboratorium Analityczne")
    tab1, tab2, tab3 = st.tabs(["🔙 Test Wsteczny (Backtest)", "⚖️ Tabela Sprawiedliwości (xPts)", "🌍 Ranking Lig"])
    
    with tab1:
        st.subheader("Sprawdź skuteczność strategii na historii")
        leagues = get_leagues_list()
        sel_lg = st.selectbox("Liga:", leagues, key="bt_lg")
        strat = st.selectbox("Strategia:", ["Mix Bezpieczny (1X, X2, U4.5, O0.5, Gole)", "Podwójna Szansa (1X, X2, 12)", "Gole Agresywne (BTS, O2.5)", "Twierdza (Home Win)", "Mur Obronny (Under 2.5)", "Złoty Środek (Over 1.5)"], key="bt_strat")
        limit = st.slider("Ile ostatnich meczów?", 20, 200, 50)
        
        if st.button("🔥 Uruchom Test"):
            df = get_data_for_league(sel_lg)
            # Apply Chronos filter to Backtest too
            df = df[df['Date'] >= cutoff_date]
            
            if df.empty: st.error("Brak danych!")
            else:
                with st.spinner("Symulowanie przeszłości..."):
                    res = run_backtest(df, strat, limit)
                    col_res1, col_res2 = st.columns(2)
                    rate = (res['Correct']/res['Total'])*100 if res['Total']>0 else 0
                    col_res1.metric("Skuteczność", f"{rate:.1f}%", f"{res['Correct']}/{res['Total']}")
                    fig_bt = go.Figure(data=[go.Bar(name='Trafione', x=['Wyniki'], y=[res['Correct']], marker_color='#00C896'),
                                             go.Bar(name='Pudła', x=['Wyniki'], y=[res['Wrong']], marker_color='#FF4B4B')])
                    st.plotly_chart(fig_bt, use_container_width=True)

    with tab2:
        st.subheader("Tabela Oczekiwanych Punktów (xPoints)")
        leagues = get_leagues_list()
        sel_xp_lg = st.selectbox("Liga:", leagues, key="xp_lg")
        if sel_xp_lg:
            df = get_data_for_league(sel_xp_lg)
            df = df[df['Date'] >= cutoff_date] # Apply Chronos filter
            if not df.empty:
                x_table = calculate_xpts_table(df)
                st.info("💡 Diff > 0: Drużyna ma więcej punktów niż powinna (Szczęście).\nDiff < 0: Drużyna ma mniej punktów niż powinna (Pech - warto grać na nich).")
                def highlight_diff(val):
                    color = 'transparent'
                    if val < -3: color = '#C8E6C9' 
                    elif val > 3: color = '#FFCDD2' 
                    return f'background-color: {color}'
                st.dataframe(x_table.style.applymap(highlight_diff, subset=['Diff']).format({'xPts': '{:.1f}', 'Diff': '{:.1f}'}), use_container_width=True)

    with tab3:
        st.subheader("🌍 Ranking Statystyczny Lig")
        df_all_glob = load_and_filter_data(cutoff_date)
        
        if not df_all_glob.empty:
            df_glob = get_global_stats(df_all_glob)
            sel_metric = st.selectbox("Sortuj według:", ['Śr. Goli', '1 (%)', 'BTS (%)', 'Over 2.5 (%)'])
            df_glob = df_glob.sort_values(by=sel_metric, ascending=False)
            
            fig_glob = px.bar(df_glob, x='Liga', y=sel_metric, color=sel_metric, color_continuous_scale='Mint')
            st.plotly_chart(fig_glob, use_container_width=True)
            st.dataframe(df_glob, use_container_width=True)
        else: st.warning("Wgraj dane do bazy!")
