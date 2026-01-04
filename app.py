import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import difflib
import random
from scipy.stats import poisson
from PIL import Image
import pytesseract
import re
import io
import os
import json
from datetime import datetime, date

# --- KONFIGURACJA ---
st.set_page_config(page_title="MintStats v18.0 Analytical Beast", layout="wide")
FIXTURES_DB_FILE = "my_fixtures.csv"
COUPONS_DB_FILE = "my_coupons.csv"

# --- SŁOWNIK ALIASÓW ---
TEAM_ALIASES = {
    # --- PORTUGALIA ---
    "avs": "AFS", "avs futebol": "AFS", "afs": "AFS", "a v s": "AFS", "b ars": "AFS",
    "brag": "Sp Braga", "braga": "Sp Braga", "sc braga": "Sp Braga", "sp braga": "Sp Braga", "w braga": "Sp Braga",
    "bars": "Boavista", "boavista": "Boavista", "w tondela": "Tondela",
    "sporting": "Sp Lisbon", "sporting cp": "Sp Lisbon", "sp lisbon": "Sp Lisbon",
    "vitoria guimaraes": "Guimaraes", "v guimaraes": "Guimaraes", "guimaraes": "Guimaraes",
    "fc porto": "Porto", "porto": "Porto", "fc porte": "Porto",
    "rio ave": "Rio Ave", "b biosve": "Rio Ave", "8 biosve": "Rio Ave",
    "estoril": "Estoril", "casa pia": "Casa Pia", "gil vicente": "Gil Vicente",
    "farense": "Farense", "famalicao": "Famalicao", "arouca": "Arouca", "moreirense": "Moreirense",
    "estrela": "Estrela", "benfica": "Benfica", "santa clara": "Santa Clara", "nacional": "Nacional",
    
    # --- ANGLIA ---
    "manchester uta": "Man United", "man uta": "Man United",
    "man utd": "Man United", "manchester utd": "Man United", "man united": "Man United",
    "hull": "Hull", "hull city": "Hull", "uit": "Hull", "tui": "Hull",
    "watford": "Watford", "watford fc": "Watford", "wottora": "Watford", "wottore": "Watford",
    "qpr": "QPR", "queens park rangers": "QPR", "opr": "QPR",
    "west brom": "West Brom", "west bromwich": "West Brom",
    "blackburn": "Blackburn", "blackburn rovers": "Blackburn", "q blockouen": "Blackburn",
    "preston": "Preston", "preston north end": "Preston",
    "coventry": "Coventry", "coventry city": "Coventry", 
    "stoke": "Stoke", "stoke city": "Stoke",
    "swansea": "Swansea", "swansea city": "Swansea", 
    "cardiff": "Cardiff", "cardiff city": "Cardiff",
    "norwich": "Norwich", "norwich city": "Norwich", 
    "luton": "Luton", "luton town": "Luton",
    "derby": "Derby", "derby county": "Derby", 
    "oxford": "Oxford", "oxford united": "Oxford",
    "sheffield wed": "Sheffield Weds", "sheffield wednesday": "Sheffield Weds", 
    "shetticia wea": "Sheffield Weds", "sheila wea": "Sheffield Weds",
    "plymouth": "Plymouth", "plymouth argyle": "Plymouth", 
    "portsmouth": "Portsmouth",
    "nottm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
    "wolves": "Wolverhampton", "wolverhampton": "Wolverhampton",
    "sheff utd": "Sheffield United", "sheffield united": "Sheffield United", 
    "shettiots urs": "Sheffield United", "shottiois urs": "Sheffield United", "shettiois urd": "Sheffield United",
    "leeds": "Leeds", "leeds utd": "Leeds", 
    "manchester city": "Man City", "man city": "Man City",
    "wrexham": "Wrexham",
    "br newer": "Ipswich", "ipswich": "Ipswich",
    "mitlwatt": "Millwall", "millwall": "Millwall",

    # --- HISZPANIA ---
    "valiadolia": "Valladolid", "valladolid": "Valladolid", 
    "burgos cr": "Burgos", "burgos": "Burgos",
    "castetion": "Castellon", "castellon": "Castellon", 
    "racing santander": "Santander", "r santander": "Santander",
    "cultural leonesa": "Cultural Leonesa", "leonesa": "Cultural Leonesa",
    "real sociedad b": "Sociedad B", "sociedad b": "Sociedad B", "b real sociedad 8": "Sociedad B",
    "almeria": "Almeria", "granada": "Granada", "huesca": "Huesca", "cordoba": "Cordoba",
    "athletic bilbao": "Ath Bilbao", "atl madrid": "Ath Madrid", "atletico madrid": "Ath Madrid",
    "betis": "Real Betis", "real betis": "Real Betis", 
    "celta vigo": "Celta", "celta": "Celta",
    "cout": "Ceuta", "ceuta": "Ceuta",
    "f zoragozo": "Zaragoza", "zaragoza": "Zaragoza", "real zaragoza": "Zaragoza",
    
    # --- WŁOCHY ---
    "como": "Como", "udinese": "Udinese", "g genoa": "Genoa", "genoa": "Genoa",
    "piso": "Pisa", "pisa": "Pisa", "sassuolo": "Sassuolo", 
    "b parma": "Parma", "parma": "Parma",
    "y suventus": "Juventus", "juventus": "Juventus", 
    "lecce": "Lecce", "atalanta": "Atalanta",
    "as roma": "Roma", "roma": "Roma", 
    "inter": "Inter Milan", "inter milan": "Inter Milan", "ac milan": "Milan",
    
    # --- NIEMCY ---
    "monchengladbach": "M'gladbach", "b monchengladbach": "M'gladbach",
    "mainz": "Mainz 05", "frankfurt": "Ein Frankfurt", "eintracht frankfurt": "Ein Frankfurt",

    # --- FRANCJA ---
    "parc": "Pau FC", "pau": "Pau FC", "pau fc": "Pau FC",
    "parisre": "Paris FC", "paris fc": "Paris FC",
    "b tyon": "Lyon", "lyon": "Lyon", "olympique lyon": "Lyon",
    "thy morsylia": "Marseille", "marsylia": "Marseille", "marseille": "Marseille",
    "psc": "Paris SG", "psg": "Paris SG", "paris saint germain": "Paris SG"
}

LEAGUE_NAMES = {
    'E0': '🇬🇧 Anglia - Premier League', 'E1': '🇬🇧 Anglia - Championship',
    'D1': '🇩🇪 Niemcy - Bundesliga', 'D2': '🇩🇪 Niemcy - 2. Bundesliga',
    'I1': '🇮🇹 Włochy - Serie A', 'I2': '🇮🇹 Włochy - Serie B',
    'SP1': '🇪🇸 Hiszpania - La Liga', 'SP2': '🇪🇸 Hiszpania - La Liga 2',
    'F1': '🇫🇷 Francja - Ligue 1', 'F2': '🇫🇷 Francja - Ligue 2',
    'N1': '🇳🇱 Holandia - Eredivisie', 'P1': '🇵🇹 Portugalia - Liga Portugal',
    'B1': '🇧🇪 Belgia - Jupiler League', 'T1': '🇹🇷 Turcja - Super Lig',
    'G1': '🇬🇷 Grecja - Super League', 'SC0': '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Szkocja - Premiership',
    'POL': '🇵🇱 Polska - Ekstraklasa', 'Ekstraklasa': '🇵🇱 Polska - Ekstraklasa'
}

# --- FUNKCJE I BAZA DANYCH ---

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

def get_data_for_league(league_name):
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT * FROM all_leagues WHERE LeagueName = ?", conn, params=(league_name,))
        conn.close()
        return df
    except: return pd.DataFrame()

def get_all_data():
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT * FROM all_leagues", conn)
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
                        'id': row['ID'],
                        'name': row['Name'],
                        'date_created': row['DateCreated'],
                        'data': coupon_data
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
            'Mecz': bet['Mecz'],
            'Home': bet['Mecz'].split(' - ')[0],
            'Away': bet['Mecz'].split(' - ')[1],
            'Typ': bet['Typ'],
            'Date': bet.get('Date', 'N/A'),
            'Pewność': bet['Pewność'],
            'Result': '?',
            'Forma': bet.get('Forma', ''),
            'Stabilność': bet.get('Stabilność', '')
        })
    new_entry = {
        'ID': new_id,
        'Name': name,
        'DateCreated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'Data': json.dumps(simplified_data)
    }
    df_new = pd.DataFrame([new_entry])
    if os.path.exists(COUPONS_DB_FILE):
        df_new.to_csv(COUPONS_DB_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(COUPONS_DB_FILE, index=False)

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
            if status in ['✅', '❌']:
                processed_bets.append(bet)
                continue
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
    fthg, ftag = row['FTHG'], row['FTAG']
    goals = fthg + ftag
    try:
        if bet_type.startswith("Win"):
            if "Win " + row['HomeTeam'] == bet_type: return fthg > ftag
            if "Win " + row['AwayTeam'] == bet_type: return ftag > fthg
        if bet_type == "Over 2.5": return goals > 2.5
        if bet_type == "Over 1.5": return goals > 1.5
        if bet_type == "Over 0.5": return goals > 0.5
        if bet_type == "Under 4.5": return goals <= 4.5
        if bet_type == "Under 3.5": return goals <= 3.5
        if bet_type == "Under 2.5": return goals <= 2.5
        if bet_type == "BTS": return fthg > 0 and ftag > 0
        if bet_type == "1X": return fthg >= ftag
        if bet_type == "X2": return ftag >= fthg
        if bet_type == "12": return fthg != ftag
        if "strzeli" in bet_type:
            if row['HomeTeam'] in bet_type: return fthg > 0
            if row['AwayTeam'] in bet_type: return ftag > 0
        if "HT Over 1.5" in bet_type:
            if 'HTHG' in row and 'HTAG' in row: return (row['HTHG'] + row['HTAG']) > 1.5
            return False
    except: return False
    return False

def check_team_conflict(home, away, pool):
    for m in pool:
        if m['Home'] == home and m['Away'] == away:
            return f"⛔ Mecz {home} vs {away} jest już na liście!"
    return None

def clean_expired_matches(pool):
    today_str = datetime.today().strftime('%Y-%m-%d')
    new_pool = []
    removed = 0
    for m in pool:
        if 'Date' not in m or not m['Date'] or str(m['Date']) == 'nan':
            new_pool.append(m); continue
        try:
            if str(m['Date']) >= today_str: new_pool.append(m)
            else: removed += 1
        except: new_pool.append(m)
    return new_pool, removed

def process_uploaded_history(files):
    all_data = []
    for uploaded_file in files:
        try:
            bytes_data = uploaded_file.getvalue()
            try: df = pd.read_csv(io.BytesIO(bytes_data)); 
            except: df = pd.read_csv(io.BytesIO(bytes_data), sep=';')
            if len(df.columns) < 2: continue
            df.columns = [c.strip() for c in df.columns]
            base_req = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if not all(col in df.columns for col in base_req): continue
            cols = base_req + ['FTR']
            if 'HTHG' in df.columns and 'HTAG' in df.columns: cols.extend(['HTHG', 'HTAG'])
            df_cl = df[cols].copy().dropna(subset=['HomeTeam', 'FTHG'])
            df_cl['Date'] = pd.to_datetime(df_cl['Date'], dayfirst=True, errors='coerce')
            df_cl['LeagueName'] = df_cl['Div'].map(LEAGUE_NAMES).fillna(df_cl['Div'])
            all_data.append(df_cl)
        except Exception as e: st.error(f"Błąd pliku {uploaded_file.name}: {e}")
    if all_data:
        master = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect("mintstats.db")
        master.to_sql('all_leagues', conn, if_exists='replace', index=False)
        conn.close()
        return len(master)
    return 0

def clean_ocr_text_debug(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        normalized = re.sub(r'[^a-zA-Z0-9 ]', ' ', line).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        if "liga" in normalized.lower() or "serie" in normalized.lower(): continue
        if len(normalized) > 2: cleaned.append(normalized)
    return cleaned

def extract_text_from_image(uploaded_file):
    try: 
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image, lang='eng', config='--psm 6')
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
    lines = text_input.split('\n')
    found_matches = []
    today_str = datetime.today().strftime('%Y-%m-%d')
    for line in lines:
        line = line.strip()
        if not line: continue
        line = re.sub(r'\d{2}:\d{2}', '', line)
        parts = []
        if " - " in line: parts = line.split(" - ")
        elif " vs " in line: parts = line.split(" vs ")
        if len(parts) >= 2:
            raw_home = parts[0]
            raw_away_chunk = parts[1]
            raw_away = re.split(r'[\d\.]+', raw_away_chunk)[0]
            home_team = resolve_team_name(raw_home, available_teams)
            away_team = resolve_team_name(raw_away, available_teams)
            if home_team and away_team and home_team != away_team:
                found_matches.append({'Home': home_team, 'Away': away_team, 'League': 'Text Import', 'Date': today_str})
    return found_matches

def smart_parse_matches_v3(text_input, available_teams):
    cleaned_lines = clean_ocr_text_debug(text_input)
    found_teams = []
    debug_log = []
    today_str = datetime.today().strftime('%Y-%m-%d')
    for line in cleaned_lines:
        cur = line.lower().strip()
        matched = resolve_team_name(cur, available_teams)
        if matched:
            if not found_teams or found_teams[-1] != matched: found_teams.append(matched)
            debug_log.append(f"✅ '{cur}' -> '{matched}'")
        else:
            debug_log.append(f"❌ '{cur}'")
    matches = [{'Home': found_teams[i], 'Away': found_teams[i+1], 'League': 'OCR Import', 'Date': today_str} for i in range(0, len(found_teams) - 1, 2)]
    return matches, debug_log, cleaned_lines

def parse_fixtures_csv(file):
    try:
        df = pd.read_csv(file)
        if not {'Div', 'HomeTeam', 'AwayTeam'}.issubset(df.columns): return [], "Brak kolumn Div/HomeTeam/AwayTeam"
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            df['Date'] = datetime.today().strftime('%Y-%m-%d')
        matches = []
        for _, row in df.iterrows():
            matches.append({'Home': row['HomeTeam'], 'Away': row['AwayTeam'], 'League': row['Div'], 'Date': row['Date']})
        return matches, None
    except Exception as e: return [], str(e)

# --- MODEL POISSONA Z CHAOSEM I MONTE CARLO ---
class PoissonModel:
    def __init__(self, data):
        self.data = data
        self.team_stats_ft = {}
        self.team_stats_ht = {}
        self.team_form = {} 
        self.team_chaos = {} # Nowy wskaźnik
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
        if has_ht: lg_ht = self.data['HTHG'].sum() + self.data['HTAG'].sum(); self.league_avg_ht = lg_ht / matches if matches > 0 else 1.0
        
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for team in teams:
            home = self.data[self.data['HomeTeam'] == team]; away = self.data[self.data['AwayTeam'] == team]
            scored_ft = home['FTHG'].sum() + away['FTAG'].sum(); conceded_ft = home['FTAG'].sum() + away['FTHG'].sum()
            played = len(home) + len(away)
            if played > 0:
                self.team_stats_ft[team] = {'attack': (scored_ft/played)/self.league_avg_ft, 'defense': (conceded_ft/played)/self.league_avg_ft}
                if has_ht:
                    scored_ht = home['HTHG'].sum() + away['HTAG'].sum(); conceded_ht = home['HTAG'].sum() + away['HTHG'].sum()
                    self.team_stats_ht[team] = {'attack': (scored_ht/played)/self.league_avg_ht, 'defense': (conceded_ht/played)/self.league_avg_ht}

    def _calculate_form_and_chaos(self):
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for team in teams:
            # 1. Forma (Ostatnie 5)
            matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)].tail(5)
            form_icons = []
            scored_recent = 0
            for _, row in matches.iterrows():
                is_home = row['HomeTeam'] == team
                gf = row['FTHG'] if is_home else row['FTAG']
                ga = row['FTAG'] if is_home else row['FTHG']
                scored_recent += gf
                if gf > ga: form_icons.append("🟢")
                elif gf == ga: form_icons.append("🤝")
                else: form_icons.append("🔴")
            
            att_boost = 1.0
            if len(matches) > 0:
                avg_scored = scored_recent / len(matches)
                att_boost = 1.0 + (avg_scored * 0.05)
            
            self.team_form[team] = {'icons': "".join(reversed(form_icons)), 'att_boost': att_boost}

            # 2. Chaos (Standard Deviation) - Cały sezon
            all_matches = self.data[(self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)]
            goals_sequence = []
            for _, row in all_matches.iterrows():
                gf = row['FTHG'] if row['HomeTeam'] == team else row['FTAG']
                goals_sequence.append(gf)
            
            if len(goals_sequence) > 3:
                std_dev = np.std(goals_sequence)
                # Interpretacja: StdDev > 1.5 to duży chaos (wyniki typu 0, 4, 1, 5)
                # StdDev < 0.8 to stabilność (wyniki typu 1, 1, 2, 1)
                chaos_rating = "Stabilny 🧊" if std_dev < 0.9 else ("Chaos 🌪️" if std_dev > 1.4 else "Norma")
                chaos_penalty = 0.9 if std_dev > 1.4 else (1.05 if std_dev < 0.9 else 1.0)
                self.team_chaos[team] = {'rating': chaos_rating, 'factor': chaos_penalty}
            else:
                self.team_chaos[team] = {'rating': "Brak danych", 'factor': 1.0}

    def get_h2h_analysis(self, home, away):
        mask = ((self.data['HomeTeam'] == home) & (self.data['AwayTeam'] == away)) | \
               ((self.data['HomeTeam'] == away) & (self.data['AwayTeam'] == home))
        h2h_matches = self.data[mask].sort_values(by='Date', ascending=False).head(5)
        if h2h_matches.empty: return None
        home_wins = 0
        for _, row in h2h_matches.iterrows():
            if (row['HomeTeam'] == home and row['FTHG'] > row['FTAG']) or (row['AwayTeam'] == home and row['FTAG'] > row['FTHG']):
                home_wins += 1
        if len(h2h_matches) >= 3 and home_wins == 0: return "⚠️ H2H (Kryptonit!)"
        return None

    # --- MONTE CARLO SIMULATION ---
    def simulate_match_monte_carlo(self, xg_h, xg_a, n=1000):
        # Symulacja 1000 meczów z lekkim szumem (noise)
        home_wins = 0
        for _ in range(n):
            # Dodajemy losowy szum do xG (form dyspozycja dnia +/- 20%)
            adj_h = xg_h * np.random.uniform(0.8, 1.2)
            adj_a = xg_a * np.random.uniform(0.8, 1.2)
            
            sim_h_goals = np.random.poisson(adj_h)
            sim_a_goals = np.random.poisson(adj_a)
            
            if sim_h_goals > sim_a_goals: home_wins += 1
            
        stability = (home_wins / n) * 100
        return stability

    def predict(self, home, away):
        if home not in self.team_stats_ft or away not in self.team_stats_ft: return None, None, None, None
        
        h_att = self.team_stats_ft[home]['attack']; h_def = self.team_stats_ft[home]['defense']
        a_att = self.team_stats_ft[away]['attack']; a_def = self.team_stats_ft[away]['defense']
        
        # Boost formy
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
        prob_0_0 = poisson.pmf(0, xg_h_ft) * poisson.pmf(0, xg_a_ft)

        return {
            "1": prob_1, "X": prob_x, "2": prob_2, 
            "1X": prob_1+prob_x, "X2": prob_x+prob_2, "12": prob_1+prob_2,
            "BTS_Yes": np.sum(mat_ft[1:, 1:]), "BTS_No": 1.0-np.sum(mat_ft[1:, 1:]),
            "Over_1.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Over_2.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 2.5]),
            "Over_0.5_FT": 1.0 - prob_0_0,
            "Under_2.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 2.5]),
            "Under_3.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 3.5]),
            "Under_4.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 4.5]),
            "Over_1.5_HT": np.sum([mat_ht[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Home_Yes": 1.0 - poisson.pmf(0, xg_h_ft), "Away_Yes": 1.0 - poisson.pmf(0, xg_a_ft)
        }
    
    def get_team_info(self, team):
        form = self.team_form.get(team, {'icons': '⚪', 'att_boost': 1.0})['icons']
        chaos = self.team_chaos.get(team, {'rating': '-', 'factor': 1.0})
        return form, chaos

class CouponGenerator:
    def __init__(self, model): self.model = model
    def analyze_pool(self, pool, strategy="Mix Bezpieczny"):
        res = []
        for m in pool:
            xg_h, xg_a, xg_h_ht, xg_a_ht = self.model.predict(m['Home'], m['Away'])
            if xg_h is None: continue
            probs = self.model.calculate_probs(xg_h, xg_a, xg_h_ht, xg_a_ht)
            
            # Pobierz dane analityczne
            h2h_warning = self.model.get_h2h_analysis(m['Home'], m['Away'])
            form_h, chaos_h = self.model.get_team_info(m['Home'])
            form_a, chaos_a = self.model.get_team_info(m['Away'])
            
            # --- ZASTOSOWANIE CZYNNIKA CHAOSU ---
            # Mnożymy prawdopodobieństwo przez czynnik stabilności
            chaos_factor = chaos_h['factor'] * chaos_a['factor']
            # Spłaszczamy pewność (np. 0.8 * 0.9 = 0.72)
            for key in probs: probs[key] *= chaos_factor

            # --- MONTE CARLO (Dla typów na zwycięstwo) ---
            mc_stability = self.model.simulate_match_monte_carlo(xg_h, xg_a)
            
            warning = ""
            if "🔴🔴🔴" in form_h and probs['1'] > 0.6: warning = "⚠️ KRYZYS GOSP."
            if "🔴🔴🔴" in form_a and probs['2'] > 0.6: warning = "⚠️ KRYZYS GOŚĆ"
            if h2h_warning and probs['1'] > 0.5: warning += " " + h2h_warning

            # Opis chaosu do tabeli
            chaos_desc = ""
            if "🌪️" in chaos_h['rating']: chaos_desc += f"🌪️ {m['Home']} "
            if "🌪️" in chaos_a['rating']: chaos_desc += f"🌪️ {m['Away']}"
            if "🧊" in chaos_h['rating'] and "🧊" in chaos_a['rating']: chaos_desc = "🧊 STABLE"

            potential_bets = []
            
            if "Mix Bezpieczny" in strategy:
                potential_bets.append({'typ': "1X", 'prob': probs['1X'], 'cat': 'DC'})
                potential_bets.append({'typ': "X2", 'prob': probs['X2'], 'cat': 'DC'})
                potential_bets.append({'typ': "Under 4.5", 'prob': probs['Under_4.5_FT'], 'cat': 'U/O'})
                potential_bets.append({'typ': "Over 0.5", 'prob': probs['Over_0.5_FT'], 'cat': 'U/O'})
                potential_bets.append({'typ': f"{m['Home']} strzeli", 'prob': probs['Home_Yes'], 'cat': 'TEAM'})
                potential_bets.append({'typ': f"{m['Away']} strzeli", 'prob': probs['Away_Yes'], 'cat': 'TEAM'})

            elif "Podwójna Szansa" in strategy:
                potential_bets.append({'typ': "1X", 'prob': probs['1X'], 'cat': 'MAIN'})
                potential_bets.append({'typ': "X2", 'prob': probs['X2'], 'cat': 'MAIN'})
                potential_bets.append({'typ': "12", 'prob': probs['12'], 'cat': 'MAIN'})

            elif "Gole Agresywne" in strategy:
                potential_bets.append({'typ': "BTS", 'prob': probs['BTS_Yes'], 'cat': 'MAIN'})
                potential_bets.append({'typ': "Over 2.5", 'prob': probs['Over_2.5_FT'], 'cat': 'MAIN'})

            elif "Do Przerwy" in strategy:
                potential_bets.append({'typ': "HT Over 1.5", 'prob': probs['Over_1.5_HT'], 'cat': 'MAIN'})

            elif "Twierdza" in strategy:
                potential_bets.append({'typ': f"Win {m['Home']}", 'prob': probs['1'], 'cat': 'MAIN'})

            elif "Mur Obronny" in strategy:
                potential_bets.append({'typ': "Under 2.5", 'prob': probs['Under_2.5_FT'], 'cat': 'MAIN'})
                potential_bets.append({'typ': "Under 3.5", 'prob': probs['Under_3.5_FT'], 'cat': 'MAIN'})

            elif "Złoty Środek" in strategy:
                potential_bets.append({'typ': "Over 1.5", 'prob': probs['Over_1.5_FT'], 'cat': 'MAIN'})

            elif "Wszystkie" in strategy:
                potential_bets = [
                    {'typ': "1", 'prob': probs['1'], 'cat': 'MAIN'}, {'typ': "2", 'prob': probs['2'], 'cat': 'MAIN'},
                    {'typ': "1X", 'prob': probs['1X'], 'cat': 'MAIN'}, {'typ': "X2", 'prob': probs['X2'], 'cat': 'MAIN'},
                    {'typ': "Over 2.5", 'prob': probs['Over_2.5_FT'], 'cat': 'MAIN'}, {'typ': "Under 4.5", 'prob': probs['Under_4.5_FT'], 'cat': 'MAIN'},
                    {'typ': "BTS", 'prob': probs['BTS_Yes'], 'cat': 'MAIN'}
                ]

            if potential_bets:
                best = sorted(potential_bets, key=lambda x: x['prob'], reverse=True)[0]
                
                combined_form = f"{form_h} vs {form_a}"
                if warning: combined_form += f" {warning}"
                
                # Dodajemy info o stabilności do tabeli (tylko dla typów na wygraną)
                mc_info = ""
                if "Win" in best['typ'] or "1X" in best['typ']:
                    if mc_stability > 80: mc_info = " (MC: 80%+)"
                    elif mc_stability < 50 and probs['1'] > 0.5: mc_info = " (MC: RYZYKO)"

                res.append({
                    'Mecz': f"{m['Home']} - {m['Away']}", 
                    'Liga': m.get('League', 'N/A'), 
                    'Date': m.get('Date', 'N/A'),
                    'Typ': best['typ'] + mc_info, 
                    'Pewność': best['prob'], 
                    'Kategoria': best.get('cat', 'MAIN'),
                    'Forma': combined_form,
                    'Stabilność': chaos_desc,
                    'xG': f"{xg_h:.2f}:{xg_a:.2f}"
                })
        return res

# --- INIT ---
if 'fixture_pool' not in st.session_state: st.session_state.fixture_pool = load_fixture_pool()
if 'generated_coupons' not in st.session_state: st.session_state.generated_coupons = [] 
if 'last_ocr_debug' not in st.session_state: st.session_state.last_ocr_debug = None

# --- INTERFEJS ---
st.title("☁️ MintStats v18.0: Analytical Beast")

st.sidebar.header("Panel Sterowania")
mode = st.sidebar.radio("Wybierz moduł:", ["1. 🛠️ ADMIN (Baza Danych)", "2. 🚀 GENERATOR KUPONÓW", "3. 📜 MOJE KUPONY"])

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
    uploaded_history = st.file_uploader("Wgraj pliki ligowe (Historia)", type=['csv'], accept_multiple_files=True)
    if uploaded_history and st.button("Aktualizuj Bazę Danych"):
        with st.spinner("Przetwarzanie..."):
            count = process_uploaded_history(uploaded_history)
            if count > 0: st.success(f"✅ Baza zaktualizowana ({count} meczów).")
            else: st.error("Błąd importu.")
    leagues = get_leagues_list()
    if leagues:
        st.write("---"); st.success(f"Dostępne ligi w bazie: {len(leagues)}"); st.write(leagues)
    else: st.warning("Baza pusta!")

elif mode == "2. 🚀 GENERATOR KUPONÓW":
    leagues = get_leagues_list()
    if not leagues: st.error("⛔ Baza pusta!"); st.stop()
        
    df_all = get_all_data()
    model = PoissonModel(df_all)
    gen = CouponGenerator(model)
    all_teams_list = pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).unique()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dodaj Mecze")
    tab_manual, tab_ocr, tab_text, tab_csv = st.sidebar.tabs(["Ręczny", "📸 Zdjęcie", "📝 Wklej Tekst", "📁 CSV"])
    
    new_items = []
    with tab_manual:
        sel_league = st.selectbox("Liga:", leagues)
        df_l = get_data_for_league(sel_league)
        teams = sorted(pd.concat([df_l['HomeTeam'], df_l['AwayTeam']]).unique())
        with st.form("manual_add"):
            col_date, col_h, col_a = st.columns([1,2,2])
            with col_date: date_input = st.date_input("Data", datetime.today())
            with col_h: h = st.selectbox("Dom", teams)
            with col_a: a = st.selectbox("Wyjazd", teams)
            if st.form_submit_button("➕ Dodaj") and h!=a: 
                new_items.append({'Home':h, 'Away':a, 'League':sel_league, 'Date': str(date_input)})
    
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
            if parsed:
                new_items.extend(parsed); st.success(f"✅ Znaleziono {len(parsed)} par!")
            else: st.error("Brak par.")

    with tab_csv:
        uploaded_fix = st.file_uploader("fixtures.csv", type=['csv'])
        if uploaded_fix and st.button("📥 Import"):
            m_list, err = parse_fixtures_csv(uploaded_fix)
            if not err: new_items.extend(m_list); st.success(f"Import {len(m_list)}")
            else: st.error(err)

    if new_items:
        added_count = 0
        errors = []
        for item in new_items:
            conflict_msg = check_team_conflict(item['Home'], item['Away'], st.session_state.fixture_pool)
            if conflict_msg: errors.append(conflict_msg)
            else:
                st.session_state.fixture_pool.append(item); added_count += 1
        save_fixture_pool(st.session_state.fixture_pool)
        if added_count > 0: st.toast(f"Dodano {added_count} meczów!")
        if errors:
            with st.expander("⚠️ Pominięto (Duplikaty)", expanded=True):
                for e in errors: st.warning(e)
        st.rerun()

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
            "Wszystkie Zdarzenia (Max Pewność)"
        ])
        with c3:
            if gen_mode == "Jeden Pewny Kupon": coupon_len = st.number_input("Długość", 1, 50, 12)
            else:
                num_coupons = st.number_input("Ile kuponów?", 1, 10, 3)
                events_per_coupon = st.number_input("Mecze na kupon?", 1, 20, 5)
                chaos_factor = st.slider("Pula (Top X)", 10, 100, 30)

        if st.button("🚀 GENERUJ", type="primary"):
            analyzed_pool = gen.analyze_pool(st.session_state.fixture_pool, strat)
            
            # --- LOGIKA SORTOWANIA (KARUZELA DLA MIXU) ---
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
            else:
                final_pool = sorted(analyzed_pool, key=lambda x: x['Pewność'], reverse=True)

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
                for kupon in st.session_state.generated_coupons:
                    save_new_coupon(kupon['name'], kupon['data'])
                st.success("Zapisano w Historii!")
            
            for kupon in st.session_state.generated_coupons:
                with st.container():
                    st.subheader(f"🎫 {kupon['name']}")
                    df_k = pd.DataFrame(kupon['data'])
                    if not df_k.empty:
                        disp_cols = ['Date', 'Mecz', 'Forma', 'Stabilność', 'Liga', 'Typ', 'Pewność', 'xG']
                        st.dataframe(df_k[disp_cols].style.background_gradient(subset=['Pewność'], cmap="RdYlGn", vmin=0.4, vmax=0.9).format({'Pewność':'{:.1%}'}), use_container_width=True)
                        st.caption(f"Średnia pewność: {df_k['Pewność'].mean()*100:.1f}%")
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
                    color = 'white'
                    if val == '✅': color = '#90EE90'
                    elif val == '❌': color = '#FFB6C1'
                    return f'background-color: {color}'
                st.dataframe(df_c.style.applymap(highlight_result, subset=['Result']), use_container_width=True)
                wins = len(df_c[df_c['Result'] == '✅'])
                losses = len(df_c[df_c['Result'] == '❌'])
                pending = len(df_c[df_c['Result'] == '?'])
                st.caption(f"✅ Trafione: {wins} | ❌ Pudła: {losses} | ⏳ Oczekujące: {pending}")
        if st.button("🗑️ Wyczyść Historię"):
            if os.path.exists(COUPONS_DB_FILE): os.remove(COUPONS_DB_FILE); st.rerun()
    else: st.info("Brak kuponów.")
