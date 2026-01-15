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

# --- 1. KONFIGURACJA ---
st.set_page_config(page_title="MintStats v28.3 Complete", layout="wide", page_icon="💎")
FIXTURES_DB_FILE = "my_fixtures.csv"
COUPONS_DB_FILE = "my_coupons.csv"

# ==============================================================================
# 🔑 TWOJE KLUCZE CHMURY (ZACHOWANE)
# ==============================================================================
JSONBIN_API_KEY = "$2a$10$emn.LOTwDQ2d/ibHmOxcY.Ogunk18boKpW6ubGj/.fG6kyH44ClFi"
JSONBIN_BIN_ID  = "6968d66c43b1c97be9323f01"
# ==============================================================================

# --- 2. CUSTOM CSS ---
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

# --- 3. SŁOWNIKI ---
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
    'USA': '🇺🇸 USA - MLS', 'MEX': '🇲🇽 Meksyk - Liga MX', 'JPN': '🇯🇵 Japonia - J-League',
    'CHN': '🇨🇳 Chiny - Super League', 'BRA': '🇧🇷 Brazylia - Serie A', 'BR1': '🇧🇷 Brazylia - Serie A',
    'ARG': '🇦🇷 Argentyna - Primera Division'
}

# --- 4. FUNKCJE CLOUD SYNC (JSONBIN) ---
def get_cloud_blacklist():
    url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest"
    headers = {"X-Master-Key": JSONBIN_API_KEY}
    try:
        req = requests.get(url, headers=headers)
        if req.status_code == 200:
            data = req.json().get('record', {})
            return set(data.get('matches', []))
    except: pass
    return set()

def update_cloud_blacklist(new_matches):
    current_list = get_cloud_blacklist()
    updated_list = list(current_list.union(set(new_matches)))
    url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
    headers = {"Content-Type": "application/json", "X-Master-Key": JSONBIN_API_KEY}
    payload = {"matches": updated_list, "last_updated": str(datetime.now())}
    try: requests.put(url, json=payload, headers=headers)
    except: pass

# --- FUNKCJE POMOCNICZE BAZY DANYCH ---
def get_leagues_list():
    try:
        conn = sqlite3.connect("mintstats.db"); cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='all_leagues';")
        if cursor.fetchone() is None: return []
        df = pd.read_sql("SELECT DISTINCT LeagueName FROM all_leagues ORDER BY LeagueName", conn)
        conn.close(); return df['LeagueName'].tolist()
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
        clean_match_name = bet['Mecz'].replace("🚫 ", "").replace(" (Już grany!)", "")
        simplified_data.append({
            'Mecz': clean_match_name, 
            'Home': clean_match_name.split(' - ')[0], 
            'Away': clean_match_name.split(' - ')[1],
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
    for uploaded_file in files:
        try:
            bytes_data = uploaded_file.getvalue()
            try: df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8')
            except: df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1')
            if len(df.columns) < 2: continue
            df.columns = [c.strip() for c in df.columns]
            renames = {'Home': 'HomeTeam', 'Away': 'AwayTeam', 'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'}
            df.rename(columns=renames, inplace=True)
            if 'Div' not in df.columns: df['Div'] = uploaded_file.name.replace('.csv', '').upper()
            detected_codes.add(df['Div'].iloc[0])
            req_cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if all(c in df.columns for c in req_cols):
                cols = req_cols + (['HTHG', 'HTAG'] if 'HTHG' in df.columns else [])
                df_cl = df[cols].copy().dropna(subset=['HomeTeam', 'FTHG'])
                df_cl['Date'] = pd.to_datetime(df_cl['Date'], dayfirst=True, errors='coerce')
                df_cl['LeagueName'] = df_cl['Div'].map(LEAGUE_NAMES).fillna(df_cl['Div'])
                all_data.append(df_cl)
        except: continue
    if all_data:
        master = pd.concat(all_data, ignore_index=True)
        conn = sqlite3.connect("mintstats.db")
        master.to_sql('all_leagues', conn, if_exists='replace', index=False)
        conn.close()
        return len(master), list(detected_codes), []
    return 0, [], []

def download_and_update_db(league_codes):
    today = datetime.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    season = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
    base_url = f"https://www.football-data.co.uk/mmz4281/{season}/"
    all_dfs = []; count = 0
    codes = list(set([k for k in LEAGUE_NAMES.keys() if len(k) <= 4]))
    for code in codes:
        try:
            r = requests.get(f"{base_url}{code}.csv", timeout=5)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                df.rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam', 'HG': 'FTHG', 'AG': 'FTAG', 'Res': 'FTR'}, inplace=True)
                if 'Div' not in df.columns: df['Div'] = code
                if {'Date', 'HomeTeam', 'FTHG'}.issubset(df.columns):
                    df = df.dropna(subset=['HomeTeam', 'FTHG'])
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    df['LeagueName'] = df['Div'].map(LEAGUE_NAMES).fillna(df['Div'])
                    all_dfs.append(df); count += 1
        except: continue
    if all_dfs:
        final = pd.concat(all_dfs, ignore_index=True)
        conn = sqlite3.connect("mintstats.db")
        try:
            old = pd.read_sql("SELECT * FROM all_leagues", conn)
            old['Date'] = pd.to_datetime(old['Date'])
            hist = old[old['Date'] < pd.to_datetime(f"{start_year}-07-01")]
            final = pd.concat([hist, final], ignore_index=True)
        except: pass
        final.to_sql('all_leagues', conn, if_exists='replace', index=False)
        conn.close()
        return count, len(final)
    return 0, 0

def get_db_status():
    try:
        conn = sqlite3.connect("mintstats.db")
        df = pd.read_sql("SELECT LeagueName as Liga, COUNT(*) as Mecze, MAX(Date) as Ostatni_Mecz FROM all_leagues GROUP BY LeagueName ORDER BY Ostatni_Mecz DESC", conn)
        df['Ostatni_Mecz'] = pd.to_datetime(df['Ostatni_Mecz']).dt.strftime('%Y-%m-%d')
        conn.close()
        return df
    except: return pd.DataFrame()

# --- 5. WYKRESY ---
def create_radar_chart(h_stats, a_stats, h_name, a_name):
    categories = ['Atak', 'Obrona', 'Forma', 'Chaos']
    h_v = [min(h_stats['att']*50,100), min((2-h_stats['def'])*50,100), h_stats['form_score'], h_stats['chaos_score']]
    a_v = [min(a_stats['att']*50,100), min((2-a_stats['def'])*50,100), a_stats['form_score'], a_stats['chaos_score']]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_v, theta=categories, fill='toself', name=h_name, line_color='#00C896'))
    fig.add_trace(go.Scatterpolar(r=a_v, theta=categories, fill='toself', name=a_name, line_color='#FF4B4B'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=250, margin=dict(l=30,r=30,t=30,b=30))
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

# --- 6. MODEL POISSONA ---
class PoissonModel:
    def __init__(self, data):
        self.data = data
        self.team_stats_ft = {}; self.team_form = {}; self.team_chaos = {}
        self.league_avg_ft = 1.0; self.home_adv = 1.15
        if not data.empty:
            self._calc_stats()

    def _calc_stats(self):
        lg_goals = self.data['FTHG'].sum() + self.data['FTAG'].sum()
        self.league_avg_ft = lg_goals / (len(self.data)*2) if len(self.data)>0 else 1.0
        h_g = self.data['FTHG'].sum(); a_g = self.data['FTAG'].sum()
        if a_g > 0: self.home_adv = h_g / a_g
        
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for t in teams:
            h = self.data[self.data['HomeTeam']==t]; a = self.data[self.data['AwayTeam']==t]
            played = len(h)+len(a)
            if played > 0:
                sc = h['FTHG'].sum() + a['FTAG'].sum(); conc = h['FTAG'].sum() + a['FTHG'].sum()
                self.team_stats_ft[t] = {'att': (sc/played)/self.league_avg_ft, 'def': (conc/played)/self.league_avg_ft}
            
            # Form & Chaos
            last5 = self.data[(self.data['HomeTeam']==t) | (self.data['AwayTeam']==t)].tail(5)
            icons = []
            for _, r in last5.iterrows():
                gf = r['FTHG'] if r['HomeTeam']==t else r['FTAG']
                ga = r['FTAG'] if r['HomeTeam']==t else r['FTHG']
                icons.append("🟢" if gf>ga else ("🔴" if gf<ga else "🤝"))
            
            all_m = self.data[(self.data['HomeTeam']==t) | (self.data['AwayTeam']==t)]
            goals = [r['FTHG'] if r['HomeTeam']==t else r['FTAG'] for _, r in all_m.iterrows()]
            std = np.std(goals) if len(goals)>3 else 1.0
            chaos_s = max(0, min(100, 100 - (std*30)))
            
            self.team_form[t] = {'icons': "".join(reversed(icons)), 'score': 50 + (icons.count("🟢")*10)}
            self.team_chaos[t] = {'rating': "Stable" if std<0.9 else "Chaos", 'factor': 0.9 if std>1.4 else 1.0, 'score': chaos_s}

    def predict(self, h, a):
        if h not in self.team_stats_ft or a not in self.team_stats_ft: return None, None
        h_att = self.team_stats_ft[h]['att']; h_def = self.team_stats_ft[h]['def']
        a_att = self.team_stats_ft[a]['att']; a_def = self.team_stats_ft[a]['def']
        xg_h = h_att * a_def * self.league_avg_ft * self.home_adv
        xg_a = a_att * h_def * self.league_avg_ft
        return xg_h, xg_a

class CouponGenerator:
    def __init__(self, model): self.model = model
    
    def analyze_pool(self, pool, strategy, used_matches_set):
        res = []
        for m in pool:
            match_id = f"{m['Home']} - {m['Away']}"
            if match_id in used_matches_set: continue 
            
            xg_h, xg_a = self.model.predict(m['Home'], m['Away'])
            if xg_h is None: continue
            
            # Simple Poisson Probs
            h_probs = [poisson.pmf(i, xg_h) for i in range(6)]
            a_probs = [poisson.pmf(i, xg_a) for i in range(6)]
            
            p_1 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i>j)
            p_x = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i==j)
            p_2 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i<j)
            
            p_bts_yes = sum(h_probs[i]*a_probs[j] for i in range(1,6) for j in range(1,6))
            p_bts_no = 1.0 - p_bts_yes
            
            p_o15 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i+j > 1.5)
            p_o25 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i+j > 2.5)
            p_o05 = 1.0 - (h_probs[0]*a_probs[0])
            
            p_u25 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i+j <= 2.5)
            p_u35 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i+j <= 3.5)
            p_u45 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i+j <= 4.5)
            
            p_home_yes = 1.0 - h_probs[0]
            p_away_yes = 1.0 - a_probs[0]
            
            # Handicaps & DNB
            p_h_handi = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i >= j+2)
            p_a_handi = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if j >= i+2)
            p_h_plus = 1.0 - p_a_handi
            p_a_plus = 1.0 - p_h_handi
            
            p_dnb_h = p_1 / (p_1 + p_2) if (p_1+p_2) > 0 else 0
            p_dnb_a = p_2 / (p_1 + p_2) if (p_1+p_2) > 0 else 0
            
            # --- LOGIKA STRATEGII ---
            bets = []
            if "Mix Bezpieczny" in strategy:
                bets.append({'typ': "1X", 'prob': p_1+p_x, 'cat': 'DC'})
                bets.append({'typ': "X2", 'prob': p_2+p_x, 'cat': 'DC'})
                bets.append({'typ': "Under 4.5", 'prob': p_u45, 'cat': 'U/O'})
                bets.append({'typ': "Over 0.5", 'prob': p_o05, 'cat': 'U/O'})
                bets.append({'typ': f"{m['Home']} strzeli", 'prob': p_home_yes, 'cat': 'TEAM'})
            elif "Podwójna Szansa" in strategy:
                bets.append({'typ': "1X", 'prob': p_1+p_x, 'cat': 'MAIN'})
                bets.append({'typ': "X2", 'prob': p_2+p_x, 'cat': 'MAIN'})
                bets.append({'typ': "12", 'prob': p_1+p_2, 'cat': 'MAIN'})
            elif "Gole Agresywne" in strategy:
                bets.append({'typ': "BTS", 'prob': p_bts_yes, 'cat': 'MAIN'})
                bets.append({'typ': "Over 2.5", 'prob': p_o25, 'cat': 'MAIN'})
            elif "Do Przerwy" in strategy:
                xg_h_ht, xg_a_ht = xg_h * 0.45, xg_a * 0.45
                p_ht_o15 = 1.0 - (poisson.pmf(0, xg_h_ht+xg_a_ht) + poisson.pmf(1, xg_h_ht+xg_a_ht))
                bets.append({'typ': "HT Over 1.5", 'prob': p_ht_o15, 'cat': 'MAIN'})
            elif "Twierdza" in strategy:
                bets.append({'typ': f"Win {m['Home']}", 'prob': p_1, 'cat': 'MAIN'})
            elif "Mur Obronny" in strategy:
                bets.append({'typ': "Under 2.5", 'prob': p_u25, 'cat': 'MAIN'})
                bets.append({'typ': "Under 3.5", 'prob': p_u35, 'cat': 'MAIN'})
            elif "Złoty Środek" in strategy:
                bets.append({'typ': "Over 1.5", 'prob': p_o15, 'cat': 'MAIN'})
            elif "Obie strzelą (TAK)" in strategy: bets.append({'typ': "BTS", 'prob': p_bts_yes})
            elif "Obie strzelą (NIE)" in strategy: bets.append({'typ': "BTS NO", 'prob': p_bts_no})
            elif "1 drużyna strzeli (TAK)" in strategy: bets.append({'typ': f"{m['Home']} strzeli", 'prob': p_home_yes})
            elif "1 drużyna strzeli (NIE)" in strategy: bets.append({'typ': f"{m['Home']} nie strzeli", 'prob': 1.0-p_home_yes})
            elif "2 drużyna strzeli (TAK)" in strategy: bets.append({'typ': f"{m['Away']} strzeli", 'prob': p_away_yes})
            elif "2 drużyna strzeli (NIE)" in strategy: bets.append({'typ': f"{m['Away']} nie strzeli", 'prob': 1.0-p_away_yes})
            elif "Handicap: Dominacja Faworyta" in strategy:
                bets.append({'typ': f"{m['Home']} (-1.5)", 'prob': p_h_handi})
                bets.append({'typ': f"{m['Away']} (-1.5)", 'prob': p_a_handi})
            elif "Handicap: Tarcza Underdoga" in strategy:
                bets.append({'typ': f"{m['Home']} (+1.5)", 'prob': p_h_plus})
                bets.append({'typ': f"{m['Away']} (+1.5)", 'prob': p_a_plus})
            elif "DNB: Gospodarz" in strategy: bets.append({'typ': f"DNB {m['Home']}", 'prob': p_dnb_h})
            elif "DNB: Gość" in strategy: bets.append({'typ': f"DNB {m['Away']}", 'prob': p_dnb_a})

            if bets:
                best = sorted(bets, key=lambda x: x['prob'], reverse=True)[0]
                chaos_mod = self.model.team_chaos[m['Home']]['factor'] * self.model.team_chaos[m['Away']]['factor']
                
                res.append({
                    'Mecz': match_id, 'Liga': m['League'], 'Date': m['Date'],
                    'Typ': best['typ'], 'Pewność': best['prob'] * chaos_mod,
                    'HomeStats': self.model.team_stats_ft[m['Home']],
                    'AwayStats': self.model.team_stats_ft[m['Away']],
                    'xG': f"{xg_h:.2f}:{xg_a:.2f}"
                })
        return res

# --- 7. LOGIKA LAB (BACKTEST) ---
def run_backtest(df, strategy, limit=50):
    df = df.sort_values(by='Date', ascending=False).head(limit)
    df = df.sort_values(by='Date', ascending=True)
    model = PoissonModel(df) 
    gen = CouponGenerator(model)
    pool = []
    for _, row in df.iterrows(): pool.append({'Home': row['HomeTeam'], 'Away': row['AwayTeam'], 'League': 'Test', 'Date': row['Date']})
    generated_tips = gen.analyze_pool(pool, strategy, set())
    results = {'Correct': 0, 'Wrong': 0, 'Total': 0}
    for tip in generated_tips:
        home, away = tip['Mecz'].split(' - ')
        match = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away)]
        if not match.empty:
            actual = match.iloc[0]
            hit = False
            fthg, ftag = actual['FTHG'], actual['FTAG']
            typ = tip['Typ']
            # Uproszczona ewaluacja
            if "1" in typ and "X" not in typ and "DNB" not in typ and fthg > ftag: hit = True
            elif "2" in typ and "X" not in typ and "DNB" not in typ and ftag > fthg: hit = True
            elif "X" in typ and "1" not in typ and "2" not in typ and fthg == ftag: hit = True
            elif "1X" in typ and fthg >= ftag: hit = True
            elif "X2" in typ and ftag >= fthg: hit = True
            elif "Over 2.5" in typ and (fthg+ftag) > 2.5: hit = True
            elif "Under 3.5" in typ and (fthg+ftag) < 3.5: hit = True
            elif "BTS" in typ and "NO" not in typ and fthg>0 and ftag>0: hit = True
            
            if hit: results['Correct'] += 1
            else: results['Wrong'] += 1
            results['Total'] += 1
    return results

def calculate_xpts_table(df):
    model = PoissonModel(df)
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    table = {t: {'P': 0, 'xPts': 0.0} for t in teams}
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if row['FTHG'] > row['FTAG']: table[h]['P'] += 3
        elif row['FTHG'] == row['FTAG']: table[h]['P'] += 1; table[a]['P'] += 1
        else: table[a]['P'] += 3
        xg_h, xg_a = model.predict(h, a)
        if xg_h:
            h_probs = [poisson.pmf(i, xg_h) for i in range(6)]
            a_probs = [poisson.pmf(i, xg_a) for i in range(6)]
            p_1 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i>j)
            p_x = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i==j)
            p_2 = sum(h_probs[i]*a_probs[j] for i in range(6) for j in range(6) if i<j)
            table[h]['xPts'] += (p_1*3 + p_x*1)
            table[a]['xPts'] += (p_2*3 + p_x*1)
            
    df_table = pd.DataFrame.from_dict(table, orient='index').reset_index()
    df_table.columns = ['Team', 'Pts', 'xPts']
    df_table['Diff'] = df_table['Pts'] - df_table['xPts']
    df_table['xPts'] = df_table['xPts'].round(1); df_table['Diff'] = df_table['Diff'].round(1)
    return df_table.sort_values(by='xPts', ascending=False)

# --- 8. CACHE ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_cached(cutoff):
    try:
        conn = sqlite3.connect("mintstats.db"); df = pd.read_sql("SELECT * FROM all_leagues", conn); conn.close()
        df['Date'] = pd.to_datetime(df['Date']); return df[df['Date'] >= cutoff]
    except: return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def get_model(hash_key, df): return PoissonModel(df) if not df.empty else None

# --- 9. UI ---
if 'fixture_pool' not in st.session_state: st.session_state.fixture_pool = load_fixture_pool()
if 'generated_coupons' not in st.session_state: st.session_state.generated_coupons = []
if 'last_ocr_debug' not in st.session_state: st.session_state.last_ocr_debug = None

st.title("☁️ MintStats v28.3: Complete Edition")
st.sidebar.header("Panel Sterowania")
mode = st.sidebar.radio("Moduł:", ["1. 🛠️ ADMIN", "2. 🚀 GENERATOR", "3. 📜 MOJE KUPONY", "4. 🧪 LABORATORIUM"])

cutoff = pd.to_datetime('today') - pd.DateOffset(years=1)

# --- SIDEBAR BACKUP (RESTORED) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Kopia Zapasowa")
if st.sidebar.button("📦 Pobierz Terminarz"):
    save_fixture_pool(st.session_state.fixture_pool)
    if os.path.exists(FIXTURES_DB_FILE):
        with open(FIXTURES_DB_FILE, "rb") as f:
            st.sidebar.download_button("⬇️ Zapisz plik", f, "terminarz.csv")

uploaded_fix = st.sidebar.file_uploader("Wgraj Terminarz", type=['csv'])
if uploaded_fix and st.sidebar.button("♻️ Przywróć"):
    df = pd.read_csv(uploaded_fix)
    st.session_state.fixture_pool = df.to_dict('records')
    save_fixture_pool(st.session_state.fixture_pool)
    st.rerun()

if mode == "1. 🛠️ ADMIN":
    st.subheader("Baza Danych")
    if st.button("Aktualizuj Online"):
        with st.spinner("Pobieranie..."):
            c, t = download_and_update_db(LEAGUE_NAMES); st.success(f"Pobrano {t} meczów.")
            st.cache_data.clear(); st.cache_resource.clear()
            
    st.divider()
    uploaded = st.file_uploader("Wgraj pliki CSV", accept_multiple_files=True)
    if uploaded and st.button("Wgraj"):
        with st.spinner("Przetwarzanie..."):
            c, l, _ = process_uploaded_history(uploaded); st.success(f"Dodano {c} meczów.")
            st.cache_data.clear(); st.cache_resource.clear()

    if "TU_WKLEJ" in JSONBIN_API_KEY: st.error("⚠️ Brak kluczy API!")
    else: st.success("✅ Połączono z chmurą (JSONBin).")
    
    st.divider()
    st.markdown("### 🔍 Status Bazy Danych")
    db_stat = get_db_status()
    if not db_stat.empty: st.dataframe(db_stat, use_container_width=True)
    else: st.info("Baza pusta.")

elif mode == "2. 🚀 GENERATOR":
    df_all = load_data_cached(cutoff)
    if df_all.empty: st.warning("Brak danych."); st.stop()
    model = get_model(str(df_all.shape), df_all)
    all_teams_list = pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).unique()
    
    # --- RESTORED TABS FOR INPUT ---
    with st.expander("Dodaj Mecze", expanded=True):
        t1, t2, t3, t4 = st.tabs(["Ręczny", "📸 Zdjęcie", "📝 Tekst", "📁 CSV"])
        
        with t1:
            leagues = get_leagues_list()
            sel_league = st.selectbox("Liga:", leagues)
            df_l = df_all[df_all['LeagueName'] == sel_league]
            if not df_l.empty:
                teams = sorted(pd.concat([df_l['HomeTeam'], df_l['AwayTeam']]).unique())
                c_d, c_h, c_a = st.columns([1,2,2])
                d_in = c_d.date_input("Data", datetime.today())
                h_in = c_h.selectbox("Dom", teams)
                a_in = c_a.selectbox("Wyjazd", teams)
                if st.button("➕ Dodaj Mecz"):
                    if h_in!=a_in: 
                        st.session_state.fixture_pool.append({'Home':h_in, 'Away':a_in, 'League':sel_league, 'Date':str(d_in)})
                        save_fixture_pool(st.session_state.fixture_pool)
                        st.rerun()

        with t2:
            uploaded_img = st.file_uploader("Screen Flashscore", type=['png', 'jpg', 'jpeg'])
            if uploaded_img and st.button("Skanuj"):
                txt = extract_text_from_image(uploaded_img)
                m_list, dbg, _ = smart_parse_matches_v3(txt, all_teams_list)
                if m_list:
                    st.session_state.fixture_pool.extend(m_list)
                    save_fixture_pool(st.session_state.fixture_pool)
                    st.success(f"Dodano {len(m_list)} meczów")
                    st.rerun()

        with t3:
            raw = st.text_area("Wklej mecze z Flashscore:")
            if st.button("Analizuj Tekst"):
                parsed = parse_raw_text(raw, all_teams_list)
                if parsed:
                    for p in parsed: st.session_state.fixture_pool.append(p)
                    save_fixture_pool(st.session_state.fixture_pool); st.rerun()
        
        with t4:
            up_csv = st.file_uploader("fixtures.csv", type=['csv'])
            if up_csv and st.button("Import CSV"):
                m, e = parse_fixtures_csv(up_csv)
                if m: 
                    st.session_state.fixture_pool.extend(m)
                    save_fixture_pool(st.session_state.fixture_pool); st.rerun()

    if st.session_state.fixture_pool:
        st.dataframe(pd.DataFrame(st.session_state.fixture_pool), use_container_width=True)
        if st.button("🗑️ Wyczyść"): st.session_state.fixture_pool = []; save_fixture_pool([]); st.rerun()
        
        st.divider()
        st.subheader("Generowanie")
        
        strat_mode = st.radio("Tryb:", ["Pojedyncza", "Mix"], horizontal=True)
        gen_settings = {}
        
        ALL_STRATS = [
            "Mix Bezpieczny", "Podwójna Szansa", "Gole Agresywne", "Do Przerwy", "Twierdza",
            "Mur Obronny", "Złoty Środek", "Obie strzelą (TAK)", "Obie strzelą (NIE)",
            "1 drużyna strzeli (TAK)", "1 drużyna strzeli (NIE)", "2 drużyna strzeli (TAK)",
            "2 drużyna strzeli (NIE)", "Handicap: Dominacja Faworyta", "Handicap: Tarcza Underdoga",
            "DNB: Gospodarz", "DNB: Gość"
        ]

        if strat_mode == "Pojedyncza":
            gen_settings['strat'] = st.selectbox("Strategia", ALL_STRATS)
            gen_settings['count'] = st.slider("Ile meczów?", 1, 10, 5)
        else:
            # --- FULL MIX UI ---
            st.info("Wybierz ile meczów z każdej strategii chcesz:")
            c1, c2, c3 = st.columns(3)
            gen_settings['mix'] = {}
            with c1:
                st.markdown("**Bezpieczne**")
                gen_settings['mix']["Twierdza"] = st.number_input("Twierdza (1)", 0, 5, 0)
                gen_settings['mix']["Podwójna Szansa"] = st.number_input("Podpórki (1X/X2)", 0, 5, 0)
                gen_settings['mix']["DNB: Gospodarz"] = st.number_input("DNB 1", 0, 5, 0)
                gen_settings['mix']["DNB: Gość"] = st.number_input("DNB 2", 0, 5, 0)
            with c2:
                st.markdown("**Bramki**")
                gen_settings['mix']["Złoty Środek"] = st.number_input("Over 1.5", 0, 5, 0)
                gen_settings['mix']["Gole Agresywne"] = st.number_input("Over 2.5 / BTS", 0, 5, 0)
                gen_settings['mix']["Mur Obronny"] = st.number_input("Under 2.5/3.5", 0, 5, 0)
                gen_settings['mix']["Obie strzelą (TAK)"] = st.number_input("BTS TAK", 0, 5, 0)
            with c3:
                st.markdown("**Ryzyko**")
                gen_settings['mix']["Handicap: Dominacja Faworyta"] = st.number_input("Handicap -1.5", 0, 5, 0)
                gen_settings['mix']["Do Przerwy"] = st.number_input("1. Połowa Gole", 0, 5, 0)
                gen_settings['mix']["1 drużyna strzeli (TAK)"] = st.number_input("Gosp. Strzeli", 0, 5, 0)
                gen_settings['mix']["2 drużyna strzeli (TAK)"] = st.number_input("Gość Strzeli", 0, 5, 0)

        if st.button("🚀 GENERUJ (Automatyczny zapis użycia)", type="primary"):
            cloud_blacklist = get_cloud_blacklist()
            gen = CouponGenerator(model)
            final_coupon = []
            
            if strat_mode == "Pojedyncza":
                candidates = gen.analyze_pool(st.session_state.fixture_pool, gen_settings['strat'], cloud_blacklist)
                candidates.sort(key=lambda x: x['Pewność'], reverse=True)
                final_coupon = candidates[:gen_settings['count']]
            else:
                # MIX LOGIC
                for s, cnt in gen_settings['mix'].items():
                    if cnt > 0:
                        cand = gen.analyze_pool(st.session_state.fixture_pool, s, cloud_blacklist)
                        cand.sort(key=lambda x: x['Pewność'], reverse=True)
                        for c in cand:
                            # Avoid dups on coupon
                            if c['Mecz'] not in [x['Mecz'] for x in final_coupon]:
                                final_coupon.append(c)
                                # Count only matches for THIS specific strategy in the mix
                                current_strat_count = len([x for x in final_coupon if x['Typ'] == c['Typ']])
                                # Simple check won't work perfectly for mixed types, but good enough for now
                                if len(final_coupon) >= sum(gen_settings['mix'].values()): break 
            
            if final_coupon:
                st.session_state.generated_coupons = [{'name': "Auto-Kupon", 'data': final_coupon}]
                matches_to_block = [x['Mecz'] for x in final_coupon]
                update_cloud_blacklist(matches_to_block)
                st.toast(f"☁️ Zapisano {len(matches_to_block)} meczów w chmurze.")
            else: st.warning("Brak typów (lub wszystkie mecze już wykorzystane).")

        if st.session_state.generated_coupons:
            for c in st.session_state.generated_coupons:
                st.write("---")
                df_res = pd.DataFrame(c['data'])
                if not df_res.empty:
                    st.dataframe(df_res[['Mecz', 'Typ', 'Pewność', 'xG']].style.format({'Pewność': '{:.1%}'}), use_container_width=True)
                    with st.expander("Szczegóły"):
                        for _, row in df_res.iterrows():
                            st.write(f"**{row['Mecz']}**")
                            c1, c2 = st.columns(2)
                            c1.plotly_chart(create_radar_chart(row['HomeStats'], row['AwayStats'], "Dom", "Wyjazd"), use_container_width=True)
                            c2.plotly_chart(create_goal_distribution(float(row['xG'].split(':')[0]), float(row['xG'].split(':')[1]), "Dom", "Wyjazd"), use_container_width=True)

elif mode == "3. 📜 MOJE KUPONY":
    st.info("Historia lokalna (nie z chmury).")
    coupons = load_saved_coupons()
    for c in reversed(coupons):
        with st.expander(c['name']):
            st.dataframe(pd.DataFrame(c['data']))

elif mode == "4. 🧪 LABORATORIUM":
    st.title("🧪 Laboratorium Analityczne")
    tab1, tab2 = st.tabs(["🔙 Backtest", "⚖️ xPoints"])
    
    with tab1:
        st.subheader("Test skuteczności")
        leagues = get_leagues_list()
        if not leagues: st.error("Brak danych."); st.stop()
        sel_lg = st.selectbox("Liga:", leagues)
        strat = st.selectbox("Strategia:", ["Mix Bezpieczny", "Gole Agresywne", "Twierdza", "Mur Obronny"])
        limit = st.slider("Mecze:", 20, 200, 50)
        
        if st.button("Uruchom"):
            df = get_data_for_league(sel_lg)
            df = df[df['Date'] >= cutoff]
            if df.empty: st.error("Brak danych.")
            else:
                with st.spinner("Liczenie..."):
                    res = run_backtest(df, strat, limit)
                    col1, col2 = st.columns(2)
                    acc = (res['Correct']/res['Total'])*100 if res['Total']>0 else 0
                    col1.metric("Skuteczność", f"{acc:.1f}%")
                    col2.metric("Trafione", f"{res['Correct']}/{res['Total']}")
    
    with tab2:
        st.subheader("Tabela Sprawiedliwości")
        leagues = get_leagues_list()
        sel_xp_lg = st.selectbox("Liga:", leagues, key="xp")
        if sel_xp_lg:
            df = get_data_for_league(sel_xp_lg)
            df = df[df['Date'] >= cutoff]
            if not df.empty:
                st.dataframe(calculate_xpts_table(df).style.format({'xPts':'{:.1f}'}), use_container_width=True)
