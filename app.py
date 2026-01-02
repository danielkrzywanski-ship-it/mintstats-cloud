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

# --- KONFIGURACJA ---
st.set_page_config(page_title="MintStats v13.10 Ultimate Dict", layout="wide")
FIXTURES_DB_FILE = "my_fixtures.csv"

# --- SŁOWNIK ALIASÓW (TWOJE AKTUALIZACJE) ---
TEAM_ALIASES = {
    # --- PORTUGALIA ---
    "avs": "AFS", "avs futebol": "AFS", "afs": "AFS", "a v s": "AFS", "b ars": "AFS",
    "brag": "Sp Braga", "braga": "Sp Braga", "sc braga": "Sp Braga", "sp braga": "Sp Braga", "w braga": "Sp Braga",
    "bars": "Boavista", "boavista": "Boavista", "w tondela": "Tondela",
    "sporting": "Sp Lisbon", "sporting cp": "Sp Lisbon", "sp lisbon": "Sp Lisbon",
    "vitoria guimaraes": "Guimaraes", "v guimaraes": "Guimaraes", "guimaraes": "Guimaraes",
    "fc porto": "Porto", "porto": "Porto", "fc porte": "Porto",
    "rio ave": "Rio Ave", "b biosve": "Rio Ave",
    "estoril": "Estoril", "casa pia": "Casa Pia", "gil vicente": "Gil Vicente",
    "farense": "Farense", "famalicao": "Famalicao", "arouca": "Arouca", "moreirense": "Moreirense",
    "estrela": "Estrela", "benfica": "Benfica", "santa clara": "Santa Clara", "nacional": "Nacional",
    
    # --- ANGLIA ---
    "manchester uta": "Man United", "man uta": "Man United",
    "man utd": "Man United", "manchester utd": "Man United", "man united": "Man United",
    "hull": "Hull", "hull city": "Hull", "uit": "Hull",
    "watford": "Watford", "watford fc": "Watford", "wottora": "Watford",
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
    "sheffield wed": "Sheffield Weds", "sheffield wednesday": "Sheffield Weds", "shetticia wea": "Sheffield Weds",
    "plymouth": "Plymouth", "plymouth argyle": "Plymouth", 
    "portsmouth": "Portsmouth",
    "nottm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
    "wolves": "Wolverhampton", "wolverhampton": "Wolverhampton",
    "sheff utd": "Sheffield United", "sheffield united": "Sheffield United", "shettiots urs": "Sheffield United", "shottiois urs": "Sheffield United",
    "leeds": "Leeds", "leeds utd": "Leeds", 
    "manchester city": "Man City", "man city": "Man City",
    "wrexham": "Wrexham",
    "br newer": "Ipswich", "ipswich": "Ipswich",

    # --- HISZPANIA ---
    "valiadolia": "Valladolid", "valladolid": "Valladolid", 
    "burgos cr": "Burgos", "burgos": "Burgos",
    "castetion": "Castellon", "castellon": "Castellon", 
    "racing santander": "Santander", "r santander": "Santander",
    "cultural leonesa": "Cultural Leonesa", "leonesa": "Cultural Leonesa",
    "real sociedad b": "Sociedad B", "sociedad b": "Sociedad B",
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
    "parc": "Pau FC", "pau": "Pau FC", "pau fc": "Pau FC"
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

# --- FUNKCJE POMOCNICZE I BAZODANOWE ---

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
    # 1. Alias
    for alias, db_name in TEAM_ALIASES.items():
        if alias == cur: return db_name
        if len(alias) > 3 and alias in cur: return db_name
    # 2. Fuzzy
    match = difflib.get_close_matches(cur, [t.lower() for t in available_teams], n=1, cutoff=0.7)
    if match:
        for real_name in available_teams:
            if real_name.lower() == match[0]: return real_name
    return None

def parse_raw_text(text_input, available_teams):
    lines = text_input.split('\n')
    found_matches = []
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
                found_matches.append({'Home': home_team, 'Away': away_team, 'League': 'Text Import'})
    return found_matches

def smart_parse_matches_v3(text_input, available_teams):
    cleaned_lines = clean_ocr_text_debug(text_input)
    found_teams = []
    debug_log = []
    for line in cleaned_lines:
        cur = line.lower().strip()
        matched = resolve_team_name(cur, available_teams)
        if matched:
            if not found_teams or found_teams[-1] != matched: found_teams.append(matched)
            debug_log.append(f"✅ '{cur}' -> '{matched}'")
        else:
            debug_log.append(f"❌ '{cur}'")
    matches = [{'Home': found_teams[i], 'Away': found_teams[i+1], 'League': 'OCR Import'} for i in range(0, len(found_teams) - 1, 2)]
    return matches, debug_log, cleaned_lines

def parse_fixtures_csv(file):
    try:
        df = pd.read_csv(file)
        if not {'Div', 'HomeTeam', 'AwayTeam'}.issubset(df.columns): return [], "Brak kolumn Div/HomeTeam/AwayTeam"
        matches = []
        for _, row in df.iterrows():
            matches.append({'Home': row['HomeTeam'], 'Away': row['AwayTeam'], 'League': row['Div']})
        return matches, None
    except Exception as e: return [], str(e)

# --- MODEL POISSONA ---
class PoissonModel:
    def __init__(self, data):
        self.data = data
        self.team_stats_ft = {}
        self.team_stats_ht = {}
        self.league_avg_ft = 1.0
        self.league_avg_ht = 1.0
        if not data.empty: self._calculate_strength()

    def _calculate_strength(self):
        lg_ft = self.data['FTHG'].sum() + self.data['FTAG'].sum()
        matches = len(self.data) * 2
        self.league_avg_ft = lg_ft / matches if matches > 0 else 1.0

        has_ht = 'HTHG' in self.data.columns and 'HTAG' in self.data.columns
        if has_ht:
            lg_ht = self.data['HTHG'].sum() + self.data['HTAG'].sum()
            self.league_avg_ht = lg_ht / matches if matches > 0 else 1.0
        
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        for team in teams:
            home = self.data[self.data['HomeTeam'] == team]
            away = self.data[self.data['AwayTeam'] == team]
            scored_ft = home['FTHG'].sum() + away['FTAG'].sum()
            conceded_ft = home['FTAG'].sum() + away['FTHG'].sum()
            played = len(home) + len(away)
            if played > 0:
                self.team_stats_ft[team] = {
                    'attack': (scored_ft / played) / self.league_avg_ft,
                    'defense': (conceded_ft / played) / self.league_avg_ft
                }
                if has_ht:
                    scored_ht = home['HTHG'].sum() + away['HTAG'].sum()
                    conceded_ht_real = home['HTAG'].sum() + away['HTHG'].sum()
                    self.team_stats_ht[team] = {
                        'attack': (scored_ht / played) / self.league_avg_ht,
                        'defense': (conceded_ht_real / played) / self.league_avg_ht
                    }

    def predict(self, home, away):
        if home not in self.team_stats_ft or away not in self.team_stats_ft: return None, None, None, None
        xg_h_ft = self.team_stats_ft[home]['attack'] * self.team_stats_ft[away]['defense'] * self.league_avg_ft * 1.15
        xg_a_ft = self.team_stats_ft[away]['attack'] * self.team_stats_ft[home]['defense'] * self.league_avg_ft
        xg_h_ht, xg_a_ht = 0.0, 0.0
        if home in self.team_stats_ht and away in self.team_stats_ht:
            xg_h_ht = self.team_stats_ht[home]['attack'] * self.team_stats_ht[away]['defense'] * self.league_avg_ht * 1.10
            xg_a_ht = self.team_stats_ht[away]['attack'] * self.team_stats_ht[home]['defense'] * self.league_avg_ht
        return xg_h_ft, xg_a_ft, xg_h_ht, xg_a_ht

    def calculate_probs(self, xg_h_ft, xg_a_ft, xg_h_ht, xg_a_ht):
        max_goals = 8
        mat_ft = np.array([[poisson.pmf(i, xg_h_ft) * poisson.pmf(j, xg_a_ft) for j in range(max_goals)] for i in range(max_goals)])
        mat_ht = np.array([[poisson.pmf(i, xg_h_ht) * poisson.pmf(j, xg_a_ht) for j in range(max_goals)] for i in range(max_goals)])
        
        prob_1 = np.sum(np.tril(mat_ft, -1)); prob_x = np.sum(np.diag(mat_ft)); prob_2 = np.sum(np.triu(mat_ft, 1))
        bts_yes = np.sum(mat_ft[1:, 1:])
        return {
            "1": prob_1, "X": prob_x, "2": prob_2, "1X": prob_1+prob_x, "X2": prob_x+prob_2,
            "BTS_Yes": bts_yes, "BTS_No": 1.0-bts_yes,
            "Over_1.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Over_2.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 2.5]),
            "Under_3.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 3.5]),
            "Under_4.5_FT": np.sum([mat_ft[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 4.5]),
            "Over_1.5_HT": np.sum([mat_ht[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5])
        }

class CouponGenerator:
    def __init__(self, model): self.model = model
    def analyze_pool(self, pool, strategy="Mieszany"):
        res = []
        for m in pool:
            xg_h, xg_a, xg_h_ht, xg_a_ht = self.model.predict(m['Home'], m['Away'])
            if xg_h is None: continue
            probs = self.model.calculate_probs(xg_h, xg_a, xg_h_ht, xg_a_ht)
            sel_name, sel_prob = None, 0.0
            
            if strategy == "1 (Gospodarz)": sel_name, sel_prob = f"Wygrana {m['Home']}", probs['1']
            elif strategy == "2 (Gość)": sel_name, sel_prob = f"Wygrana {m['Away']}", probs['2']
            elif strategy == "Over 2.5": sel_name, sel_prob = "Over 2.5", probs['Over_2.5_FT']
            elif strategy == "Under 4.5": sel_name, sel_prob = "Under 4.5", probs['Under_4.5_FT']
            elif strategy == "1. Połowa Over 1.5": sel_name, sel_prob = "HT Over 1.5", probs['Over_1.5_HT']
            elif strategy == "BTS Tak": sel_name, sel_prob = "BTS Tak", probs['BTS_Yes']
            elif strategy == "1X": sel_name, sel_prob = "1X", probs['1X']
            elif strategy == "X2": sel_name, sel_prob = "X2", probs['X2']
            else:
                opts = [('1', f"Win {m['Home']}", probs['1']), ('2', f"Win {m['Away']}", probs['2']),
                        ('O2.5', "Over 2.5", probs['Over_2.5_FT']), ('U4.5', "Under 4.5", probs['Under_4.5_FT']),
                        ('HT1.5', "HT Over 1.5", probs['Over_1.5_HT']), ('BTS', "BTS", probs['BTS_Yes']), 
                        ('1X', "1X", probs['1X']), ('X2', "X2", probs['X2'])]
                best = sorted(opts, key=lambda x: x[2], reverse=True)[0]
                _, sel_name, sel_prob = best
            res.append({'Mecz': f"{m['Home']} - {m['Away']}", 'Liga': m.get('League', 'N/A'), 'Typ': sel_name, 'Pewność': sel_prob, 'xG': f"{xg_h:.2f}:{xg_a:.2f}"})
        return res

# --- INIT ---
if 'fixture_pool' not in st.session_state: st.session_state.fixture_pool = load_fixture_pool()
if 'generated_coupons' not in st.session_state: st.session_state.generated_coupons = [] 
if 'last_ocr_debug' not in st.session_state: st.session_state.last_ocr_debug = None

# --- INTERFEJS ---
st.title("☁️ MintStats v13.10: Ultimate Dict")

st.sidebar.header("Panel Sterowania")
mode = st.sidebar.radio("Wybierz moduł:", ["1. 🛠️ ADMIN (Baza Danych)", "2. 🚀 GENERATOR KUPONÓW"])

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
    
    # --- DODAWANIE MECZÓW ---
    st.sidebar.markdown("---")
    st.sidebar.header("Dodaj Mecze")
    tab_manual, tab_ocr, tab_text, tab_csv = st.sidebar.tabs(["Ręczny", "📸 Zdjęcie", "📝 Wklej Tekst", "📁 CSV"])
    
    new_items = []
    
    # 1. RĘCZNY
    with tab_manual:
        sel_league = st.selectbox("Liga:", leagues)
        df_l = get_data_for_league(sel_league)
        teams = sorted(pd.concat([df_l['HomeTeam'], df_l['AwayTeam']]).unique())
        with st.form("manual_add"):
            h = st.selectbox("Dom", teams); a = st.selectbox("Wyjazd", teams)
            if st.form_submit_button("➕ Dodaj") and h!=a: new_items.append({'Home':h, 'Away':a, 'League':sel_league})
    
    # 2. OCR
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

    # 3. TEXT PARSER
    with tab_text:
        st.info("💡 Skopiuj listę meczów z Flashscore (Ctrl+C) i wklej tutaj.")
        raw_text_input = st.text_area("Wklej mecze (np. 'Braga - Benfica')", height=150)
        if st.button("🔍 Analizuj Tekst"):
            parsed = parse_raw_text(raw_text_input, all_teams_list)
            if parsed:
                new_items.extend(parsed)
                st.success(f"✅ Znaleziono {len(parsed)} par!")
                for p in parsed: st.caption(f"{p['Home']} vs {p['Away']}")
            else:
                st.error("Nie udało się rozpoznać par. Format: 'Drużyna A - Drużyna B'.")

    # 4. CSV
    with tab_csv:
        uploaded_fix = st.file_uploader("fixtures.csv", type=['csv'])
        if uploaded_fix and st.button("📥 Import"):
            m_list, err = parse_fixtures_csv(uploaded_fix)
            if not err: new_items.extend(m_list); st.success(f"Import {len(m_list)}")
            else: st.error(err)

    if new_items:
        for item in new_items:
            if not any(x['Home']==item['Home'] and x['Away']==item['Away'] for x in st.session_state.fixture_pool):
                st.session_state.fixture_pool.append(item)
        save_fixture_pool(st.session_state.fixture_pool)
        st.rerun()

    # --- EDYTOR I GENERATOR ---
    st.subheader("📋 Terminarz")
    if st.session_state.fixture_pool:
        df_pool = pd.DataFrame(st.session_state.fixture_pool)
        edited_df = st.data_editor(df_pool, num_rows="dynamic", use_container_width=True, key="fixture_editor")
        if edited_df.to_dict('records') != st.session_state.fixture_pool:
            st.session_state.fixture_pool = edited_df.to_dict('records')
            save_fixture_pool(st.session_state.fixture_pool)
        if st.button("🗑️ Wyczyść WSZYSTKO"): st.session_state.fixture_pool = []; save_fixture_pool([]); st.rerun()

        st.divider()
        st.header("🎲 Generator Kuponów")
        c1, c2, c3 = st.columns(3)
        with c1: gen_mode = st.radio("Tryb:", ["Jeden Pewny Kupon", "System Rozpisowy"])
        with c2: strat = st.selectbox("Strategia", ["Mieszany", "Over 2.5", "Under 4.5", "1. Połowa Over 1.5", "1", "2", "1X", "X2", "BTS Tak"])
        with c3:
            if gen_mode == "Jeden Pewny Kupon": coupon_len = st.number_input("Długość", 1, 50, 12)
            else:
                num_coupons = st.number_input("Ile kuponów?", 1, 10, 3)
                events_per_coupon = st.number_input("Mecze na kupon?", 1, 20, 5)
                chaos_factor = st.slider("Pula (Top X)", 10, 100, 30)

        if st.button("🚀 GENERUJ", type="primary"):
            analyzed_pool = gen.analyze_pool(st.session_state.fixture_pool, strat)
            analyzed_pool = sorted(analyzed_pool, key=lambda x: x['Pewność'], reverse=True)
            st.session_state.generated_coupons = [] 
            if gen_mode == "Jeden Pewny Kupon":
                st.session_state.generated_coupons.append({"name": "Top Pewniaki", "data": analyzed_pool[:coupon_len]})
            else: 
                candidate_pool = analyzed_pool[:chaos_factor]
                if len(candidate_pool) < events_per_coupon: st.error("Za mało meczów w puli!")
                else:
                    for i in range(num_coupons):
                        random_selection = random.sample(candidate_pool, min(len(candidate_pool), events_per_coupon))
                        st.session_state.generated_coupons.append({"name": f"Kupon Losowy #{i+1}", "data": random_selection})

        if st.session_state.generated_coupons:
            st.write("---")
            for kupon in st.session_state.generated_coupons:
                with st.container():
                    st.subheader(f"🎫 {kupon['name']} ({strat})")
                    df_k = pd.DataFrame(kupon['data'])
                    if not df_k.empty:
                        st.dataframe(df_k.style.background_gradient(subset=['Pewność'], cmap="RdYlGn", vmin=0.4, vmax=0.9).format({'Pewność':'{:.1%}'}), use_container_width=True)
                        st.caption(f"Średnia pewność: {df_k['Pewność'].mean()*100:.1f}%")
                    else: st.warning("Brak typów.")
                    st.write("---")
    else: st.info("Pula pusta.")
