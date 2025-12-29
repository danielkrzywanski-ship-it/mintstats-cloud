import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import difflib
from scipy.stats import poisson
from PIL import Image
import pytesseract
import re
import io
import os

# --- KONFIGURACJA ---
st.set_page_config(page_title="MintStats v12.3 Aliases Fix", layout="wide")
FIXTURES_DB_FILE = "my_fixtures.csv"

# --- SŁOWNIK ALIASÓW (TŁUMACZ FLASHSCORE -> BAZA DANYCH) ---
TEAM_ALIASES = {
    # --- ANGLIA (Championship / Premier League) ---
    "Hull City": "Hull",
    "Hull": "Hull", # Na wypadek gdyby OCR czytał krótko
    "Watford": "Watford", # Czasem OCR myli 'Watford' z czymś innym
    "Watford FC": "Watford",
    "QPR": "QPR", 
    "Queens Park Rangers": "QPR",
    "West Brom": "West Brom",
    "West Bromwich": "West Brom",
    "Blackburn Rovers": "Blackburn",
    "Preston North End": "Preston",
    "Preston": "Preston",
    "Coventry City": "Coventry",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Norwich City": "Norwich",
    "Luton Town": "Luton",
    "Derby County": "Derby",
    "Oxford United": "Oxford",
    "Sheffield Wed": "Sheffield Weds",
    "Sheffield Wednesday": "Sheffield Weds",
    "Plymouth Argyle": "Plymouth",
    "Portsmouth": "Portsmouth",
    
    # --- PORTUGALIA ---
    "Sporting": "Sp Lisbon", "Sporting CP": "Sp Lisbon",
    "Vitoria Guimaraes": "Guimaraes", "V. Guimaraes": "Guimaraes",
    "FC Porto": "Porto", "Rio Ave": "Rio Ave",
    
    # --- ANGLIA (GŁÓWNE) ---
    "Man Utd": "Man United", "Manchester Utd": "Man United",
    "Nottm Forest": "Nott'm Forest", "Wolves": "Wolverhampton",
    "Sheff Utd": "Sheffield United", "Leeds Utd": "Leeds",
    
    # --- HISZPANIA ---
    "Athletic Bilbao": "Ath Bilbao", "Atl. Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid", "Betis": "Real Betis",
    
    # --- WŁOCHY ---
    "Inter": "Inter Milan", "AC Milan": "Milan",
    
    # --- NIEMCY ---
    "B. Monchengladbach": "M'gladbach", "Monchengladbach": "M'gladbach",
    "Mainz": "Mainz 05", "Frankfurt": "Ein Frankfurt",
    
    # --- POLSKA ---
    "Legia Warszawa": "Legia Warsaw", "Śląsk Wrocław": "Slask Wroclaw",
    "Lech Poznań": "Lech Poznan", "Górnik Zabrze": "Gornik Zabrze",
    "Jagiellonia Białystok": "Jagiellonia", "Pogoń Szczecin": "Pogon Szczecin",
    "Cracovia Kraków": "Cracovia", "Raków Częstochowa": "Rakow Czestochowa"
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

# --- FUNKCJE BAZODANOWE ---
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
        query = "SELECT * FROM all_leagues WHERE LeagueName = ?"
        df = pd.read_sql(query, conn, params=(league_name,))
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

# --- ZARZĄDZANIE TERMINARZEM (PERSISTENCE) ---
def load_fixture_pool():
    if os.path.exists(FIXTURES_DB_FILE):
        try: return pd.read_csv(FIXTURES_DB_FILE).to_dict('records')
        except: return []
    return []

def save_fixture_pool(pool_data):
    if pool_data: pd.DataFrame(pool_data).to_csv(FIXTURES_DB_FILE, index=False)
    else:
        if os.path.exists(FIXTURES_DB_FILE): os.remove(FIXTURES_DB_FILE)

# --- MODEL POISSONA ---
class PoissonModel:
    def __init__(self, data):
        self.data = data
        self.team_stats = {}
        self.league_avg = 1.0
        if not data.empty: self._calculate_strength()

    def _calculate_strength(self):
        league_goals = self.data['FTHG'].sum() + self.data['FTAG'].sum()
        matches = len(self.data) * 2
        self.league_avg = league_goals / matches if matches > 0 else 1.0
        teams = pd.concat([self.data['HomeTeam'], self.data['AwayTeam']]).unique()
        stats = {}
        for team in teams:
            home = self.data[self.data['HomeTeam'] == team]
            away = self.data[self.data['AwayTeam'] == team]
            scored = home['FTHG'].sum() + away['FTAG'].sum()
            conceded = home['FTAG'].sum() + away['FTHG'].sum()
            played = len(home) + len(away)
            if played > 0:
                stats[team] = {'attack': (scored / played) / self.league_avg, 'defense': (conceded / played) / self.league_avg}
        self.team_stats = stats

    def predict(self, home, away):
        if home not in self.team_stats or away not in self.team_stats: return None, None
        xg_h = self.team_stats[home]['attack'] * self.team_stats[away]['defense'] * self.league_avg * 1.15
        xg_a = self.team_stats[away]['attack'] * self.team_stats[home]['defense'] * self.league_avg
        return xg_h, xg_a

    def calculate_probs(self, xg_h, xg_a):
        max_goals = 8
        matrix = np.array([[poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a) for j in range(max_goals)] for i in range(max_goals)])
        prob_1 = np.sum(np.tril(matrix, -1)); prob_x = np.sum(np.diag(matrix)); prob_2 = np.sum(np.triu(matrix, 1))
        bts_yes = np.sum(matrix[1:, 1:])
        prob_home_0 = poisson.pmf(0, xg_h); prob_away_0 = poisson.pmf(0, xg_a)
        return {
            "1": prob_1, "X": prob_x, "2": prob_2, "1X": prob_1+prob_x, "X2": prob_x+prob_2, "12": prob_1+prob_2,
            "BTS_Yes": bts_yes, "BTS_No": 1.0-bts_yes,
            "Over_1.5": np.sum([matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 1.5]),
            "Over_2.5": np.sum([matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i+j > 2.5]),
            "Under_3.5": np.sum([matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i+j <= 3.5]),
            "Home_Score_Yes": 1.0-prob_home_0, "Home_Score_No": prob_home_0, "Away_Score_Yes": 1.0-prob_away_0, "Away_Score_No": prob_away_0
        }

class CouponGenerator:
    def __init__(self, model): self.model = model
    def analyze_pool(self, pool, strategy="Mieszany"):
        res = []
        for m in pool:
            xg_h, xg_a = self.model.predict(m['Home'], m['Away'])
            if xg_h is None: continue
            probs = self.model.calculate_probs(xg_h, xg_a)
            sel_name, sel_prob = None, 0.0
            if strategy == "1 (Gospodarz)": sel_name, sel_prob = f"Wygrana {m['Home']}", probs['1']
            elif strategy == "2 (Gość)": sel_name, sel_prob = f"Wygrana {m['Away']}", probs['2']
            elif strategy == "Over 2.5": sel_name, sel_prob = "Over 2.5", probs['Over_2.5']
            elif strategy == "BTS Tak": sel_name, sel_prob = "BTS Tak", probs['BTS_Yes']
            elif strategy == "1X": sel_name, sel_prob = "1X", probs['1X']
            elif strategy == "X2": sel_name, sel_prob = "X2", probs['X2']
            else:
                opts = [('1', f"Win {m['Home']}", probs['1']), ('2', f"Win {m['Away']}", probs['2']),
                        ('O2.5', "Over 2.5", probs['Over_2.5']), ('BTS', "BTS", probs['BTS_Yes']), ('1X', "1X", probs['1X']), ('X2', "X2", probs['X2'])]
                best = sorted(opts, key=lambda x: x[2], reverse=True)[0]
                _, sel_name, sel_prob = best
            res.append({'Mecz': f"{m['Home']} - {m['Away']}", 'Liga': m.get('League', 'N/A'), 'Typ': sel_name, 'Pewność': sel_prob, 'xG': f"{xg_h:.2f}:{xg_a:.2f}"})
        return res

# --- NARZĘDZIA POMOCNICZE ---
def clean_ocr_text(text):
    return [re.sub(r'[^a-zA-Z \-]', '', line).strip() for line in text.split('\n') if len(re.sub(r'[^a-zA-Z \-]', '', line).strip()) > 3]

def extract_text_from_image(uploaded_file):
    try: 
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image, lang='eng', config='--psm 6')
    except Exception as e: return f"Error OCR: {e}"

def smart_parse_matches_v2(text_input, available_teams):
    cleaned_lines = clean_ocr_text(text_input); found_teams = []
    for line in cleaned_lines:
        cur = line.strip(); matched = None
        # 1. Aliasy (Najpierw dokładne, potem zawierające)
        for alias, db_name in TEAM_ALIASES.items():
            if alias.lower() == cur.lower() or (len(alias) > 4 and alias.lower() in cur.lower()):
                 if db_name in available_teams: matched = db_name; break
        
        # 2. Fuzzy Match
        if not matched:
            match = difflib.get_close_matches(cur, available_teams, n=1, cutoff=0.6)
            if match: matched = match[0]

        if matched:
            if not found_teams or found_teams[-1] != matched: found_teams.append(matched)
            
    return [{'Home': found_teams[i], 'Away': found_teams[i+1], 'League': 'OCR Import', 'Original': f"{found_teams[i]} vs {found_teams[i+1]}"} for i in range(0, len(found_teams) - 1, 2)], found_teams

def process_uploaded_history(files):
    all_data = []
    for uploaded_file in files:
        try:
            bytes_data = uploaded_file.getvalue()
            try: df = pd.read_csv(io.BytesIO(bytes_data)); 
            except: df = pd.read_csv(io.BytesIO(bytes_data), sep=';')
            if len(df.columns) < 2: continue

            df.columns = [c.strip() for c in df.columns]
            req = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            if not all(col in df.columns for col in req): continue
            
            cols = req + ['FTR']
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

def parse_fixtures_csv(file):
    try:
        df = pd.read_csv(file)
        if not {'Div', 'HomeTeam', 'AwayTeam'}.issubset(df.columns): return [], "Brak kolumn Div/HomeTeam/AwayTeam"
        matches = []
        for _, row in df.iterrows():
            matches.append({'Home': row['HomeTeam'], 'Away': row['AwayTeam'], 'League': row['Div']})
        return matches, None
    except Exception as e: return [], str(e)

# --- INIT ---
if 'fixture_pool' not in st.session_state:
    st.session_state.fixture_pool = load_fixture_pool()
if 'generated_coupon' not in st.session_state:
    st.session_state.generated_coupon = None

# --- INTERFEJS ---
st.title("☁️ MintStats v12.3: Aliases Fix")

st.sidebar.header("Panel Sterowania")
mode = st.sidebar.radio("Wybierz moduł:", ["1. 🛠️ ADMIN (Baza Danych)", "2. 🚀 GENERATOR KUPONÓW"])

if mode == "1. 🛠️ ADMIN (Baza Danych)":
    st.subheader("🛠️ Zarządzanie Bazą Danych")
    uploaded_history = st.file_uploader("Wgraj pliki ligowe (Historia)", type=['csv'], accept_multiple_files=True)
    if uploaded_history and st.button("Aktualizuj Bazę Danych"):
        with st.spinner("Przetwarzanie..."):
            count = process_uploaded_history(uploaded_history)
            if count > 0: st.success(f"✅ Baza zawiera teraz {count} meczów historycznych.")
            else: st.error("Błąd importu.")
    leagues = get_leagues_list()
    if leagues:
        st.write("---"); st.success(f"Dostępne ligi w bazie: {len(leagues)}"); st.write(leagues)
    else: st.warning("Baza pusta!")

elif mode == "2. 🚀 GENERATOR KUPONÓW":
    leagues = get_leagues_list()
    if not leagues: st.error("⛔ Baza pusta! Idź do ADMINA."); st.stop()
        
    df_all = get_all_data()
    model = PoissonModel(df_all)
    gen = CouponGenerator(model)
    all_teams_list = pd.concat([df_all['HomeTeam'], df_all['AwayTeam']]).unique()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dodaj Mecze")
    
    tab_manual, tab_ocr, tab_csv = st.sidebar.tabs(["Ręczny", "📸 Zdjęcie", "📁 CSV"])
    
    new_items = []
    with tab_manual:
        sel_league = st.selectbox("Liga:", leagues)
        df_l = get_data_for_league(sel_league)
        teams = sorted(pd.concat([df_l['HomeTeam'], df_l['AwayTeam']]).unique())
        with st.form("manual_add"):
            h = st.selectbox("Dom", teams); a = st.selectbox("Wyjazd", teams)
            if st.form_submit_button("➕ Dodaj") and h!=a: new_items.append({'Home':h, 'Away':a, 'League':sel_league})

    with tab_ocr:
        uploaded_img = st.file_uploader("Screen Flashscore", type=['png', 'jpg', 'jpeg'])
        if uploaded_img and st.button("Skanuj"):
            with st.spinner("OCR..."):
                txt = extract_text_from_image(uploaded_img)
                m_list, _ = smart_parse_matches_v2(txt, all_teams_list)
                if m_list: new_items.extend(m_list); st.success(f"Wykryto {len(m_list)}")
                else: st.warning("Brak")

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

    st.subheader("📋 Terminarz (Edytowalny)")
    if st.session_state.fixture_pool:
        df_pool = pd.DataFrame(st.session_state.fixture_pool)
        edited_df = st.data_editor(df_pool, num_rows="dynamic", use_container_width=True, key="fixture_editor")
        current_data = edited_df.to_dict('records')
        if current_data != st.session_state.fixture_pool:
            st.session_state.fixture_pool = current_data
            save_fixture_pool(current_data)
            
        if st.button("🗑️ Wyczyść CAŁOŚĆ"):
            st.session_state.fixture_pool = []; save_fixture_pool([]); st.rerun()

        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1: size = st.slider("Długość", 1, 50, 12)
        with c2: strat = st.selectbox("Strategia", ["Mieszany", "Over 2.5", "1 (Gospodarz)", "2 (Gość)", "1X", "X2", "BTS Tak"])
        with c3:
            st.write(""); st.write("")
            if st.button("🚀 GENERUJ", type="primary"):
                res = gen.analyze_pool(st.session_state.fixture_pool, strat)
                fin = sorted(res, key=lambda x: x['Pewność'], reverse=True)[:size]
                st.session_state.generated_coupon = {'data': fin, 'strat': strat}
                
        if st.session_state.generated_coupon:
            d = st.session_state.generated_coupon['data']
            st.write("---")
            st.subheader(f"🎫 Kupon: {st.session_state.generated_coupon['strat']}")
            if d: st.dataframe(pd.DataFrame(d).style.background_gradient(subset=['Pewność'], cmap="RdYlGn", vmin=0.4, vmax=0.9).format({'Pewność':'{:.1%}'}), use_container_width=True)
            else: st.warning("Brak pewnych typów.")
    else: st.info("Pula pusta.")
