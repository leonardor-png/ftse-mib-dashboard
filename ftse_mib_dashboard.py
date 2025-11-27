import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# =============================================================================
# CONFIGURAZIONE INIZIALE E DIPENDENZE
# =============================================================================
# Imposta il layout della pagina su 'wide' per sfruttare tutto lo schermo
# Definisce il titolo della tab del browser.
st.set_page_config(page_title="FTSE MIB Dashboard", layout="wide", page_icon="📈")

# =============================================================================
# STYLING CSS PERSONALIZZATO
# =============================================================================
# Utilizziamo st.markdown con unsafe_allow_html per usare CSS avanzato.
# Questo è necessario per sovrascrivere il tema di default di Streamlit e
# ottenere l'aspetto "Financial Dashboard"
st.markdown("""
<style>
    /* Importazione Font Roboto per un look moderno e pulito */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* 1. SFONDO GENERALE */
    /* Imposta il colore di sfondo dell'intera applicazione su un giallo pastello molto chiaro */
    .stApp {
        background-color: #FFFDE7; 
    }
    
    /* 2. TIPOGRAFIA GLOBALE */
    /* Forza il colore del testo a nero per garantire massimo contrasto */
    html, body, p, li, div, span, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: #000000 !important; 
    }

    /* 3. CARD DEL TITOLO PRINCIPALE */
    /* Box grigio con bordi arrotondati per racchiudere titolo e sottotitolo */
    .title-card {
        background-color: #e0e0e0;
        border: 1px solid #d1d1d1;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
        text-align: center;
    }
    .title-card h1 {
        margin: 0;
        padding-bottom: 10px;
        font-size: 2.5rem;
        color: #000000 !important;
    }

    /* 4. STYLING AVANZATO DELLE TABELLE (st.table) */
    /* Utilizziamo selettori specifici per forzare i colori delle intestazioni e delle celle,
       bypassando le impostazioni di default del tema Streamlit */
    
    /* Intestazioni (Header in alto e Indice a sinistra) */
    div[data-testid="stTable"] table thead th, 
    div[data-testid="stTable"] table tbody th {
        background-color: #999999 !important; /* Grigio Scuro/Medio */
        color: #000000 !important;            /* Testo Nero */
        font-weight: 900 !important;          /* Grassetto */
        border: 1px solid #ffffff !important; /* Bordo bianco per separazione */
        text-align: center !important;
    }

    /* Celle Dati (Corpo centrale) */
    div[data-testid="stTable"] table tbody td {
        background-color: #eeeeee !important; /* Grigio Chiaro */
        color: #000000 !important;            /* Testo Nero */
        border: 1px solid #ffffff !important;
    }

    /* Contenitore esterno della tabella */
    div[data-testid="stTable"] {
        border: 1px solid #999999;
        border-radius: 4px;
        overflow: hidden;
    }

    /* 5. PULSANTI DI NAVIGAZIONE (TAB) */
    /* Stile personalizzato per sembrare pulsanti fisici invece che link */
    button[data-baseweb="tab"] {
        background-color: #e0e0e0; 
        border: 1px solid #d1d1d1;
        color: #000000;
        font-weight: bold;
    }
    /* Stato Attivo: Grigio più scuro, mantiene il testo nero */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #c0c0c0 !important;
        border: 1px solid #666 !important;
        color: #000000 !important;
    }
    /* Rimuove la linea decorativa standard di Streamlit */
    div[data-testid="stTabs"] > div > div {
        box-shadow: none !important;
        border-bottom: none !important;
        gap: 0px;
    }

    /* 6. PULSANTI STANDARD (Avvia Ottimizzazione / Aggiorna) */
    div.stButton > button {
        background-color: #e0e0e0;
        color: #000000;
        border: 1px solid #999;
        font-weight: bold;
        width: 100%; 
    }
    div.stButton > button:hover {
        border-color: #000000;
        background-color: #dcdcdc;
    }

    /* 7. BOX METRICHE KPI */
    .metrics-card {
        background-color: #e0e0e0;
        border: 1px solid #999;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* 8. ELEMENTI UTILITY */
    /* Linea divisoria personalizzata */
    hr.custom-divider {
        margin-top: 0px;
        margin-bottom: 25px;
        border: 0;
        border-top: 2px solid #999;
    }
    
    /* Contenitore per descrizione grafico e legenda */
    .chart-desc-container {
        background-color: #ffffff;
        border-left: 4px solid #999;
        padding: 15px;
        border-radius: 0 5px 5px 0;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    /* Box scrollabile per la legenda dei ticker */
    .legend-scroll-box {
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px dashed #ccc;
        max-height: 120px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    .legend-item {
        margin-bottom: 4px;
        line-height: 1.2;
    }
    
    /* Box esplicativo per la frontiera efficiente */
    .frontier-explainer {
        background-color: #ffffff;
        border: 1px solid #999;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURAZIONE GRAFICI (MATPLOTLIB & SEABORN)
# =============================================================================
sns.set_theme(style="ticks", context="notebook")
# Dimensione ridotta (6.0, 3.5) per visualizzazione compatta
plt.rcParams['figure.figsize'] = (6.0, 3.5)
# Colore di sfondo esterno (match con il sito)
plt.rcParams['figure.facecolor'] = '#FFFDE7' 
# Colore interno del grafico (bianco per contrasto dati)
plt.rcParams['axes.facecolor'] = '#FFFFFF'
# Colori testi e assi
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1

# =============================================================================
# CLASSE 1: DATA MANAGER
# Gestisce il recupero dati, la pulizia e il calcolo della classifica.
# =============================================================================
class DataManager:
    """
    Classe responsabile per l'interazione con le API di Yahoo Finance.
    Gestisce il download dei prezzi e il calcolo real-time della Capitalizzazione.
    """
    def __init__(self, benchmark="FTSEMIB.MI", start_date="2019-01-01"):
        self.benchmark = benchmark
        self.start_date = start_date

    def _get_mapping(self):
        """Restituisce il dizionario completo dei ticker del paniere FTSE MIB."""
        return {
            "A2A": "A2A.MI", "Amplifon": "AMP.MI", "Generali": "G.MI",
            "Azimut": "AZM.MI", "Banca Mediolanum": "BMED.MI", "Banca Pop. Sondrio": "BPSO.MI",
            "Banco Bpm": "BAMI.MI", "MPS": "BMPS.MI", "Bper Banca": "BPE.MI",
            "Brunello Cucinelli": "BC.MI", "Buzzi Unicem": "BZU.MI", "Campari": "CPR.MI",
            "DiaSorin": "DIA.MI", "Enel": "ENEL.MI", "Eni": "ENI.MI", "Ferrari": "RACE.MI",
            "FinecoBank": "FBK.MI", "Hera": "HER.MI", "Interpump": "IP.MI",
            "Intesa Sanpaolo": "ISP.MI", "Inwit": "INW.MI", "Italgas": "IG.MI", "Iveco": "IVG.MI",
            "Leonardo": "LDO.MI", "Lottomatica": "LTMC.MI", "Mediobanca": "MB.MI",
            "Moncler": "MONC.MI", "Nexi": "NEXI.MI", "Poste Italiane": "PST.MI",
            "Prysmian": "PRY.MI", "Recordati": "REC.MI", "Saipem": "SPM.MI", "Snam": "SRG.MI",
            "Stellantis": "STLAM.MI", "STMicro": "STM.MI", "Telecom Italia": "TIT.MI",
            "Tenaris": "TEN.MI", "Terna": "TRN.MI", "UniCredit": "UCG.MI", "Unipol": "UNI.MI"
        }

    def get_ticker_to_name_mapping(self):
        """Restituisce una mappa inversa {Ticker: Nome Azienda} per la legenda."""
        original_map = self._get_mapping()
        return {v: k for k, v in original_map.items()}

    def get_top_10_tickers(self):
        """
        Calcola la Top 10 per Market Cap in tempo reale.
        Scarica il numero di azioni (shares) e il prezzo live per ogni titolo.
        """
        mapping = self._get_mapping()
        all_tickers = list(mapping.values())
        market_caps = {}
        
        # Feedback visivo per l'utente durante il download
        progress_bar = st.progress(0, text="Calcolo Market Cap in tempo reale...")
        try:
            # Batch download dei prezzi (veloce)
            batch_data = yf.download(all_tickers, period="1d", progress=False)
            if 'Close' in batch_data.columns:
                last_prices = batch_data['Close'].iloc[-1]
            else:
                last_prices = batch_data.iloc[-1]
            
            # Loop per scaricare metadati (shares) - più lento per evitare il blocco di yahoo finance
            total_tickers = len(all_tickers)
            for i, ticker in enumerate(all_tickers):
                progress_bar.progress((i + 1) / total_tickers, text=f"Analisi ticker: {ticker}")
                try:
                    price = last_prices.get(ticker)
                    if pd.isna(price): continue
                    ticker_obj = yf.Ticker(ticker)
                    shares = ticker_obj.fast_info.get('shares', 0)
                    if shares > 0 and price > 0:
                        market_caps[ticker] = price * shares
                except: continue
            
            progress_bar.empty()
            # Ordinamento decrescente
            sorted_caps = dict(sorted(market_caps.items(), key=lambda item: item[1], reverse=True))
            return list(sorted_caps.keys())[:10]
        except Exception as e:
            st.error(f"Errore critico classifica: {e}")
            return []

    def download_historical_data(self, tickers):
        """Scarica i prezzi storici chiusi aggiustati."""
        full_list = tickers + [self.benchmark]
        try:
            raw_data = yf.download(full_list, start=self.start_date, progress=False)
            # Gestione robusta colonne (Adj Close vs Close)
            if 'Adj Close' in raw_data.columns:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                data = raw_data['Close']
            else:
                try:
                    data = raw_data.xs('Adj Close', level=0, axis=1)
                except:
                    data = raw_data.xs('Close', level=0, axis=1)
            return data.dropna()
        except: return pd.DataFrame()

# =============================================================================
# CLASSE 2: FINANCIAL ANALYZER
# Esegue i calcoli statistici sui dati finanziari.
# =============================================================================
class FinancialAnalyzer:
    def __init__(self, data, benchmark_data=None):
        self.prices = data
        self.bench_p = benchmark_data
        self.returns = pd.DataFrame()
        self.bench_r = None

    def calculate_returns(self):
        """Calcola i rendimenti giornalieri e allinea le date col benchmark."""
        self.returns = self.prices.pct_change().dropna()
        if self.bench_p is not None:
            if isinstance(self.bench_p, pd.DataFrame):
                self.bench_p = self.bench_p.iloc[:, 0]
            self.bench_r = self.bench_p.pct_change().dropna()
            idx = self.returns.index.intersection(self.bench_r.index)
            self.returns = self.returns.loc[idx]
            self.bench_r = self.bench_r.loc[idx]
        return self.returns

    def _calc_max_drawdown(self, series):
        """Calcola il Max Drawdown (perdita massima dal picco)."""
        comp = (1 + series).cumprod()
        peak = comp.expanding(min_periods=1).max()
        if peak.empty: return 0.0
        return ((comp/peak) - 1).min()

    def _prepare_stats_dataframe(self):
        """Unisce stock e benchmark per le tabelle statistiche."""
        df_calc = self.returns.copy()
        if self.bench_r is not None:
            bench_s = self.bench_r.copy()
            bench_s.name = "FTSE MIB"
            df_calc = pd.concat([df_calc, bench_s], axis=1)
        return df_calc

    def get_table_1_central_metrics(self):
        """Crea la tabella 1 e calcola statistiche fondamentali"""
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['median', 'std', 'var', 'mean']).T
        stats_df['Media Geom. (Ann)'] = df_calc.apply(lambda x: (stats.gmean(x + 1)**252 - 1) if len(x) > 0 else 0)
        stats_df.rename(columns={'mean': 'Media Giorn.', 'median': 'Mediana', 'std': 'Dev.Std', 'var': 'Varianza'}, inplace=True)
        return stats_df[['Media Geom. (Ann)', 'Media Giorn.', 'Mediana', 'Dev.Std', 'Varianza']]

    def get_table_2_risk_extremes(self):
        """Crea la tabella 2 calcolando altre statistiche fondamentali"""
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['min', 'max']).T
        stats_df['Range'] = stats_df['max'] - stats_df['min']
        stats_df['Max Drawdown'] = df_calc.apply(self._calc_max_drawdown)
        if self.bench_r is not None:
            bench_s = self.bench_r
            for col in self.returns.columns: 
                stats_df.loc[col, 'Cov. Mkt'] = self.returns[col].cov(bench_s)
                stats_df.loc[col, 'Corr. Mkt'] = self.returns[col].corr(bench_s)
            stats_df.loc['FTSE MIB', ['Cov. Mkt', 'Corr. Mkt']] = np.nan
        stats_df.rename(columns={'min': 'Min', 'max': 'Max'}, inplace=True)
        return stats_df[['Min', 'Max', 'Range', 'Max Drawdown', 'Cov. Mkt', 'Corr. Mkt']]

    def get_table_3_non_normality(self):
        """Crea la tabella per i dati della distribuzione normale"""
        df_calc = self._prepare_stats_dataframe()
        non_norm = df_calc.agg(['skew', 'kurt']).T
        non_norm.rename(columns={'skew': 'Asimmetria', 'kurt': 'Curtosi'}, inplace=True)
        return non_norm

    def get_jarque_bera_test(self):
        """Crea una tabella in cui viene eseguito il test statistico per valutare se i rendimenti dei titoli hanno distribuzione normale"""
        res = pd.DataFrame({'p-value': self.returns.apply(lambda x: stats.jarque_bera(x)[1])})
        res['Esito'] = np.where(res['p-value'] > 0.05, "NORMALE", "NON NORMALE")
        return res

# =============================================================================
# CLASSE 3: VISUALIZER
# Gestisce la creazione dei grafici
# =============================================================================
class Visualizer:
    def __init__(self, prices, returns, benchmark=None, non_norm_metrics=None):
        self.prices, self.returns, self.bench = prices, returns, benchmark
        self.non_norm_metrics = non_norm_metrics

    def _add_border(self, fig):
        """Aggiunge il bordo grigio esterno alla figura."""
        fig.patch.set_linewidth(1.5)
        fig.patch.set_edgecolor('#999999')
        return fig

    def plot_normalized_prices(self):
        """Grafico performance base 100."""
        norm = (self.prices / self.prices.iloc[0]) * 100
        fig, ax = plt.subplots() 
        colors = sns.color_palette("husl", len(norm.columns))
        for i, c in enumerate(norm.columns):
            ax.plot(norm.index, norm[c], label=c, alpha=0.9, linewidth=1.5, color=colors[i])
        if self.bench is not None:
            bn = (self.bench / self.bench.iloc[0]) * 100
            ax.plot(bn.index, bn, label="FTSE MIB", color='#000000', ls='--', lw=2.5)
        
        # Asse Y diviso per 100
        ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
        
        ax.set_xlabel("")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize='small')
        plt.tight_layout()
        return self._add_border(fig)

    def plot_returns_boxplot(self):
        """Boxplot distribuzione rendimenti."""
        fig, ax = plt.subplots()
        sns.boxplot(data=self.returns, ax=ax, palette="light:b", fliersize=3, linewidth=1)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._add_border(fig)

    def plot_histogram_grid(self):
        """Creazione di grafico di distribuzione dei rendimenti tramite normale, con istogramma separato per titolo ."""
        df_combined = self.returns.copy()
        if self.bench is not None:
             df_combined['FTSE MIB'] = self.bench.pct_change().dropna()
        df_combined = df_combined.dropna()
        melt = df_combined.melt(var_name='Ticker', value_name='Rendimento')
        
        # Calibrazione misure di aspetto
        g = sns.FacetGrid(melt, col="Ticker", col_wrap=3, sharex=False, sharey=False, height=1.5, aspect=1.3)
        g.map_dataframe(sns.histplot, x="Rendimento", kde=True, color="#778899", edgecolor="white", linewidth=0.5)
        
        # Titoli piccoli
        g.set_titles("{col_name}", fontweight='bold', size=8)
        
        g.set_axis_labels("", "")
        for ax in g.axes.flat:
            ax.set_xlabel("")
            ax.set_ylabel("")
            # Etichette assi più piccole per evitare sovrapposizioni
            ax.tick_params(axis='both', which='major', labelsize=6)
            
        g.despine(left=True)
        legend_elements = [
            Patch(facecolor='#778899', edgecolor='none', label='Frequenza'),
            Line2D([0], [0], color='#778899', lw=2, label='Densità (KDE)')
        ]
        g.fig.legend(handles=legend_elements, loc='lower right', fontsize=8, bbox_to_anchor=(0.95, 0.05), frameon=False)
        plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
        
        # Bordi grigi ai grafici
        g.fig.patch.set_linewidth(1.5)
        g.fig.patch.set_edgecolor('#999999')
        return g.fig

    def plot_correlation_heatmap(self):
        """Matrice di correlazione."""
        fig, ax = plt.subplots()
        sns.heatmap(self.returns.corr(), annot=True, cmap='vlag', center=0, fmt=".2f", 
                    ax=ax, cbar_kws={'label': 'Correlazione'}, linewidths=0.5, linecolor='white', annot_kws={"size": 7})
        return self._add_border(fig)

# =============================================================================
# CLASSE 4: PORTFOLIO OPTIMIZER
# Simulazione Monte Carlo per la Frontiera Efficiente.
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, returns_df, num_portfolios=5000):
        """Configura l'ambiente per la simulazione Monte Carlo (input dati e parametri iniziali)"""
        self.ret = returns_df
        self.n = num_portfolios
        self.results = None
        self.weights = [] 

    # Calcola le metriche statistiche fondamentali (media e covarianza) e imposta la riproducibilità (seed) prima di avviare il calcolo dei portafogli
    def simulate(self):
        np.random.seed(42)
        mean_daily = self.ret.mean()
        cov_matrix = self.ret.cov()
        n_assets = len(self.ret.columns)
        results_list = []
        weights_list = [] 

        #  Esegue il ciclo Monte Carlo: genera pesi casuali normalizzati e calcola rendimento, volatilità e Sharpe Ratio per ogni portafoglio simulato
        for _ in range(self.n):
            w = np.random.random(n_assets)
            w /= np.sum(w)
            ret_ann = np.sum(mean_daily * w) * 252
            vol_ann = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
            sharpe = ret_ann / vol_ann if vol_ann > 0 else 0
            results_list.append([ret_ann, vol_ann, sharpe])
            weights_list.append(w)
        # Individua i due portafogli Max Sharpe e Min Vol, estraendone le metriche e la composizione   
        self.results = pd.DataFrame(results_list, columns=['Rendimento', 'Volatilità', 'Sharpe'])
        self.weights = np.array(weights_list)
        max_sharpe_idx = self.results['Sharpe'].idxmax()
        min_vol_idx = self.results['Volatilità'].idxmin()
        max_sharpe_pt = self.results.iloc[max_sharpe_idx]
        min_vol_pt = self.results.iloc[min_vol_idx]
        max_w = self.weights[max_sharpe_idx]
        min_w = self.weights[min_vol_idx]
        return max_sharpe_pt, min_vol_pt, max_w, min_w

    def plot_efficient_frontier(self, max_pt, min_pt):
        # Mantiene dimensione grande (10, 6) per leggibilità
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(self.results['Volatilità'], self.results['Rendimento'], c=self.results['Sharpe'], cmap='viridis', s=15, alpha=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=10)
        cbar.outline.set_visible(False)
        ax.scatter(max_pt['Volatilità'], max_pt['Rendimento'], c='#d62728', s=200, marker='*', label='Max Sharpe', edgecolors='black')
        ax.scatter(min_pt['Volatilità'], min_pt['Rendimento'], c='#1f77b4', s=200, marker='*', label='Min Vol', edgecolors='black')
        ax.set_title("Frontiera Efficiente (Markowitz)", fontweight='bold', pad=15)
        ax.set_xlabel("Volatilità (Rischio Annualizzato)")
        ax.set_ylabel("Rendimento Atteso (Annualizzato)")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        fig.patch.set_linewidth(1.5)
        fig.patch.set_edgecolor('#999999')
        return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    col_title, col_btn = st.columns([5, 1])
    # Intestazione (header) della dashboard con HTML per applicare uno stile grafico e un layout
    with col_title:
        st.markdown("""
            <div class="title-card">
                <h1>FTSE MIB Top 10 Dashboard</h1>
                <p>ANALISI FINANZIARIA AUTOMATIZZATA SUI TOP PLAYER DEL MERCATO ITALIANO.</p>
            </div>
        """, unsafe_allow_html=True)
    # Inserisce un pulsante di 'Refresh' allineato verticalmente, che permette di ricaricare l'intera applicazione per ottenere dati aggiornati
    with col_btn:
        st.write("") 
        st.write("") 
        if st.button("🔄 Aggiorna Dati"):
            st.cache_data.clear()
            st.rerun()
    # Setup della memoria persistente per evitare di perdere i dati della simulazione ad ogni interazione dell'utente
    if 'opt_done' not in st.session_state:
        st.session_state.opt_done = False
        st.session_state.opt_max = None
        st.session_state.opt_min = None
        st.session_state.opt_w_max = None
        st.session_state.opt_w_min = None
        st.session_state.opt_obj = None
    
    # Recupera i dati di mercato (ticker, storico prezzi e nomi completi) sfruttando il sistema di caching di Streamlit per evitare download ridondanti per 1 ora (3600s)
    @st.cache_data(ttl=3600)
    def get_market_data():
        dm = DataManager()
        tickers = dm.get_top_10_tickers()
        if not tickers: return None, None
        df = dm.download_historical_data(tickers)
        mapping = dm.get_ticker_to_name_mapping()
        return tickers, df, mapping
    # Esegue il recupero dati mostrando un indicatore di caricamento all'utente e gestisce eventuali fallimenti bloccando l'esecuzione in caso di errore
    with st.spinner("Scansione in tempo reale del mercato (potrebbe richiedere qualche secondo)..."):
        data_result = get_market_data()
        if data_result is None or data_result[0] is None:
            st.error("Errore recupero dati.")
            return
        tickers, df_tot, ticker_mapping = data_result

    # Separa l'indice di riferimento (FTSE MIB) delle singole azioni e inizializza il motore di analisi calcolando i rendimenti iniziali
    if df_tot is not None and not df_tot.empty:
        bench = None
        df_stocks = df_tot.copy()
        if "FTSEMIB.MI" in df_tot.columns:
            bench = df_tot["FTSEMIB.MI"]
            df_stocks = df_tot.drop("FTSEMIB.MI", axis=1)
        
        an = FinancialAnalyzer(df_stocks, bench)
        rets = an.calculate_returns()

        # Valida i dati, esegue tutti i calcoli statistici necessari, inizializza il motore grafico e organizza il layout della pagina in tre schede tematiche
        if rets.empty:
            st.error("Dati insufficienti per i calcoli.")
            return

        t1 = an.get_table_1_central_metrics()
        t2 = an.get_table_2_risk_extremes()
        t3 = an.get_table_3_non_normality()
        t_jb = an.get_jarque_bera_test()
        
        viz = Visualizer(df_stocks, rets, bench, non_norm_metrics=t3)

        tab1, tab2, tab3 = st.tabs(["Statistiche Avanzate", "Analisi Grafica", "Frontiera Efficiente"])
        # Funzione di callback: esegue la simulazione di portafoglio e salva i risultati della sessione per renderli persistenti e accessibili all'interfaccia
        def run_optimization_callback():
            opt = PortfolioOptimizer(rets)
            res_max, res_min, w_max, w_min = opt.simulate()
            st.session_state.opt_max = res_max
            st.session_state.opt_min = res_min
            st.session_state.opt_w_max = w_max
            st.session_state.opt_w_min = w_min
            st.session_state.opt_obj = opt
            st.session_state.opt_done = True

        # --- TAB 1: TABELLE ---
        with tab1:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            
            st.subheader("1. Performance e Volatilità")
            st.table(t1.style.format("{:.2%}", subset=['Media Geom. (Ann)', 'Media Giorn.', 'Dev.Std', 'Varianza']))

            st.subheader("2. Analisi del Rischio")
            st.table(t2.style.format("{:.2%}", subset=['Min', 'Max', 'Range', 'Max Drawdown'])
                         .format("{:.4f}", subset=['Cov. Mkt', 'Corr. Mkt']))

            c1, c2 = st.columns(2)
            with c1: 
                st.subheader("3. Asimmetria e Curtosi")
                st.table(t3.style.format("{:.4f}"))
            with c2: 
                st.subheader("4. Test di Normalità (Jarque-Bera)")
                st.table(t_jb.style.format({"p-value": "{:.4f}"}))

        # --- TAB 2: GRAFICI SIDE-BY-SIDE CON LEGENDA ---
        with tab2:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            
            def render_plot_with_description(title, fig, description, active_tickers, mapping):
                col_chart, col_text = st.columns([2, 1])
                
                with col_chart:
                    st.pyplot(fig)
                
                with col_text:
                    legend_html = "<b>Legenda Ticker:</b><div class='legend-scroll-box'>"
                    for t in active_tickers:
                        clean_t = t.replace(".MI", "")
                        name = mapping.get(t, "N/A")
                        legend_html += f"<div class='legend-item'><b>{clean_t}</b>: {name}</div>"
                    legend_html += "</div>"

                    st.markdown(f"""
                    <div class="chart-desc-container">
                        <h3 style="margin-top:0;">{title}</h3>
                        <p style="font-size:0.95rem; line-height:1.5;">{description}</p>
                        {legend_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")

            current_tickers = df_stocks.columns.tolist()

            render_plot_with_description(
                "Performance Relativa", 
                viz.plot_normalized_prices(),
                "Mostra l'andamento dei prezzi normalizzati a quota 100 all'inizio del periodo.<br>"
                "Permette di confrontare la performance percentuale cumulata di titoli con prezzi diversi.<br>"
                "Se una linea raggiunge 110, significa che il titolo ha guadagnato il 10%.",
                current_tickers, ticker_mapping
            )

            render_plot_with_description(
                "Dispersione (Rischio)", 
                viz.plot_returns_boxplot(),
                "Visualizza la volatilità dei rendimenti giornalieri.<br>"
                "Una scatola più alta indica un titolo più rischioso.<br>"
                "I 'baffi' (linee verticali) delimitano l'intervallo di oscillazione tipico, mentre i punti esterni rappresentano i movimenti di prezzo estremi o anomali (outlier).",
                current_tickers, ticker_mapping
            )

            render_plot_with_description(
                "Correlazioni", 
                viz.plot_correlation_heatmap(),
                "Misura il grado di interdipendenza tra i rendimenti dei titoli. Un valore vicino a +1 indica che i titoli tendono a muoversi nella stessa direzione, mentre un valore verso -1 indica movimenti opposti.<br>"
                "I valori vicini allo 0 indicano assenza di legame.",
                current_tickers, ticker_mapping
            )

            render_plot_with_description(
                "Distribuzioni", 
                viz.plot_histogram_grid(),
                "Mostra la frequenza dei rendimenti giornalieri: una forma a campana indica normalità.<br>"
                "Code lunghe o asimmetrie segnalano un rischio maggiore di eventi estremi non previsti.",
                current_tickers, ticker_mapping
            )

        # --- TAB 3: OTTIMIZZAZIONE ---
        with tab3:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.markdown("### Ottimizzazione di Portafoglio (Markowitz)")
            st.caption("Simulazione Monte Carlo su 5000 portafogli casuali.")
            
            st.button("🚀 Avvia Ottimizzazione", on_click=run_optimization_callback)
            
            if st.session_state.opt_done:
                max_pt = st.session_state.opt_max
                min_pt = st.session_state.opt_min
                w_max = st.session_state.opt_w_max
                w_min = st.session_state.opt_w_min
                opt_obj = st.session_state.opt_obj
                
                with st.container():
                    st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("🚀 Max Sharpe", f"{max_pt['Rendimento']:.2%}", f"Vol: {max_pt['Volatilità']:.2%}")
                    c2.metric("🛡️ Min Volatility", f"{min_pt['Rendimento']:.2%}", f"Vol: {min_pt['Volatilità']:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)

                col_plot, col_info = st.columns(2)
                
                with col_plot:
                    st.pyplot(opt_obj.plot_efficient_frontier(max_pt, min_pt))
                    
                    # --- AGGIUNTA SPIEGAZIONE METODOLOGICA ---
                    st.markdown("""
                    <div class="frontier-explainer">
                        <b>Nota Metodologica:</b> Il grafico utilizza il modello <i>Mean-Variance</i> (Markowitz) con simulazione Monte Carlo.<br>
                        <ul>
                            <li><b>Max Sharpe (Stella Rossa):</b> Rappresenta il portafoglio più efficiente, quello che offre il miglior rendimento per unità di rischio assunto.</li>
                            <li><b>Min Volatility (Stella Blu):</b> Indica il portafoglio più conservativo in assoluto, minimizzando la fluttuazione (rischio) indipendentemente dal rendimento.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_info:
                    st.write("### 🏗️ Allocazione")
                    
                    def make_weight_df(weights, tickers):
                        df_w = pd.DataFrame({'Ticker': tickers, 'Peso': weights * 100})
                        df_w = df_w[df_w['Peso'] > 0.0001].sort_values('Peso', ascending=False)
                        df_w['Peso'] = df_w['Peso'].apply(lambda x: f"{x:.2f}%")
                        return df_w.set_index('Ticker')

                    df_w_max = make_weight_df(w_max, rets.columns)
                    df_w_min = make_weight_df(w_min, rets.columns)

                    t1, t2 = st.tabs(["Max Sharpe", "Min Volatility"])
                    
                    with t1:
                        st.table(df_w_max)
                    with t2:
                        st.table(df_w_min)
            else:
                st.info("Clicca sul pulsante per avviare la simulazione.")
    else:
        st.error("Errore critico: Impossibile scaricare i dati. Controlla la connessione internet o le API di Yahoo Finance.")

if __name__ == "__main__":
    main()






