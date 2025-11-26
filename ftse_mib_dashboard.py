import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="FTSE MIB Dashboard", layout="wide", page_icon="📈")

# --- CUSTOM CSS AVANZATO ---
st.markdown("""
<style>
    /* Importa font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* 1. SFONDO GENERALE */
    .stApp {
        background-color: #FDFBF7; /* Bianco panna */
    }
    
    /* 2. TESTI GLOBALI */
    html, body, p, li, div, span, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: #484848 !important;
    }

    /* 3. CARD TITOLO */
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
        color: #2c2c2c !important;
    }
    .title-card p {
        margin: 0;
        font-size: 1.1rem;
        color: #555 !important;
    }

    /* 4. CARD METRICHE FRONTIERA */
    .metrics-card {
        background-color: #e0e0e0;
        border: 1px solid #d1d1d1;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* 5. STILE PULSANTI TAB (SCHEDE) */
    button[data-baseweb="tab"] {
        background-color: #e0e0e0; 
        border: 1px solid #d1d1d1;
        border-radius: 6px;
        color: #484848;
        margin-right: 5px;
        padding: 8px 16px;
    }
    /* Bottone Attivo */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #d6d6d6 !important;
        border: 1px solid #999 !important;
        color: #000000 !important;
        font-weight: 900 !important;
    }
    div[data-testid="stTabs"] > div > div {
        box-shadow: none !important;
        border-bottom: none !important;
        gap: 0px;
    }

    /* 6. STILE PULSANTI STANDARD (Avvia Ottimizzazione & Aggiorna) */
    div.stButton > button {
        background-color: #e0e0e0;
        color: #484848;
        border: 1px solid #b0b0b0;
        border-radius: 5px;
        font-weight: 600;
        width: 100%; 
    }
    div.stButton > button:focus, div.stButton > button:active {
        background-color: #d0d0d0 !important;
        color: #000000 !important;
        border-color: #808080 !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        border-color: #484848;
        color: #000000;
    }

    /* 7. TABELLE */
    .stDataFrame {
        border: 1px solid #dcdcdc;
        border-radius: 5px;
    }
    [data-testid="stDataFrame"] table {
        --ag-selected-row-background-color: #d3e2f2 !important;
        --ag-row-hover-color: #f0f0f0 !important;
    }

    /* 8. METRICHE STANDARD */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #2c2c2c !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #808080;
    }
    
    /* LINEA DIVISORIA PERSONALIZZATA */
    hr.custom-divider {
        margin-top: 0px;
        margin-bottom: 25px;
        border: 0;
        border-top: 2px solid #b0b0b0;
    }
</style>
""", unsafe_allow_html=True)

# Impostazioni grafici globali
sns.set_theme(style="ticks", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.facecolor'] = '#FDFBF7' 
plt.rcParams['axes.facecolor'] = '#FDFBF7'
plt.rcParams['text.color'] = '#484848'
plt.rcParams['axes.labelcolor'] = '#484848'
plt.rcParams['xtick.color'] = '#484848'
plt.rcParams['ytick.color'] = '#484848'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.edgecolor'] = '#484848'

# =============================================================================
# CLASSE 1: DATA MANAGER
# =============================================================================
class DataManager:
    def __init__(self, benchmark="FTSEMIB.MI", start_date="2019-01-01"):
        self.benchmark = benchmark
        self.start_date = start_date

    def _get_mapping(self):
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

    def get_top_10_tickers(self):
        mapping = self._get_mapping()
        all_tickers = list(mapping.values())
        market_caps = {}
        
        progress_bar = st.progress(0, text="Calcolo Market Cap in tempo reale...")
        
        try:
            batch_data = yf.download(all_tickers, period="1d", progress=False)
            if 'Close' in batch_data.columns:
                last_prices = batch_data['Close'].iloc[-1]
            else:
                last_prices = batch_data.iloc[-1]
            
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
                except Exception:
                    continue

            progress_bar.empty()
            sorted_caps = dict(sorted(market_caps.items(), key=lambda item: item[1], reverse=True))
            return list(sorted_caps.keys())[:10]

        except Exception as e:
            st.error(f"Errore critico classifica: {e}")
            return []

    def download_historical_data(self, tickers):
        full_list = tickers + [self.benchmark]
        try:
            raw_data = yf.download(full_list, start=self.start_date, progress=False)
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
        except Exception as e:
            st.error(f"Errore download: {e}")
            return pd.DataFrame()

# =============================================================================
# CLASSE 2: FINANCIAL ANALYZER
# =============================================================================
class FinancialAnalyzer:
    def __init__(self, data, benchmark_data=None):
        self.prices = data
        self.bench_p = benchmark_data
        self.returns = pd.DataFrame()
        self.bench_r = None

    def calculate_returns(self):
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
        comp = (1 + series).cumprod()
        peak = comp.expanding(min_periods=1).max()
        if peak.empty: return 0.0
        return ((comp/peak) - 1).min()

    def _prepare_stats_dataframe(self):
        df_calc = self.returns.copy()
        if self.bench_r is not None:
            bench_s = self.bench_r.copy()
            bench_s.name = "FTSE MIB"
            df_calc = pd.concat([df_calc, bench_s], axis=1)
        return df_calc

    def get_table_1_central_metrics(self):
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['median', 'std', 'var', 'mean']).T
        stats_df['Media Geom. (Ann)'] = df_calc.apply(lambda x: (stats.gmean(x + 1)**252 - 1) if len(x) > 0 else 0)
        stats_df.rename(columns={'mean': 'Media Giorn.', 'median': 'Mediana', 'std': 'Dev.Std', 'var': 'Varianza'}, inplace=True)
        return stats_df[['Media Geom. (Ann)', 'Media Giorn.', 'Mediana', 'Dev.Std', 'Varianza']]

    def get_table_2_risk_extremes(self):
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
        df_calc = self._prepare_stats_dataframe()
        non_norm = df_calc.agg(['skew', 'kurt']).T
        non_norm.rename(columns={'skew': 'Asimmetria', 'kurt': 'Curtosi'}, inplace=True)
        return non_norm

    def get_jarque_bera_test(self):
        res = pd.DataFrame({'p-value': self.returns.apply(lambda x: stats.jarque_bera(x)[1])})
        res['Esito'] = np.where(res['p-value'] > 0.05, "NORMALE", "NON NORMALE")
        return res

# =============================================================================
# CLASSE 3: VISUALIZER
# =============================================================================
class Visualizer:
    def __init__(self, prices, returns, benchmark=None, non_norm_metrics=None):
        self.prices, self.returns, self.bench = prices, returns, benchmark
        self.non_norm_metrics = non_norm_metrics

    # Helper per aggiungere contorno ai grafici
    def _add_border(self, fig):
        fig.patch.set_linewidth(1.5)
        fig.patch.set_edgecolor('#cccccc')
        return fig

    def plot_normalized_prices(self):
        norm = (self.prices / self.prices.iloc[0]) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("husl", len(norm.columns))
        for i, c in enumerate(norm.columns):
            ax.plot(norm.index, norm[c], label=c, alpha=0.9, linewidth=1.5, color=colors[i])
        if self.bench is not None:
            bn = (self.bench / self.bench.iloc[0]) * 100
            ax.plot(bn.index, bn, label="FTSE MIB", color='#2c2c2c', ls='--', lw=2.5)
        ax.set_title("Performance Relativa (Base 100)", fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.tight_layout()
        return self._add_border(fig)

    def plot_returns_boxplot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.returns, ax=ax, palette="light:b", fliersize=3, linewidth=1)
        ax.set_title("Dispersione Rendimenti Giornalieri", fontweight='bold', pad=15)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._add_border(fig)

    def plot_histogram_grid(self):
        df_combined = self.returns.copy()
        if self.bench is not None:
             df_combined['FTSE MIB'] = self.bench.pct_change().dropna()
        df_combined = df_combined.dropna()
        melt = df_combined.melt(var_name='Ticker', value_name='Rendimento')
        g = sns.FacetGrid(melt, col="Ticker", col_wrap=3, sharex=False, sharey=False, height=2.0, aspect=1.5)
        g.map_dataframe(sns.histplot, x="Rendimento", kde=True, color="#778899", edgecolor="white", linewidth=0.5)
        g.set_titles("{col_name}", fontweight='bold')
        g.set_axis_labels("", "")
        g.despine(left=True)
        legend_elements = [
            Patch(facecolor='#778899', edgecolor='none', label='Frequenza'),
            Line2D([0], [0], color='#778899', lw=2, label='Densità (KDE)')
        ]
        g.fig.legend(handles=legend_elements, loc='lower right', fontsize=9, bbox_to_anchor=(0.95, 0.05), frameon=False)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('Distribuzione Rendimenti', fontweight='bold', y=0.98)
        
        g.fig.patch.set_linewidth(1.5)
        g.fig.patch.set_edgecolor('#cccccc')
        return g.fig

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self.returns.corr(), annot=True, cmap='vlag', center=0, fmt=".2f", 
                    ax=ax, cbar_kws={'label': 'Correlazione'}, linewidths=0.5, linecolor='white')
        ax.set_title("Matrice di Correlazione", fontweight='bold', pad=15)
        return self._add_border(fig)

# =============================================================================
# CLASSE 4: PORTFOLIO OPTIMIZER
# =============================================================================
class PortfolioOptimizer:
    # MODIFICA: 5000 simulazioni
    def __init__(self, returns_df, num_portfolios=5000):
        self.ret = returns_df
        self.n = num_portfolios
        self.results = None
        self.weights = [] 

    def simulate(self):
        np.random.seed(42)
        mean_daily = self.ret.mean()
        cov_matrix = self.ret.cov()
        n_assets = len(self.ret.columns)
        results_list = []
        weights_list = [] 

        for _ in range(self.n):
            w = np.random.random(n_assets)
            w /= np.sum(w)
            ret_ann = np.sum(mean_daily * w) * 252
            vol_ann = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)
            sharpe = ret_ann / vol_ann if vol_ann > 0 else 0
            results_list.append([ret_ann, vol_ann, sharpe])
            weights_list.append(w)
            
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
        fig.patch.set_edgecolor('#cccccc')
        return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    # --- HEADER: TITOLO (SX) E PULSANTE UPDATE (DX) ---
    col_title, col_btn = st.columns([5, 1])
    
    with col_title:
        st.markdown("""
            <div class="title-card">
                <h1>🇮🇹 FTSE MIB Top 10 Dashboard</h1>
                <p>Analisi finanziaria automatizzata sui top player del mercato italiano.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_btn:
        st.write("") # Spaziatura per allineare
        st.write("") 
        if st.button("🔄 Aggiorna Dati"):
            st.cache_data.clear()
            st.rerun()
    
    if 'opt_done' not in st.session_state:
        st.session_state.opt_done = False
        st.session_state.opt_max = None
        st.session_state.opt_min = None
        st.session_state.opt_w_max = None
        st.session_state.opt_w_min = None
        st.session_state.opt_obj = None

    @st.cache_data(ttl=3600)
    def get_market_data():
        dm = DataManager()
        tickers = dm.get_top_10_tickers()
        if not tickers: return None, None
        df = dm.download_historical_data(tickers)
        return tickers, df

    with st.spinner("Scansione in tempo reale del mercato (potrebbe richiedere qualche secondo)..."):
        tickers, df_tot = get_market_data()

    if df_tot is not None and not df_tot.empty:
        bench = None
        df_stocks = df_tot.copy()
        if "FTSEMIB.MI" in df_tot.columns:
            bench = df_tot["FTSEMIB.MI"]
            df_stocks = df_tot.drop("FTSEMIB.MI", axis=1)
        
        an = FinancialAnalyzer(df_stocks, bench)
        rets = an.calculate_returns()
        
        if rets.empty:
            st.error("Dati insufficienti per i calcoli.")
            return

        t1 = an.get_table_1_central_metrics()
        t2 = an.get_table_2_risk_extremes()
        t3 = an.get_table_3_non_normality()
        t_jb = an.get_jarque_bera_test()
        
        viz = Visualizer(df_stocks, rets, bench, non_norm_metrics=t3)

        # TAB
        tab1, tab2, tab3 = st.tabs(["Statistiche Avanzate", "Analisi Grafica", "Frontiera Efficiente"])

        # Funzione helper per stile celle FINALE
        def style_final_table(styler):
            styler.set_properties(**{
                'background-color': '#e0e0e0',  # Grigio scuro
                'color': '#2c2c2c',             # Testo scuro
                'border-color': '#ffffff'       # Bordo bianco
            })
            styler.set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#cccccc'), 
                    ('color', '#000000'), 
                    ('font-weight', 'bold'),
                    ('border', '1px solid white')
                ]}
            ])
            return styler

        # --- TAB 1: TABELLE ---
        with tab1:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.subheader("1. Performance e Volatilità")
            st.dataframe(style_final_table(t1.style.format("{:.2%}", subset=['Media Geom. (Ann)', 'Media Giorn.', 'Dev.Std', 'Varianza'])), use_container_width=True)

            st.subheader("2. Analisi del Rischio")
            st.dataframe(style_final_table(t2.style.format("{:.2%}", subset=['Min', 'Max', 'Range', 'Max Drawdown'])
                         .format("{:.4f}", subset=['Cov. Mkt', 'Corr. Mkt'])), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1: 
                st.subheader("3. Asimmetria e Curtosi")
                st.dataframe(style_final_table(t3.style.format("{:.4f}")), use_container_width=True)
            with c2: 
                st.subheader("4. Test di Normalità (Jarque-Bera)")
                st.dataframe(style_final_table(t_jb.style.format({"p-value": "{:.4f}"})), use_container_width=True)

        # --- TAB 2: GRAFICI ---
        with tab2:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Performance Relativa**")
                st.pyplot(viz.plot_normalized_prices())
                st.markdown("---")
                st.write("**Dispersione (Rischio)**")
                st.pyplot(viz.plot_returns_boxplot())
            with col2:
                st.write("**Correlazioni**")
                st.pyplot(viz.plot_correlation_heatmap())
                st.markdown("---")
                st.write("**Distribuzioni**")
                st.pyplot(viz.plot_histogram_grid())

        # --- TAB 3: OTTIMIZZAZIONE ---
        with tab3:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.markdown("### Ottimizzazione di Portafoglio (Markowitz)")
            st.caption("Simulazione Monte Carlo su 5000 portafogli casuali.")
            
            if st.button("🚀 Avvia Ottimizzazione"):
                opt = PortfolioOptimizer(rets)
                res_max, res_min, w_max, w_min = opt.simulate()
                st.session_state.opt_max = res_max
                st.session_state.opt_min = res_min
                st.session_state.opt_w_max = w_max
                st.session_state.opt_w_min = w_min
                st.session_state.opt_obj = opt
                st.session_state.opt_done = True
            
            if st.session_state.opt_done:
                max_pt = st.session_state.opt_max
                min_pt = st.session_state.opt_min
                w_max = st.session_state.opt_w_max
                w_min = st.session_state.opt_w_min
                opt_obj = st.session_state.opt_obj
                
                # BOX METRICHE
                with st.container():
                    st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("🚀 Max Sharpe", f"{max_pt['Rendimento']:.2%}", f"Vol: {max_pt['Volatilità']:.2%}")
                    c2.metric("🛡️ Min Volatility", f"{min_pt['Rendimento']:.2%}", f"Vol: {min_pt['Volatilità']:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)

                col_plot, col_info = st.columns(2)
                
                with col_plot:
                    st.pyplot(opt_obj.plot_efficient_frontier(max_pt, min_pt))
                
                with col_info:
                    st.write("### 🏗️ Allocazione")
                    
                    def make_weight_df(weights, tickers):
                        df_w = pd.DataFrame({'Ticker': tickers, 'Peso': weights})
                        df_w = df_w[df_w['Peso'] > 0.01].sort_values('Peso', ascending=False)
                        return df_w

                    df_w_max = make_weight_df(w_max, rets.columns)
                    df_w_min = make_weight_df(w_min, rets.columns)

                    t1, t2 = st.tabs(["Max Sharpe", "Min Volatility"])
                    
                    col_config = {
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Peso": st.column_config.NumberColumn("Peso (%)", format="%.2f%%", width="small")
                    }

                    with t1:
                        st.dataframe(
                            style_final_table(df_w_max.style),
                            column_config=col_config,
                            use_container_width=False,
                            hide_index=True 
                        )
                    with t2:
                        st.dataframe(
                            style_final_table(df_w_min.style),
                            column_config=col_config,
                            use_container_width=False,
                            hide_index=True
                        )
            else:
                st.info("Clicca sul pulsante per avviare la simulazione.")
    else:
        st.error("Errore critico: Impossibile scaricare i dati. Controlla la connessione internet o le API di Yahoo Finance.")

if __name__ == "__main__":
    main()

