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

# --- CUSTOM CSS (STILE FORZATO) ---
st.markdown("""
<style>
    /* Importa font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* 1. SFONDO GENERALE */
    .stApp {
        background-color: #FFFDE7; 
    }
    
    /* 2. TESTI GLOBALI */
    html, body, p, li, div, span, label, h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: #000000 !important;
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
        color: #000000 !important;
    }

    /* 4. TABELLE - STILE IMPERATIVO */
    div[data-testid="stTable"] table thead th, 
    div[data-testid="stTable"] table tbody th {
        background-color: #999999 !important; /* GRIGIO SCURO */
        color: #000000 !important;            /* TESTO NERO */
        font-weight: 900 !important;          /* Grassetto */
        border: 1px solid #ffffff !important; /* Bordo bianco */
        text-align: center !important;
    }
    div[data-testid="stTable"] table tbody td {
        background-color: #eeeeee !important; /* Grigio Chiaro */
        color: #000000 !important;            /* Testo NERO */
        border: 1px solid #ffffff !important;
    }
    div[data-testid="stTable"] {
        border: 1px solid #999999;
        border-radius: 4px;
        overflow: hidden;
    }

    /* 5. STILE PULSANTI TAB */
    button[data-baseweb="tab"] {
        background-color: #e0e0e0; 
        border: 1px solid #d1d1d1;
        color: #000000;
        font-weight: bold;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #c0c0c0 !important;
        border: 1px solid #666 !important;
        color: #000000 !important;
    }
    div[data-testid="stTabs"] > div > div {
        box-shadow: none !important;
        border-bottom: none !important;
        gap: 0px;
    }

    /* 6. PULSANTI STANDARD */
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

    /* 7. BOX METRICHE */
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
    
    /* DIVISORE */
    hr.custom-divider {
        margin-top: 0px;
        margin-bottom: 25px;
        border: 0;
        border-top: 2px solid #999;
    }
    
    /* BOX DESCRIZIONE GRAFICO + LEGENDA */
    .chart-desc-container {
        background-color: #ffffff;
        border-left: 4px solid #999;
        padding: 15px;
        border-radius: 0 5px 5px 0;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
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
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAZIONE GRAFICI ---
sns.set_theme(style="ticks", context="notebook")
plt.rcParams['figure.figsize'] = (6.0, 3.5)
plt.rcParams['figure.facecolor'] = '#FFFDE7' 
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1

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

    def get_ticker_to_name_mapping(self):
        original_map = self._get_mapping()
        return {v: k for k, v in original_map.items()}

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
                except: continue
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
        except: return pd.DataFrame()

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
# CLASSE 3: VISUALIZER (SENZA TITOLI)
# =============================================================================
class Visualizer:
    def __init__(self, prices, returns, benchmark=None, non_norm_metrics=None):
        self.prices, self.returns, self.bench = prices, returns, benchmark
        self.non_norm_metrics = non_norm_metrics

    def _add_border(self, fig):
        fig.patch.set_linewidth(1.5)
        fig.patch.set_edgecolor('#999999')
        return fig

    def plot_normalized_prices(self):
        norm = (self.prices / self.prices.iloc[0]) * 100
        fig, ax = plt.subplots() 
        colors = sns.color_palette("husl", len(norm.columns))
        for i, c in enumerate(norm.columns):
            ax.plot(norm.index, norm[c], label=c, alpha=0.9, linewidth=1.5, color=colors[i])
        if self.bench is not None:
            bn = (self.bench / self.bench.iloc[0]) * 100
            ax.plot(bn.index, bn, label="FTSE MIB", color='#000000', ls='--', lw=2.5)
        
        ax.set_xlabel("")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize='small')
        plt.tight_layout()
        return self._add_border(fig)

    def plot_returns_boxplot(self):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.returns, ax=ax, palette="light:b", fliersize=3, linewidth=1)
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
        
        g = sns.FacetGrid(melt, col="Ticker", col_wrap=3, sharex=False, sharey=False, height=1.3, aspect=1.3)
        g.map_dataframe(sns.histplot, x="Rendimento", kde=True, color="#778899", edgecolor="white", linewidth=0.5)
        g.set_titles("{col_name}", fontweight='bold')
        
        g.set_axis_labels("", "")
        for ax in g.axes.flat:
            ax.set_xlabel("")
            ax.set_ylabel("")
            
        g.despine(left=True)
        g.fig.patch.set_linewidth(1.5)
        g.fig.patch.set_edgecolor('#999999')
        return g.fig

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots()
        sns.heatmap(self.returns.corr(), annot=True, cmap='vlag', center=0, fmt=".2f", 
                    ax=ax, cbar_kws={'label': 'Correlazione'}, linewidths=0.5, linecolor='white', annot_kws={"size": 7})
        return self._add_border(fig)

# =============================================================================
# CLASSE 4: PORTFOLIO OPTIMIZER
# =============================================================================
class PortfolioOptimizer:
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
        fig.patch.set_edgecolor('#999999')
        return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    col_title, col_btn = st.columns([5, 1])
    
    with col_title:
        st.markdown("""
            <div class="title-card">
                <h1>🇮🇹 FTSE MIB Top 10 Dashboard</h1>
                <p>Analisi finanziaria automatizzata sui top player del mercato italiano.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_btn:
        st.write("") 
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
        mapping = dm.get_ticker_to_name_mapping()
        return tickers, df, mapping

    with st.spinner("Scansione in tempo reale del mercato (potrebbe richiedere qualche secondo)..."):
        data_result = get_market_data()
        if data_result is None or data_result[0] is None:
            st.error("Errore recupero dati.")
            return
        tickers, df_tot, ticker_mapping = data_result

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

        tab1, tab2, tab3 = st.tabs(["Statistiche Avanzate", "Analisi Grafica", "Frontiera Efficiente"])

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

            # 1. Performance Relativa
            render_plot_with_description(
                "Performance Relativa", 
                viz.plot_normalized_prices(),
                "Questo grafico normalizza il prezzo di tutti i titoli a quota 100 all'inizio del periodo. "
                "Permette di confrontare la <b>performance percentuale cumulativa</b> indipendentemente dal valore nominale dell'azione.<br><br>"
                "Ad esempio, se una linea raggiunge 110, significa che il titolo ha guadagnato il 10% nel periodo osservato.",
                current_tickers, ticker_mapping
            )

            # 2. Boxplot
            render_plot_with_description(
                "Dispersione (Rischio)", 
                viz.plot_returns_boxplot(),
                "Il Boxplot visualizza la distribuzione dei rendimenti giornalieri.<br><br>"
                "<b>Scatola centrale:</b> Contiene il 50% dei dati centrali. Più è alta, più il titolo è volatile.<br>"
                "<b>Linea interna:</b> Mediana dei rendimenti.<br>"
                "<b>Punti esterni (Baffi):</b> Eventi estremi o anomali (outlier) positivi o negativi.",
                current_tickers, ticker_mapping
            )

            # 3. Heatmap
            render_plot_with_description(
                "Correlazioni", 
                viz.plot_correlation_heatmap(),
                "Mostra come i titoli si muovono l'uno rispetto all'altro.<br><br>"
                "<b>+1 (Rosso):</b> Si muovono in modo identico.<br>"
                "<b>0 (Bianco):</b> Nessuna relazione.<br>"
                "<b>-1 (Blu):</b> Si muovono in direzioni opposte.<br><br>"
                "Utile per la diversificazione del portafoglio.",
                current_tickers, ticker_mapping
            )

            # 4. Histograms
            render_plot_with_description(
                "Distribuzioni", 
                viz.plot_histogram_grid(),
                "Mostra la frequenza dei rendimenti giornalieri per ogni titolo.<br><br>"
                "<b>Campana simmetrica:</b> Distribuzione Normale (prevedibile).<br>"
                "<b>Code lunghe (Fat Tails):</b> Indicano che eventi estremi (crolli o boom) sono più frequenti del previsto.<br>"
                "<b>Sbilanciamento:</b> Tendenza a rendimenti positivi o negativi.",
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

