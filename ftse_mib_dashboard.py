import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# --- CONFIGURAZIONE PAGINA E TEMA ---
st.set_page_config(page_title="FTSE MIB Dashboard", layout="wide", page_icon="📈")

# CUSTOM CSS: STILE BIANCO PANNA, MINIMAL E TAB A PULSANTE
st.markdown("""
<style>
    /* Importa font moderno Roboto */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    /* 1. SFONDO BIANCO PANNA PER TUTTA L'APP */
    .stApp {
        background-color: #FDFBF7;
    }
    
    /* 2. TESTI */
    html, body, p, li, div, span, label, h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        color: #484848 !important;
    }
    
    /* 3. TRASFORMAZIONE TAB IN PULSANTI */
    /* Stile base del bottone (Tab inattivo) */
    button[data-baseweb="tab"] {
        background-color: #e0e0e0; /* Grigio chiaro pulsante */
        border-radius: 8px; /* Angoli arrotondati */
        border: none;
        color: #484848;
        font-weight: 600;
        margin-right: 8px; /* Spazio tra i pulsanti */
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    /* Stile bottone attivo (Tab selezionato) */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #484848 !important; /* Grigio scuro attivo */
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Rimuove la linea rossa/sottolineatura di default di Streamlit */
    div[data-testid="stTabs"] > div > div {
        box-shadow: none !important;
        border-bottom: none !important;
        gap: 0px;
    }

    /* 4. TABELLE */
    .stDataFrame {
        border: 1px solid #dcdcdc;
        border-radius: 5px;
    }
    
    /* 5. METRICHE */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #2c2c2c !important;
    }
    
    /* Spaziature */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stProgress > div > div > div > div {
        background-color: #484848;
    }
</style>
""", unsafe_allow_html=True)

# Impostazioni grafici globali
sns.set_theme(style="ticks", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.facecolor'] = '#FDFBF7' 
plt.rcParams['axes.facecolor'] = '#FDFBF7'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['text.color'] = '#484848'
plt.rcParams['axes.labelcolor'] = '#484848'
plt.rcParams['xtick.color'] = '#484848'
plt.rcParams['ytick.color'] = '#484848'

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
        return fig

    def plot_returns_boxplot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=self.returns, ax=ax, palette="light:b", fliersize=3, linewidth=1)
        ax.set_title("Dispersione Rendimenti Giornalieri", fontweight='bold', pad=15)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

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
        return g.fig

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self.returns.corr(), annot=True, cmap='vlag', center=0, fmt=".2f", 
                    ax=ax, cbar_kws={'label': 'Correlazione'}, linewidths=0.5, linecolor='white')
        ax.set_title("Matrice di Correlazione", fontweight='bold', pad=15)
        return fig

# =============================================================================
# CLASSE 4: PORTFOLIO OPTIMIZER
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, returns_df, num_portfolios=3000):
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
        return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    st.title("🇮🇹 FTSE MIB Top 10 Dashboard")
    st.markdown("Analisi finanziaria automatizzata sui top player del mercato italiano.")
    st.markdown("---")
    
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

        # --- TAB 1: TABELLE CON CELLE GRIGIO CHIARO ---
        with tab1:
            # Funzione helper per stile celle grigio chiaro
            def style_gray_cells(styler):
                return styler.set_properties(**{
                    'background-color': '#f4f4f4',  # Grigio chiaro richiesto
                    'color': '#484848',             # Testo scuro
                    'border-color': '#ffffff'       # Bordo bianco tra le celle
                })

            st.subheader("1. Performance e Volatilità")
            st.dataframe(style_gray_cells(t1.style.format("{:.2%}", subset=['Media Geom. (Ann)', 'Media Giorn.', 'Dev.Std', 'Varianza'])))

            st.subheader("2. Analisi del Rischio")
            st.dataframe(style_gray_cells(t2.style.format("{:.2%}", subset=['Min', 'Max', 'Range', 'Max Drawdown'])
                         .format("{:.4f}", subset=['Cov. Mkt', 'Corr. Mkt'])))

            c1, c2 = st.columns(2)
            with c1: 
                st.subheader("3. Asimmetria e Curtosi")
                st.dataframe(style_gray_cells(t3.style.format("{:.4f}")))
            with c2: 
                st.subheader("4. Test di Normalità (Jarque-Bera)")
                st.dataframe(style_gray_cells(t_jb.style.format({"p-value": "{:.4f}"})))

        with tab2:
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

        with tab3:
            st.markdown("### Ottimizzazione di Portafoglio (Markowitz)")
            st.caption("Simulazione Monte Carlo su 3000 portafogli casuali.")
            
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
                
                c1, c2 = st.columns(2)
                c1.metric("🚀 Max Sharpe", f"{max_pt['Rendimento']:.2%}", f"Vol: {max_pt['Volatilità']:.2%}")
                c2.metric("🛡️ Min Volatility", f"{min_pt['Rendimento']:.2%}", f"Vol: {min_pt['Volatilità']:.2%}")
                st.markdown("---")
                
                col_plot, col_info = st.columns(2)
                
                with col_plot:
                    st.pyplot(opt_obj.plot_efficient_frontier(max_pt, min_pt))
                
                with col_info:
                    st.write("### 🏗️ Allocazione")
                    
                    def make_weight_df(weights, tickers):
                        df_w = pd.DataFrame({'Ticker': tickers, 'Peso': weights})
                        df_w = df_w[df_w['Peso'] > 0.01].sort_values('Peso', ascending=False)
                        df_w.set_index('Ticker', inplace=True)
                        return df_w

                    df_w_max = make_weight_df(w_max, rets.columns)
                    df_w_min = make_weight_df(w_min, rets.columns)
                    
                    # Funzione helper per celle grigie anche qui
                    def style_gray_cells_alloc(styler):
                        return styler.set_properties(**{'background-color': '#f4f4f4', 'color': '#484848'})

                    t1, t2 = st.tabs(["Max Sharpe", "Min Volatility"])
                    with t1:
                        st.dataframe(style_gray_cells_alloc(df_w_max.style.format("{:.2%}")))
                    with t2:
                        st.dataframe(style_gray_cells_alloc(df_w_min.style.format("{:.2%}")))
            else:
                st.info("Clicca sul pulsante per avviare la simulazione.")
    else:
        st.error("Errore critico: Impossibile scaricare i dati. Controlla la connessione internet o le API di Yahoo Finance.")

if __name__ == "__main__":
    main()
