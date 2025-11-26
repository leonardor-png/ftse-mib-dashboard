import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import time

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="FTSE MIB Top 10 Analytics", layout="wide", page_icon="📈")
sns.set_theme(style="whitegrid")

# =============================================================================
# CLASSE 1: DATA MANAGER (Adattata per il Web)
# =============================================================================
class DataManager:
    def __init__(self, benchmark="FTSEMIB.MI", start_date="2019-01-01"):
        self.benchmark = benchmark
        self.start_date = start_date

    def _get_mapping(self):
        return {
            "A2A": "A2A.MI", "Amplifon SpA": "AMP.MI", "Assicurazioni Generali": "G.MI",
            "Azimut": "AZM.MI", "Banca Mediolanum": "BMED.MI", "Banca Popolare di Sondrio": "BPSO.MI",
            "Banco Bpm": "BAMI.MI", "BCA MPS": "BMPS.MI", "Bper Banca": "BPE.MI",
            "Brunello Cucinelli SpA": "BC.MI", "Buzzi Unicem": "BZU.MI", "Campari": "CPR.MI",
            "DiaSorin": "DIA.MI", "Enel": "ENEL.MI", "Eni SpA": "ENI.MI", "Ferrari NV": "RACE.MI",
            "FinecoBank": "FBK.MI", "Hera SpA": "HER.MI", "Interpump Group": "IP.MI",
            "Intesa Sanpaolo": "ISP.MI", "Inwit": "INW.MI", "Italgas": "IG.MI", "Iveco NV": "IVG.MI",
            "Leonardo": "LDO.MI", "Lottomatica": "LTMC.MI", "Mediobanca": "MB.MI",
            "Moncler SpA": "MONC.MI", "Nexi": "NEXI.MI", "Poste Italiane": "PST.MI",
            "Prysmian": "PRY.MI", "Recordati": "REC.MI", "Saipem": "SPM.MI", "Snam Rete": "SRG.MI",
            "Stellantis NV": "STLAM.MI", "STMicro": "STM.MI", "Telecom Italia": "TIT.MI",
            "Tenaris": "TEN.MI", "Terna": "TRN.MI", "UniCredit": "UCG.MI", "Unipol Gruppo": "UNI.MI"
        }

    def _get_shares_db(self):
        # Database Azioni Circolanti (Backup tecnico per calcolo Market Cap)
        return {
            "UCG.MI": 1_620_000_000, "ISP.MI": 18_280_000_000, "ENEL.MI": 10_160_000_000,
            "ENI.MI": 3_260_000_000, "G.MI": 1_550_000_000, "STLAM.MI": 2_890_000_000,
            "STM.MI": 910_000_000, "TEN.MI": 590_000_000, "RACE.MI": 182_000_000,
            "PST.MI": 1_300_000_000, "LDO.MI": 578_000_000, "TRN.MI": 2_000_000_000,
            "SRG.MI": 3_360_000_000, "PRY.MI": 280_000_000, "BMPS.MI": 1_250_000_000,
            "MONC.MI": 270_000_000, "BPE.MI": 1_410_000_000, "CNHI.MI": 1_300_000_000,
            "IVG.MI": 270_000_000
        }

    def get_top_10_tickers(self):
        """Calcola la Top 10 in tempo reale."""
        mapping = self._get_mapping()
        shares_db = self._get_shares_db()
        all_tickers = list(mapping.values())
        
        market_caps = {}
        
        try:
            # Batch Download dei prezzi attuali (evita il ban di Yahoo)
            batch_data = yf.download(all_tickers, period="1d", progress=False)['Close']
            if batch_data.empty: return []
            last_prices = batch_data.iloc[-1]
            
            for ticker in all_tickers:
                try:
                    prezzo = last_prices.get(ticker)
                    azioni = shares_db.get(ticker, 0) # Usa DB interno per velocità e sicurezza
                    
                    # Se non nel DB interno, prova il metodo lento (solo se necessario)
                    if azioni == 0:
                         fi = yf.Ticker(ticker).fast_info
                         azioni = fi.get('shares')
                    
                    if azioni and prezzo and not np.isnan(prezzo):
                        market_caps[ticker] = prezzo * azioni
                except: continue

            # Ordinamento e Selezione
            sorted_caps = dict(sorted(market_caps.items(), key=lambda item: item[1], reverse=True))
            return list(sorted_caps.keys())[:10]

        except Exception as e:
            st.error(f"Errore calcolo classifica: {e}")
            return []

    def download_historical_data(self, tickers):
        """Scarica i prezzi storici per i grafici."""
        full_list = tickers + [self.benchmark]
        try:
            raw_data = yf.download(full_list, start=self.start_date, auto_adjust=False, progress=False)
            
            if 'Adj Close' in raw_data.columns: data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns: data = raw_data['Close']
            else: data = raw_data.xs('Adj Close', level=0, axis=1)
            
            return data.dropna()
        except Exception as e:
            st.error(f"Errore download storico: {e}")
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
            self.bench_r = self.bench_p.pct_change().dropna()
            # Allinea le date
            idx = self.returns.index.intersection(self.bench_r.index)
            self.returns = self.returns.loc[idx]
            self.bench_r = self.bench_r.loc[idx]
        return self.returns

    def _calc_max_drawdown(self, series):
        comp = (1 + series).cumprod()
        peak = comp.expanding(min_periods=1).max()
        return ((comp/peak) - 1).min()

    def _calc_cagr(self, p):
        if len(p) < 2: return 0
        years = (p.index[-1] - p.index[0]).days / 365.25
        return (p.iloc[-1] / p.iloc[0]) ** (1/years) - 1 if years > 0 else 0

    def _prepare_stats_dataframe(self):
        df_calc = self.returns.copy()
        if self.bench_r is not None:
            bench_series = self.bench_r.iloc[:, 0] if isinstance(self.bench_r, pd.DataFrame) else self.bench_r
            bench_series.name = "FTSE MIB (Bench)"
            df_calc = pd.concat([df_calc, bench_series], axis=1)
        return df_calc

    def get_table_1_central_metrics(self):
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['median', 'std', 'var', 'mean']).T
        stats_df['Media Geom. (Ann)'] = df_calc.apply(lambda x: stats.gmean(x + 1)**252 - 1)
        stats_df['Moda (Approx)'] = df_calc.apply(lambda x: x.round(4).mode()[0] if not x.mode().empty else np.nan)
        stats_df.rename(columns={'mean': 'Media Giorn.', 'median': 'Mediana', 'std': 'Dev.Std', 'var': 'Varianza'}, inplace=True)
        return stats_df[['Media Geom. (Ann)', 'Moda (Approx)', 'Mediana', 'Dev.Std', 'Varianza']]

    def get_table_2_risk_extremes(self):
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['min', 'max']).T
        stats_df['Range'] = stats_df['max'] - stats_df['min']
        stats_df['Max Drawdown'] = df_calc.apply(self._calc_max_drawdown)
        
        # Calcolo Rischio Relativo
        if self.bench_r is not None:
            bench_s = self.bench_r.iloc[:, 0] if isinstance(self.bench_r, pd.DataFrame) else self.bench_r
            mkt_var = bench_s.var()
            for col in self.returns.columns: 
                cov = self.returns[col].cov(bench_s)
                stats_df.loc[col, 'Cov. Mkt'] = cov
                stats_df.loc[col, 'Corr. Mkt'] = self.returns[col].corr(bench_s)
            stats_df.loc['FTSE MIB (Bench)', ['Cov. Mkt', 'Corr. Mkt']] = np.nan
        
        stats_df.rename(columns={'min': 'Min', 'max': 'Max'}, inplace=True)
        return stats_df[['Min', 'Max', 'Range', 'Max Drawdown', 'Cov. Mkt', 'Corr. Mkt']]

    def get_table_3_non_normality(self):
        df_calc = self._prepare_stats_dataframe()
        non_norm = df_calc.agg(['skew', 'kurt']).T
        non_norm.rename(columns={'skew': 'Asimmetria (Skew)', 'kurt': 'Curtosi (Excess Kurt)'}, inplace=True)
        return non_norm

    def get_jarque_bera_test(self):
        res = pd.DataFrame({'p-value': self.returns.apply(lambda x: stats.jarque_bera(x)[1])})
        res['Esito'] = np.where(res['p-value'] > 0.05, "NORMALE", "NON NORMALE")
        return res

# =============================================================================
# CLASSE 3: VISUALIZER (Restituisce Figure per Streamlit)
# =============================================================================
class Visualizer:
    def __init__(self, prices, returns, benchmark=None, non_norm_metrics=None):
        self.prices, self.returns, self.bench = prices, returns, benchmark
        self.non_norm_metrics = non_norm_metrics

    def plot_normalized_prices(self):
        norm = (self.prices / self.prices.iloc[0]) * 100
        fig, ax = plt.subplots(figsize=(14, 6))
        
        cmap = plt.cm.get_cmap('tab10') 
        for i, c in enumerate(norm.columns):
            ax.plot(norm.index, norm[c], label=c, alpha=0.8, linewidth=2, color=cmap(i/len(norm.columns)))
        
        if self.bench is not None:
            bn = (self.bench / self.bench.iloc[0]) * 100
            ax.plot(bn.index, bn, label="FTSE MIB (Bench)", color='black', ls='--', lw=3)

        ax.yaxis.set_major_locator(mticker.MultipleLocator(100))
        ax.axhline(y=100, color='gray', ls='-', lw=1)
        ax.set_title("Andamento dei prezzi dei top10 del FTSE MIB", fontweight='bold', fontsize=16)
        ax.set_ylabel("Valore (Base 100)", fontsize=12)
        ax.legend(loc='upper left', title='Confronto su Base 100', bbox_to_anchor=(1.01, 1), fontsize='small')
        plt.tight_layout()
        return fig

    def plot_returns_boxplot(self):
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=self.returns, ax=ax)
        ax.set_title("Rischio Relativo e Outlier: Confronto della Volatilità Giornaliera", fontweight='bold', fontsize=16)
        ax.set_ylabel("Rendimento Giornaliero")
        plt.figtext(0.15, 0.9, "L'altezza della scatola indica la deviazione standard (rischio) tipica.", 
                    fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_histogram_grid(self):
        TICKER_TO_NAME = {'UCG.MI': 'UniCredit', 'ISP.MI': 'Intesa', 'ENEL.MI': 'Enel', 'RACE.MI': 'Ferrari', 
                          'G.MI': 'Generali', 'ENI.MI': 'Eni', 'STLAM.MI': 'Stellantis', 'BMPS.MI': 'MPS', 
                          'PST.MI': 'Poste', 'LDO.MI': 'Leonardo', 'FTSE MIB (Bench)': 'Mercato'}
        
        df_combined = self.returns.copy()
        if self.bench is not None:
             bench_r = self.bench.pct_change().dropna().rename('FTSE MIB (Bench)')
             df_combined['FTSE MIB (Bench)'] = bench_r
             df_combined = df_combined.dropna()
        
        melt = df_combined.melt(var_name='Ticker', value_name='Rendimento')
        
        if self.non_norm_metrics is not None:
             df_stats = self.non_norm_metrics.copy().reset_index().rename(columns={'index': 'Ticker'})
             melt = melt.merge(df_stats, on='Ticker', how='left')
             melt['FullLabel'] = melt.apply(
                lambda row: f"{row['Ticker']} | S:{float(row.get('Asimmetria (Skew)', 0)):.2f} | K:{float(row.get('Curtosi (Excess Kurt)', 0)):.2f}",
                axis=1)
        else:
             melt['FullLabel'] = melt['Ticker'].apply(lambda x: f"{x} | {TICKER_TO_NAME.get(x, x)}")

        g = sns.FacetGrid(melt, col="FullLabel", col_wrap=4, sharex=False, sharey=False)
        g.map_dataframe(sns.kdeplot, x="Rendimento", color="darkred", lw=2.5, alpha=1.0, bw_adjust=0.5)
        g.map_dataframe(sns.histplot, x="Rendimento", kde=False, stat="density", color="skyblue", edgecolor="darkblue", alpha=0.7)
        
        g.set_titles("{col_name}", size=10) 
        for ax in g.axes.flat: ax.set(ylabel="", xlabel="")
        
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Distribuzione Rendimenti (Stock vs Mercato)', fontweight='bold')
        plt.figtext(0.98, 0.01, "LEGENDA:\nBarre Blu: Densità di Probabilità\nCurva Rossa: Stima della Densità", 
                    transform=plt.gcf().transFigure, ha='right', fontsize=10, bbox={"facecolor":"white", "alpha":0.8})
        return g.fig

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Matrice di correlazione top 10 FTSE MIB", fontweight='bold')
        return fig

# =============================================================================
# CLASSE 4: PORTFOLIO OPTIMIZER
# =============================================================================
class PortfolioOptimizer:
    def __init__(self, returns_df, num_portfolios=5000):
        self.ret = returns_df
        self.n = num_portfolios
        self.results = None
        self.weights = None

    def simulate(self):
        np.random.seed(42)
        mean, cov = self.ret.mean() * 252, self.ret.cov() * 252
        n_assets = len(self.ret.columns)
        w = np.random.random((self.n, n_assets)); w /= w.sum(axis=1)[:, None]
        
        r_arr = w @ mean.values
        v_arr = np.sqrt(np.einsum('ij,ji->i', w @ cov.values, w.T))
        sharpe = r_arr / v_arr
        
        self.results = pd.DataFrame({'Rendimento': r_arr, 'Volatilità': v_arr, 'Sharpe': sharpe})
        self.weights = w
        return self.results.iloc[sharpe.argmax()], self.results.iloc[v_arr.argmin()]

    def plot_efficient_frontier(self, max_pt, min_pt):
        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(self.results['Volatilità'], self.results['Rendimento'], c=self.results['Sharpe'], cmap='viridis', s=10, alpha=0.5)
        plt.colorbar(sc, label='Sharpe Ratio')
        ax.scatter(max_pt['Volatilità'], max_pt['Rendimento'], c='red', s=200, marker='*', label='Max Sharpe')
        ax.scatter(min_pt['Volatilità'], min_pt['Rendimento'], c='blue', s=200, marker='*', label='Min Volatility')
        ax.set_title("Frontiera Efficiente di Markowitz", fontweight='bold')
        ax.set_xlabel("Rischio (Volatilità Annua)"); ax.set_ylabel("Rendimento Atteso Annuo")
        ax.legend(loc='lower right')
        return fig

# =============================================================================
# APP STREAMLIT
# =============================================================================
def main():
    st.title("🇮🇹 FTSE MIB Top 10 - Analisi Avanzata")
    st.markdown("Dashboard finanziaria interattiva per l'analisi dei top 10 titoli italiani.")

    # --- LOGICA CACHING: Evita di riscaricare se non necessario ---
    @st.cache_data(ttl=3600) # Cache valida per 1 ora
    def get_market_data():
        dm = DataManager()
        tickers = dm.get_top_10_tickers()
        if not tickers: return None, None
        df = dm.download_historical_data(tickers)
        return tickers, df

    with st.spinner("Connessione ai mercati finanziari in corso..."):
        tickers, df_tot = get_market_data()

    if df_tot is not None and not df_tot.empty:
        # Separazione Benchmark
        bench = None
        df_stocks = df_tot.copy()
        if "FTSEMIB.MI" in df_tot.columns:
            bench = df_tot["FTSEMIB.MI"]
            df_stocks = df_tot.drop("FTSEMIB.MI", axis=1)
        
        # Calcoli
        an = FinancialAnalyzer(df_stocks, bench)
        rets = an.calculate_returns()
        
        t1 = an.get_table_1_central_metrics()
        t2 = an.get_table_2_risk_extremes()
        t3 = an.get_table_3_non_normality()
        t_jb = an.get_jarque_bera_results()
        
        viz = Visualizer(df_stocks, rets, bench, non_norm_metrics=t3)

        # --- LAYOUT TABS ---
        tab1, tab2, tab3 = st.tabs(["📊 Report Statistico", "📈 Analisi Grafica", "🧠 Portafoglio Ottimale"])

        with tab1:
            st.subheader("1. Centralità e Dispersione")
            st.dataframe(t1.style.format("{:.4f}"))
            
            st.subheader("2. Estremi e Rischio Relativo")
            st.dataframe(t2.style.format("{:.4f}"))
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("3. Asimmetria e Curtosi")
                st.dataframe(t3.style.format("{:.4f}"))
            with col_b:
                st.subheader("4. Test di Normalità (Jarque-Bera)")
                st.dataframe(t_jb)

        with tab2:
            st.subheader("Andamento Temporale")
            st.pyplot(viz.plot_normalized_prices())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Volatilità (Box Plot)")
                st.pyplot(viz.plot_returns_boxplot())
            with col2:
                st.subheader("Correlazioni")
                st.pyplot(viz.plot_correlation_heatmap())
                
            st.subheader("Distribuzione dei Rendimenti")
            st.pyplot(viz.plot_histogram_grid())

        with tab3:
            st.subheader("Frontiera Efficiente di Markowitz")
            st.write("Simulazione Monte Carlo su 5.000 portafogli casuali.")
            
            if st.button("Avvia Ottimizzazione"):
                opt = PortfolioOptimizer(rets)
                max_pt, min_pt = opt.simulate()
                
                c1, c2 = st.columns(2)
                c1.metric("🚀 Max Sharpe Rendimento", f"{max_pt['Rendimento']:.2%}", f"Vol: {max_pt['Volatilità']:.2%}")
                c2.metric("🛡️ Min Volatility Rendimento", f"{min_pt['Rendimento']:.2%}", f"Vol: {min_pt['Volatilità']:.2%}")
                
                st.pyplot(opt.plot_efficient_frontier(max_pt, min_pt))
    else:
        st.error("Errore critico: Impossibile recuperare i dati da Yahoo Finance.")

if __name__ == "__main__":
    main()