import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="FTSE MIB Top 10 Analytics", layout="wide", page_icon="📈")
sns.set_theme(style="whitegrid")

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

    def _get_shares_db(self):
        # Database manuale per fallback (numero azioni approssimativo)
        return {
            "UCG.MI": 1_620_000_000, "ISP.MI": 18_280_000_000, "ENEL.MI": 10_160_000_000,
            "ENI.MI": 3_260_000_000, "G.MI": 1_550_000_000, "STLAM.MI": 2_890_000_000,
            "STM.MI": 910_000_000, "TEN.MI": 590_000_000, "RACE.MI": 182_000_000,
            "PST.MI": 1_300_000_000, "LDO.MI": 578_000_000, "TRN.MI": 2_000_000_000,
            "SRG.MI": 3_360_000_000, "PRY.MI": 280_000_000, "BMPS.MI": 1_250_000_000,
            "MONC.MI": 270_000_000, "BPE.MI": 1_410_000_000, "IVG.MI": 270_000_000
        }

    def get_top_10_tickers(self):
        """Calcola la Top 10 per capitalizzazione."""
        mapping = self._get_mapping()
        shares_db = self._get_shares_db()
        all_tickers = list(mapping.values())
        market_caps = {}
        
        try:
            # Scarica solo l'ultimo prezzo di chiusura
            batch_data = yf.download(all_tickers, period="1d", progress=False)
            
            # Gestione nuova struttura dati yfinance (MultiIndex)
            if 'Close' in batch_data.columns:
                last_prices = batch_data['Close'].iloc[-1]
            else:
                last_prices = batch_data.iloc[-1] # Fallback
            
            for ticker in all_tickers:
                try:
                    price = last_prices.get(ticker)
                    if pd.isna(price): continue
                    
                    # Usa DB interno per velocità
                    shares = shares_db.get(ticker, 0)
                    
                    # Se non nel DB, prova info (lento, ma necessario se manca)
                    if shares == 0:
                        try:
                            shares = yf.Ticker(ticker).fast_info.get('shares', 0)
                        except:
                            shares = 0
                    
                    if shares and price:
                        market_caps[ticker] = price * shares
                except:
                    continue

            # Ordina e prendi i primi 10
            sorted_caps = dict(sorted(market_caps.items(), key=lambda item: item[1], reverse=True))
            return list(sorted_caps.keys())[:10]

        except Exception as e:
            st.error(f"Errore nel calcolo della classifica: {e}")
            return []

    def download_historical_data(self, tickers):
        """Scarica i prezzi storici."""
        full_list = tickers + [self.benchmark]
        try:
            raw_data = yf.download(full_list, start=self.start_date, progress=False)
            
            # Gestione robusta per 'Adj Close' o 'Close'
            if 'Adj Close' in raw_data.columns:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns:
                data = raw_data['Close']
            else:
                # Tentativo di recupero se multi-index complesso
                try:
                    data = raw_data.xs('Adj Close', level=0, axis=1)
                except:
                    data = raw_data.xs('Close', level=0, axis=1)
            
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
            # Assicuriamoci che sia una Series
            if isinstance(self.bench_p, pd.DataFrame):
                self.bench_p = self.bench_p.iloc[:, 0]
            
            self.bench_r = self.bench_p.pct_change().dropna()
            
            # Allinea le date (Intersezione)
            idx = self.returns.index.intersection(self.bench_r.index)
            self.returns = self.returns.loc[idx]
            self.bench_r = self.bench_r.loc[idx]
        return self.returns

    def _calc_max_drawdown(self, series):
        comp = (1 + series).cumprod()
        peak = comp.expanding(min_periods=1).max()
        if peak.empty: return 0.0
        dd = (comp/peak) - 1
        return dd.min()

    def _prepare_stats_dataframe(self):
        df_calc = self.returns.copy()
        if self.bench_r is not None:
            bench_s = self.bench_r.copy()
            bench_s.name = "FTSE MIB (Bench)"
            df_calc = pd.concat([df_calc, bench_s], axis=1)
        return df_calc

    def get_table_1_central_metrics(self):
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['median', 'std', 'var', 'mean']).T
        
        # Rendimento geometrico annualizzato
        stats_df['Media Geom. (Ann)'] = df_calc.apply(lambda x: (stats.gmean(x + 1)**252 - 1) if len(x) > 0 else 0)
        
        stats_df.rename(columns={'mean': 'Media Giorn.', 'median': 'Mediana', 'std': 'Dev.Std', 'var': 'Varianza'}, inplace=True)
        return stats_df[['Media Geom. (Ann)', 'Media Giorn.', 'Mediana', 'Dev.Std', 'Varianza']]

    def get_table_2_risk_extremes(self):
        df_calc = self._prepare_stats_dataframe()
        stats_df = df_calc.agg(['min', 'max']).T
        stats_df['Range'] = stats_df['max'] - stats_df['min']
        stats_df['Max Drawdown'] = df_calc.apply(self._calc_max_drawdown)
        
        # Calcolo Rischio Relativo (Covarianza col Mercato)
        if self.bench_r is not None:
            bench_s = self.bench_r
            for col in self.returns.columns: 
                cov = self.returns[col].cov(bench_s)
                corr = self.returns[col].corr(bench_s)
                stats_df.loc[col, 'Cov. Mkt'] = cov
                stats_df.loc[col, 'Corr. Mkt'] = corr
            
            # Setta NaN per il benchmark stesso
            stats_df.loc['FTSE MIB (Bench)', ['Cov. Mkt', 'Corr. Mkt']] = np.nan
        
        stats_df.rename(columns={'min': 'Min', 'max': 'Max'}, inplace=True)
        return stats_df[['Min', 'Max', 'Range', 'Max Drawdown', 'Cov. Mkt', 'Corr. Mkt']]

    def get_table_3_non_normality(self):
        df_calc = self._prepare_stats_dataframe()
        non_norm = df_calc.agg(['skew', 'kurt']).T
        non_norm.rename(columns={'skew': 'Asimmetria (Skew)', 'kurt': 'Curtosi (Excess Kurt)'}, inplace=True)
        return non_norm

    # --- CORREZIONE QUI: Rinominato per coerenza ---
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
        # Base 100
        norm = (self.prices / self.prices.iloc[0]) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Colormap personalizzata
        colors = sns.color_palette("husl", len(norm.columns))
        
        for i, c in enumerate(norm.columns):
            ax.plot(norm.index, norm[c], label=c, alpha=0.7, linewidth=1.5, color=colors[i])
        
        if self.bench is not None:
            bn = (self.bench / self.bench.iloc[0]) * 100
            ax.plot(bn.index, bn, label="FTSE MIB", color='black', ls='--', lw=2.5)

        ax.set_title("Performance Relativa (Base 100)", fontweight='bold')
        ax.set_ylabel("Valore")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_

    main()
