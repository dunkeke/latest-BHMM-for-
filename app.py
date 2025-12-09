import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import akshare as ak
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import concurrent.futures
from tqdm import tqdm
import pickle
from typing import List, Tuple, Dict, Optional

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="BHMM A-Share Pro Plus",
    page_icon="🇨🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# 保持"彭博风"样式
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 31, 46, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px; border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; color: #E0E0E0; }
    div.stButton > button {
        background: linear-gradient(90deg, #D32F2F 0%, #FF5252 100%);
        color: white; border: none; font-weight: 600;
    }
    .scanner-card {
        background-color: rgba(33, 37, 41, 0.8);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #555;
    }
    .state-0 { border-left-color: #00E676 !important; }
    .state-1 { border-left-color: #FFD600 !important; }
    .state-2 { border-left-color: #FF1744 !important; }
    .state-3 { border-left-color: #AA00FF !important; }
    .positive-alpha { background: linear-gradient(90deg, rgba(0, 230, 118, 0.1), rgba(0, 230, 118, 0.05)) !important; }
    .negative-alpha { background: linear-gradient(90deg, rgba(255, 23, 68, 0.1), rgba(255, 23, 68, 0.05)) !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 基础工具函数 (通用)
# ==========================================

@st.cache_data(ttl=24*3600)
def get_all_a_share_list():
    """获取全市场列表"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[['代码', '名称', '总市值', '成交额', '涨跌幅']]
        # 清理数据
        df['总市值'] = pd.to_numeric(df['总市值'].str.replace('亿', '').str.replace(',', ''), errors='coerce')
        df['成交额'] = pd.to_numeric(df['成交额'].str.replace('亿', '').str.replace(',', ''), errors='coerce') * 10000  # 转换为万元
        df['Display'] = df['代码'] + " | " + df['名称'] + " | 市值:" + df['总市值'].round(2).astype(str) + "亿"
        return df, True
    except Exception as e:
        st.error(f"获取市场数据失败: {str(e)}")
        return pd.DataFrame(), False

@st.cache_data(ttl=3600)
def format_ticker_for_yfinance(raw_code: str, raw_name: str = "Unknown") -> Tuple[str, str]:
    raw_code = str(raw_code).strip()
    if raw_code.startswith("6") or raw_code.startswith("9"): 
        suffix = ".SS"
    elif raw_code.startswith("0") or raw_code.startswith("3"): 
        suffix = ".SZ"
    elif raw_code.startswith("4") or raw_code.startswith("8"): 
        suffix = ".BJ"
    else: 
        suffix = ".SS"
    return f"{raw_code}{suffix}", raw_name

@st.cache_data(ttl=3600)
def get_sector_components(sector_name: str, top_n: int = 20) -> List[Tuple[str, str]]:
    """获取板块成分股 (按市值排序)"""
    try:
        df = ak.stock_board_industry_name_em()
        if sector_name not in df['板块名称'].values:
            return []
        
        board_code = df[df['板块名称'] == sector_name]['板块代码'].values[0]
        cons = ak.stock_board_industry_cons_em(symbol=board_code)
        
        # 按市值排序
        if '总市值' in cons.columns:
            # 清理市值数据
            cons['总市值_clean'] = pd.to_numeric(
                cons['总市值'].str.replace('亿', '').str.replace(',', ''), 
                errors='coerce'
            )
            cons = cons.sort_values('总市值_clean', ascending=False)
        
        top_n = min(top_n, len(cons))
        result = []
        for i in range(top_n):
            try:
                code = str(cons.iloc[i]['代码']).strip()
                name = str(cons.iloc[i]['名称']).strip()
                if code and name:
                    result.append((code, name))
            except:
                continue
        
        return result
    except Exception as e:
        st.error(f"获取板块成分股失败: {str(e)}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker: str, start: str, end: str, use_cache: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
    """获取股票数据"""
    # 检查缓存
    cache_key = f"{ticker}_{start}_{end}"
    cache_dir = ".data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['df'], cached_data['ticker']
        except:
            pass
    
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        
        # 自动纠错后缀
        if df.empty or len(df) < 10:
            base_code = ticker.split('.')[0]
            if len(ticker.split('.')) > 1:
                current_suffix = '.' + ticker.split('.')[1]
                alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
                alt_ticker = base_code + alt_suffix
                df = yf.download(alt_ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
                if not df.empty and len(df) > 10:
                    ticker = alt_ticker
        
        if isinstance(df.columns, pd.MultiIndex):
            try: 
                df.columns = df.columns.get_level_values(0)
            except: 
                pass 
            
        if len(df) < 60: 
            return None, ticker
        
        if 'Close' not in df.columns: 
            return None, ticker
        
        # 特征工程
        data = df[['Close', 'High', 'Low', 'Volume']].copy()
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
        data['Vol_Change'] = (data['Volume'] - data['Volume'].rolling(window=5).mean()) / data['Volume'].rolling(window=5).mean()
        data.dropna(inplace=True)
        
        # 缓存数据
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'df': data, 'ticker': ticker}, f)
            except:
                pass
        
        return data, ticker
    except Exception as e:
        return None, ticker

@st.cache_data(ttl=3600, show_spinner=False)
def batch_download_data(tickers_list: List[Tuple[str, str]], start: str, end: str) -> Dict:
    """批量下载数据"""
    data_dict = {}
    if not tickers_list: 
        return data_dict
    
    # 分批处理，避免请求过大
    batch_size = 30
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i+batch_size]
        
        # 准备yfinance格式的tickers
        yf_tickers = []
        mapping = {}
        
        for code, name in batch:
            yf_code, _ = format_ticker_for_yfinance(code, name)
            yf_tickers.append(yf_code)
            mapping[yf_code] = (code, name)
        
        try:
            if len(yf_tickers) == 1:
                df_all = yf.download(yf_tickers[0], start=start, end=end, 
                                   interval="1d", auto_adjust=True, progress=False)
                if not df_all.empty:
                    ticker = yf_tickers[0]
                    df = df_all.copy()
                    df.dropna(how='all', inplace=True)
                    if len(df) >= 60:
                        # 特征工程
                        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
                        df['Volatility'] = df['Log_Ret'].rolling(window=20).std()
                        df.dropna(inplace=True)
                        
                        original_code, name = mapping[ticker]
                        data_dict[original_code] = {"data": df, "name": name}
            else:
                df_all = yf.download(" ".join(yf_tickers), start=start, end=end, 
                                   interval="1d", group_by='ticker', auto_adjust=True, 
                                   progress=False, threads=True)
                
                for t in yf_tickers:
                    try:
                        # 提取单个股票数据
                        df = df_all[t].copy() if isinstance(df_all.columns, pd.MultiIndex) else df_all.copy()
                        df.dropna(how='all', inplace=True)
                        
                        if len(df) >= 60:
                            # 特征工程
                            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
                            df['Volatility'] = df['Log_Ret'].rolling(window=20).std()
                            df.dropna(inplace=True)
                            
                            original_code, name = mapping[t]
                            data_dict[original_code] = {"data": df, "name": name}
                    except:
                        continue
        except Exception as e:
            continue
    
    return data_dict

# ==========================================
# 2. 改进的贝叶斯HMM模型
# ==========================================

def calculate_state_conditional_returns(df: pd.DataFrame, regimes: np.ndarray, 
                                        n_comps: int, window: int = 60) -> np.ndarray:
    """
    计算滚动窗口的状态条件收益率
    避免前视偏差
    """
    state_means = np.zeros((len(df), n_comps))
    
    for t in range(len(df)):
        # 确定可用的历史数据窗口
        if t < window:
            start_idx = 0
        else:
            start_idx = t - window
        
        historical_data = df.iloc[start_idx:t+1]
        historical_regimes = regimes[start_idx:t+1]
        
        for state in range(n_comps):
            state_mask = historical_regimes == state
            if np.sum(state_mask) > 5:  # 有足够的数据点
                state_returns = historical_data['Log_Ret'].values[state_mask]
                state_means[t, state] = np.mean(state_returns)
            else:
                # 数据不足时，使用全局均值
                state_means[t, state] = historical_data['Log_Ret'].mean()
    
    return state_means

def train_bhmm_improved(df: pd.DataFrame, n_comps: int, rolling_window: int = 60) -> Optional[pd.DataFrame]:
    """
    改进的贝叶斯HMM训练，避免前视偏差
    """
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        # 训练HMM模型
        model = GaussianHMM(
            n_components=n_comps, 
            covariance_type="full", 
            n_iter=1000, 
            random_state=88, 
            tol=0.01, 
            min_covar=0.001
        )
        model.fit(X)
        
        # 预测隐藏状态
        hidden_states = model.predict(X)
        
        # 状态排序（按波动率）
        state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps)]
        sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping[s] for s in hidden_states])
        
        # 获取转移矩阵
        transmat = model.transmat_
        # 重新排列转移矩阵以匹配排序后的状态
        new_transmat = np.zeros_like(transmat)
        for i in range(n_comps):
            for j in range(n_comps):
                new_transmat[mapping[i], mapping[j]] = transmat[i, j]
        
        # 获取后验概率
        posterior_probs = model.predict_proba(X)
        # 重新排列后验概率
        sorted_probs = np.zeros_like(posterior_probs)
        for old_i, new_i in mapping.items():
            sorted_probs[:, new_i] = posterior_probs[:, old_i]
        
        # 计算滚动窗口的状态条件收益率
        state_conditional_returns = calculate_state_conditional_returns(
            df, df['Regime'].values, n_comps, rolling_window
        )
        
        # 计算贝叶斯预期收益率
        bayes_expected_returns = np.zeros(len(df))
        for t in range(len(df)):
            if t == 0:
                bayes_expected_returns[t] = 0
            else:
                # 使用转移矩阵计算下一状态概率
                next_state_probs = np.dot(sorted_probs[t-1], new_transmat)
                # 计算预期收益率
                expected_return = np.dot(next_state_probs, state_conditional_returns[t-1])
                bayes_expected_returns[t] = expected_return
        
        df['Bayes_Exp_Ret'] = bayes_expected_returns
        
        # 添加置信度指标
        df['Regime_Confidence'] = np.max(sorted_probs, axis=1)
        
        return df
    except Exception as e:
        return None

# ==========================================
# 3. 回测系统
# ==========================================

def backtest_strategy(df: pd.DataFrame, cost: float = 0.001) -> Tuple[pd.DataFrame, Dict]:
    """回测策略"""
    threshold = 0.0005  # 5bps
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    df.loc[df['Bayes_Exp_Ret'] < -threshold, 'Signal'] = -1  # 允许做空
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    t_cost = df['Position'].diff().abs() * cost
    
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    # 计算性能指标
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    
    # 最大回撤
    cumulative = df['Cum_Strat']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # 夏普比率
    if df['Strategy_Ret'].std() != 0:
        sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252))
    else:
        sharpe = 0
    
    # 胜率
    winning_trades = (df['Strategy_Ret'] > 0).sum()
    total_trades = (df['Position'].diff() != 0).sum()
    win_rate = winning_trades / max(total_trades, 1)
    
    return df, {
        "Total Return": total_ret,
        "CAGR": annual_ret,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Total Trades": total_trades
    }

# ==========================================
# 4. AI 投顾
# ==========================================

def get_ai_advice(df: pd.DataFrame, metrics: Dict, n_comps: int) -> Dict:
    """获取AI投顾建议"""
    last_regime = int(df['Regime'].iloc[-1])
    last_alpha = df['Bayes_Exp_Ret'].iloc[-1]
    last_confidence = df['Regime_Confidence'].iloc[-1]
    
    advice = {
        "title": "",
        "color": "",
        "bg_color": "",
        "summary": "",
        "action": "",
        "risk_level": "",
        "position": "0%",
        "confidence": f"{last_confidence:.1%}"
    }
    
    threshold = 0.0005
    
    # 根据状态和Alpha给出建议
    if last_regime == 0:  # 低波动状态
        advice['risk_level'] = "低 (Low Risk)"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会 (Accumulation Phase)"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = f"低波动稳态，预期Alpha: {last_alpha*100:.2f}bps > 阈值5bps。置信度: {last_confidence:.1%}"
            advice['action'] = "建议：分批买入，设置止损-3%"
            advice['position'] = "60-80%"
        else:
            advice['title'] = "🟡 观望/防守 (Defensive)"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = f"低波动但预期收益不足 (Alpha: {last_alpha*100:.2f}bps)"
            advice['action'] = "建议：轻仓观察，等待信号"
            advice['position'] = "10-20%"
            
    elif last_regime == n_comps - 1:  # 高波动状态
        advice['risk_level'] = "高 (High Risk)"
        if last_alpha > threshold:
            advice['title'] = "🔵 高风险机会 (High Risk Opportunity)"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"高波动中隐含机会，Alpha: {last_alpha*100:.2f}bps"
            advice['action'] = "建议：小仓位试探，严格止损-5%"
            advice['position'] = "20-30%"
        else:
            advice['title'] = "🔴 极度风险预警 (Danger Zone)"
            advice['color'] = "#FF1744"
            advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
            advice['summary'] = "剧烈波动模式，下跌风险高"
            advice['action'] = "建议：清仓避险，等待企稳"
            advice['position'] = "0%"
    else:  # 中间状态
        advice['risk_level'] = "中 (Medium Risk)"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续 (Trend Continuation)"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"趋势运行中，Alpha: {last_alpha*100:.2f}bps"
            advice['action'] = "建议：持有为主，跟踪止盈"
            advice['position'] = "40-60%"
        else:
            advice['title'] = "🟠 减仓观望 (Reduce Exposure)"
            advice['color'] = "#FF9100"
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "上涨动能衰竭，风险上升"
            advice['action'] = "建议：逐步减仓，锁定利润"
            advice['position'] = "10-20%"
    
    return advice

# ==========================================
# 5. 高效全市场扫描系统
# ==========================================

class MarketScanner:
    def __init__(self):
        self.cache_dir = ".market_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_market_filters(self, min_market_cap: float = 100.0, 
                          min_turnover: float = 10000.0) -> pd.DataFrame:
        """获取筛选后的股票池"""
        try:
            df = ak.stock_zh_a_spot_em()
            
            # 清理数据
            df['总市值'] = pd.to_numeric(df['总市值'].str.replace('亿', '').str.replace(',', ''), errors='coerce')
            df['成交额'] = pd.to_numeric(df['成交额'].str.replace('亿', '').str.replace(',', ''), errors='coerce') * 10000
            
            # 筛选条件
            filtered = df[
                (df['总市值'] >= min_market_cap) &
                (df['成交额'] >= min_turnover)
            ].copy()
            
            filtered = filtered.sort_values('总市值', ascending=False)
            return filtered[['代码', '名称', '总市值', '成交额', '涨跌幅']]
        except Exception as e:
            st.error(f"获取市场筛选数据失败: {str(e)}")
            return pd.DataFrame()
    
    def process_single_stock_scan(self, code: str, name: str, start_date: str, 
                                 end_date: str, n_components: int = 3) -> Optional[Dict]:
        """处理单只股票的扫描分析"""
        try:
            ticker, _ = format_ticker_for_yfinance(code, name)
            df, _ = get_data(ticker, start_date, end_date, use_cache=True)
            
            if df is None or len(df) < 100:
                return None
            
            # 简化特征工程
            df_scan = df[['Close']].copy()
            df_scan['Log_Ret'] = np.log(df_scan['Close'] / df_scan['Close'].shift(1))
            df_scan['Volatility'] = df_scan['Log_Ret'].rolling(20).std()
            df_scan.dropna(inplace=True)
            
            if len(df_scan) < 60:
                return None
            
            # 训练简化版HMM（使用对角协方差矩阵加速）
            scale = 100.0
            X = df_scan[['Log_Ret', 'Volatility']].values * scale
            
            try:
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag",  # 使用对角矩阵加速
                    n_iter=200,  # 减少迭代次数
                    random_state=88
                )
                model.fit(X)
                
                hidden_states = model.predict(X)
                
                # 状态排序
                state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_components)]
                sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
                mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
                
                current_state = mapping[hidden_states[-1]]
                
                # 计算近期Alpha
                recent_alpha = df_scan['Log_Ret'].tail(5).mean()
                
                # 计算信号强度
                volatility = df_scan['Volatility'].iloc[-1]
                if volatility > 0:
                    signal_strength = recent_alpha / volatility
                else:
                    signal_strength = 0
                
                # 计算状态稳定性
                state_stability = np.sum(hidden_states[-20:] == hidden_states[-1]) / 20
                
                return {
                    'code': code,
                    'name': name,
                    'state': current_state,
                    'state_stability': state_stability,
                    'alpha': recent_alpha,
                    'volatility': volatility,
                    'signal_strength': signal_strength,
                    'close': df_scan['Close'].iloc[-1],
                    'volume': df_scan.get('Volume', pd.Series([0])).iloc[-1] if 'Volume' in df_scan.columns else 0,
                    'last_update': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            except:
                return None
                
        except Exception as e:
            return None
    
    def efficient_batch_scan(self, stock_list: pd.DataFrame, start_date: str, 
                            end_date: str, n_components: int = 3, 
                            max_workers: int = 4) -> pd.DataFrame:
        """高效批量扫描"""
        results = []
        
        # 准备股票列表
        stock_items = []
        for _, row in stock_list.iterrows():
            stock_items.append((row['代码'], row['名称']))
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_stock = {}
            for code, name in stock_items:
                future = executor.submit(
                    self.process_single_stock_scan,
                    code, name, start_date, end_date, n_components
                )
                future_to_stock[future] = (code, name)
            
            # 处理结果
            with st.spinner("正在分析..."):
                progress_bar = st.progress(0)
                completed = 0
                total = len(future_to_stock)
                
                for future in concurrent.futures.as_completed(future_to_stock):
                    completed += 1
                    progress_bar.progress(completed / total)
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except:
                        continue
                
                progress_bar.empty()
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def get_daily_scan(self, min_market_cap: float = 100.0, 
                       min_turnover: float = 10000.0,
                       sample_size: int = 200,
                       start_date: str = None,
                       end_date: str = None,
                       n_components: int = 3) -> pd.DataFrame:
        """执行每日扫描"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 检查缓存
        today = datetime.now().strftime("%Y%m%d")
        cache_file = os.path.join(self.cache_dir, f"scan_{today}_{sample_size}.pkl")
        
        if os.path.exists(cache_file):
            try:
                return pd.read_pickle(cache_file)
            except:
                pass
        
        # 获取筛选后的股票池
        filtered_stocks = self.get_market_filters(min_market_cap, min_turnover)
        
        if filtered_stocks.empty:
            return pd.DataFrame()
        
        # 抽样
        sample_size = min(sample_size, len(filtered_stocks))
        sampled_stocks = filtered_stocks.head(sample_size)
        
        # 执行扫描
        results = self.efficient_batch_scan(
            sampled_stocks, start_date, end_date, n_components
        )
        
        # 缓存结果
        if not results.empty:
            try:
                results.to_pickle(cache_file)
            except:
                pass
        
        return results

# ==========================================
# 6. 主程序逻辑
# ==========================================

def main():
    # 初始化扫描器
    scanner = MarketScanner()
    
    # 侧边栏通用配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share Pro Plus")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 单标的分析", "📡 板块扫描", "🌐 全市场扫描", "📊 回测优化"], 
            index=0
        )
        st.divider()
        
        # 通用参数
        n_components = st.slider("隐藏状态数", 2, 4, 3)
        lookback_years = st.slider("回看年限", 1, 5, 3)
        trans_cost_bps = st.number_input("交易成本(bps)", value=10, min_value=0, max_value=50)
        transaction_cost = trans_cost_bps / 10000
        
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        st.divider()
        
        # 模式特定配置
        if app_mode == "🔎 单标的分析":
            st.caption("单标的设置")
            with st.spinner("连接市场数据..."):
                stock_list_df, is_online = get_all_a_share_list()
            
            target_ticker, target_name = None, None
            if is_online and not stock_list_df.empty:
                selected = st.selectbox("代码/名称搜索", options=stock_list_df['Display'])
                if selected:
                    parts = selected.split(" | ")
                    if len(parts) >= 2:
                        c = parts[0]
                        n = parts[1]
                        target_ticker, target_name = format_ticker_for_yfinance(c, n)
            else:
                mc = st.text_input("股票代码", value="000858")
                if mc:
                    target_ticker, target_name = format_ticker_for_yfinance(mc, mc)
            
            run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
            
        elif app_mode == "📡 板块扫描":
            st.caption("板块扫描设置")
            SECTORS = ["半导体", "白酒", "证券", "中药", "光伏设备", 
                      "消费电子", "游戏", "电池", "电网设备", "汽车整车"]
            target_sector = st.selectbox("选择板块", SECTORS)
            sector_top_n = st.slider("成分股数量", 10, 50, 20)
            
            scan_btn = st.button("📡 开始扫描", type="primary", use_container_width=True)
            
        elif app_mode == "🌐 全市场扫描":
            st.caption("全市场扫描设置")
            scan_type = st.radio("扫描类型", ["快速扫描", "深度扫描"], index=0)
            
            min_market_cap = st.number_input("最小市值(亿)", value=100.0, min_value=10.0)
            min_turnover = st.number_input("最小成交额(万)", value=10000.0, min_value=1000.0)
            
            if scan_type == "快速扫描":
                sample_size = st.slider("样本数量", 100, 500, 200, 50)
                max_workers = 4
            else:
                sample_size = st.slider("样本数量", 200, 1000, 500, 100)
                max_workers = 6
            
            market_scan_btn = st.button("🌐 开始市场扫描", type="primary", use_container_width=True)
            
        elif app_mode == "📊 回测优化":
            st.caption("回测优化设置")
            opt_method = st.selectbox("优化方法", ["网格搜索", "随机搜索", "贝叶斯优化"])
            param_grid = {
                "threshold": st.slider("信号阈值(bps)", 1, 20, 5),
                "lookback": st.slider("观察窗口(日)", 10, 100, 60),
                "stop_loss": st.slider("止损比例(%)", 1, 10, 3) / 100
            }
            optimize_btn = st.button("🔧 开始优化", type="primary", use_container_width=True)
    
    # ========== 模式A: 单标的分析 ==========
    if app_mode == "🔎 单标的分析":
        st.title("🔎 A-Share 单标的深度分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在分析 {target_name}..."):
                # 获取数据
                df, final_ticker = get_data(target_ticker, start_date, end_date)
                
                if df is None:
                    st.error("无法获取股票数据，请检查代码是否正确")
                    st.stop()
                
                # 训练改进的BHMM模型
                df = train_bhmm_improved(df, n_components)
                
                if df is None:
                    st.error("模型训练失败")
                    st.stop()
                
                # 回测
                df, metrics = backtest_strategy(df, transaction_cost)
                
                # 获取AI建议
                ai_advice = get_ai_advice(df, metrics, n_components)
                
                # 显示性能指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("累计收益", f"{metrics['Total Return']*100:.1f}%")
                with col2:
                    st.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
                with col3:
                    st.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                with col4:
                    st.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("胜率", f"{metrics['Win Rate']*100:.1f}%")
                with col6:
                    st.metric("交易次数", f"{metrics['Total Trades']}")
                
                # 显示AI建议
                st.markdown(f"""
                <div style="background:{ai_advice['bg_color']}; padding:20px; border-radius:10px; 
                          border-left:5px solid {ai_advice['color']}; margin:20px 0;">
                    <h3 style="color:{ai_advice['color']}; margin:0;">{ai_advice['title']}</h3>
                    <p style="color:#ccc; margin-top:10px;">{ai_advice['summary']}</p>
                    <div style="display:flex; justify-content:space-between; margin-top:15px; font-weight:bold;">
                        <span style="color:#fff;">操作建议: {ai_advice['action']}</span>
                        <span style="color:{ai_advice['color']};">推荐仓位: {ai_advice['position']}</span>
                    </div>
                    <div style="margin-top:10px; color:#888; font-size:0.9em;">
                        风险等级: {ai_advice['risk_level']} | 模型置信度: {ai_advice['confidence']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 图表展示
                tab1, tab2, tab3 = st.tabs(["📈 价格与状态", "📊 策略收益", "📋 详细数据"])
                
                with tab1:
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3]
                    )
                    
                    # 价格与状态
                    colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
                    for i in range(n_components):
                        mask = df['Regime'] == i
                        if mask.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index[mask], 
                                    y=df['Close'][mask], 
                                    mode='markers',
                                    marker=dict(size=5, color=colors[i % 4]),
                                    name=f"状态 {i}",
                                    legendgroup=f"state_{i}"
                                ),
                                row=1, col=1
                            )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, 
                            y=df['Close'], 
                            line=dict(color='rgba(255,255,255,0.3)', width=1),
                            name="收盘价",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # 成交量
                    if 'Volume' in df.columns:
                        fig.add_trace(
                            go.Bar(
                                x=df.index, 
                                y=df['Volume'],
                                marker_color='rgba(100, 149, 237, 0.5)',
                                name="成交量",
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified"
                    )
                    
                    fig.update_yaxes(title_text="价格", row=1, col=1)
                    fig.update_yaxes(title_text="成交量", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig_eq = go.Figure()
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Cum_Bench'],
                        name="基准",
                        line=dict(color='gray', dash='dot', width=1)
                    ))
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df.index, 
                        y=df['Cum_Strat'],
                        name="BHMM策略",
                        line=dict(color='#FF5252', width=2)
                    ))
                    
                    # 添加最大回撤区域
                    cumulative = df['Cum_Strat']
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df.index,
                        y=running_max,
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(255, 82, 82, 0.2)', width=0),
                        showlegend=False
                    ))
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df.index,
                        y=cumulative,
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(255, 82, 82, 0.1)', width=0),
                        fillcolor='rgba(255, 82, 82, 0.1)',
                        name='回撤区域',
                        showlegend=True
                    ))
                    
                    fig_eq.update_layout(
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title="策略收益曲线",
                        yaxis_title="累计收益"
                    )
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
                
                with tab3:
                    # 显示详细数据
                    display_cols = ['Close', 'Log_Ret', 'Volatility', 'Regime', 
                                  'Regime_Confidence', 'Bayes_Exp_Ret', 'Signal', 'Position']
                    
                    available_cols = [col for col in display_cols if col in df.columns]
                    display_df = df[available_cols].copy()
                    
                    # 格式化显示
                    if 'Bayes_Exp_Ret' in display_df.columns:
                        display_df['Bayes_Exp_Ret'] = display_df['Bayes_Exp_Ret'] * 10000  # 转换为bps
                    
                    if 'Regime_Confidence' in display_df.columns:
                        display_df['Regime_Confidence'] = display_df['Regime_Confidence'].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(
                        display_df.tail(100).style.format({
                            'Close': '{:.2f}',
                            'Log_Ret': '{:.4f}',
                            'Volatility': '{:.4f}',
                            'Bayes_Exp_Ret': '{:.2f}bps'
                        }),
                        use_container_width=True
                    )
                    
                    # 下载数据
                    csv = display_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 下载详细数据",
                        data=csv,
                        file_name=f"{target_ticker.split('.')[0]}_analysis.csv",
                        mime="text/csv"
                    )
        
        elif run_btn:
            st.warning("请选择有效的股票代码")
        else:
            st.info("👈 请在侧边栏选择股票代码并开始分析")
    
    # ========== 模式B: 板块扫描 ==========
    elif app_mode == "📡 板块扫描":
        st.title(f"📡 板块扫描: {target_sector}")
        
        if scan_btn:
            with st.spinner(f"正在获取 {target_sector} 成分股..."):
                stock_list = get_sector_components(target_sector, sector_top_n)
                
                if not stock_list:
                    st.error("无法获取板块成分股，请稍后再试")
                    st.stop()
                
                st.success(f"获取到 {len(stock_list)} 只成分股")
            
            with st.spinner("正在批量分析..."):
                # 批量下载数据
                data_dict = batch_download_data(stock_list, start_date, end_date)
                
                if not data_dict:
                    st.error("数据下载失败")
                    st.stop()
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, (code, item) in enumerate(data_dict.items()):
                    df_scan = item['data'].copy()
                    name_scan = item['name']
                    
                    # 简化分析
                    if len(df_scan) > 100:
                        try:
                            # 计算基本指标
                            df_scan['Log_Ret'] = np.log(df_scan['Close'] / df_scan['Close'].shift(1))
                            df_scan['Volatility'] = df_scan['Log_Ret'].rolling(20).std()
                            df_scan.dropna(inplace=True)
                            
                            if len(df_scan) > 60:
                                # 简单波动率分类
                                current_vol = df_scan['Volatility'].iloc[-1]
                                vol_percentile = (df_scan['Volatility'] < current_vol).mean()
                                
                                # 状态分类（简化）
                                if vol_percentile < 0.3:
                                    regime = 0  # 低波动
                                elif vol_percentile > 0.7:
                                    regime = 2  # 高波动
                                else:
                                    regime = 1  # 中波动
                                
                                recent_alpha = df_scan['Log_Ret'].tail(5).mean()
                                signal_score = recent_alpha / (current_vol + 1e-6)
                                
                                results.append({
                                    "代码": code,
                                    "名称": name_scan,
                                    "状态": regime,
                                    "Alpha": recent_alpha,
                                    "波动率": current_vol,
                                    "信号强度": signal_score,
                                    "最新价": df_scan['Close'].iloc[-1]
                                })
                        except:
                            continue
                    
                    progress_bar.progress((idx + 1) / len(data_dict))
                
                progress_bar.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # 筛选推荐标的
                    recommendation_df = results_df[
                        (results_df['状态'] == 0) & 
                        (results_df['Alpha'] > 0.0005)
                    ].sort_values('信号强度', ascending=False)
                    
                    if not recommendation_df.empty:
                        st.success(f"🎯 发现 {len(recommendation_df)} 只潜在建仓标的")
                        
                        # 显示推荐标的
                        cols = st.columns(3)
                        for idx, row in recommendation_df.iterrows():
                            with cols[idx % 3]:
                                state_color = ['#00E676', '#FFD600', '#FF1744'][int(row['状态'])]
                                st.markdown(f"""
                                <div class="scanner-card state-{int(row['状态'])}">
                                    <h4 style="margin:0;">{row['名称']}</h4>
                                    <div style="color:#aaa; font-size:0.9em;">{row['代码']}</div>
                                    <div style="margin-top:10px; display:flex; justify-content:space-between;">
                                        <span style="color:{state_color}; font-weight:bold;">
                                            Alpha: {row['Alpha']*10000:.1f}bps
                                        </span>
                                        <span style="color:#ccc;">¥{row['最新价']:.2f}</span>
                                    </div>
                                    <div style="font-size:0.8em; color:#888; margin-top:5px;">
                                        信号强度: {row['信号强度']:.2f} | 波动率: {row['波动率']:.3f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("当前板块未发现符合条件的标的")
                    
                    # 显示完整结果
                    with st.expander("📋 查看完整分析结果"):
                        styled_df = results_df.style.format({
                            'Alpha': '{:.4%}',
                            '波动率': '{:.4f}',
                            '信号强度': '{:.2f}',
                            '最新价': '{:.2f}'
                        }).background_gradient(
                            subset=['Alpha', '信号强度'], 
                            cmap='RdYlGn'
                        )
                        
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.error("分析失败，请重试")
        else:
            st.info("👈 请在侧边栏选择板块并开始扫描")
    
    # ========== 模式C: 全市场扫描 ==========
    elif app_mode == "🌐 全市场扫描":
        st.title("🌐 全市场智能扫描")
        st.info("💡 扫描逻辑：市值筛选 → 流动性过滤 → 批量HMM分析 → 智能排序")
        
        if market_scan_btn:
            with st.spinner("正在筛选股票池..."):
                filtered_stocks = scanner.get_market_filters(min_market_cap, min_turnover)
                
                if filtered_stocks.empty:
                    st.error("筛选失败，请检查网络连接")
                    st.stop()
                
                st.success(f"筛选出 {len(filtered_stocks)} 只符合条件的股票")
                
                # 显示筛选结果摘要
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均市值", f"{filtered_stocks['总市值'].mean():.1f}亿")
                with col2:
                    st.metric("平均成交额", f"{filtered_stocks['成交额'].mean()/10000:.1f}亿")
                with col3:
                    st.metric("涨跌比", 
                            f"{(filtered_stocks['涨跌幅'] > 0).sum()}/{(filtered_stocks['涨跌幅'] < 0).sum()}")
            
            with st.spinner("正在执行全市场HMM扫描..."):
                # 执行扫描
                results = scanner.get_daily_scan(
                    min_market_cap=min_market_cap,
                    min_turnover=min_turnover,
                    sample_size=sample_size,
                    start_date=start_date,
                    end_date=end_date,
                    n_components=n_components
                )
                
                if results.empty:
                    st.error("扫描失败，请重试")
                    st.stop()
                
                st.success(f"扫描完成！共分析 {len(results)} 只股票")
                
                # 按状态分组展示
                for state in range(n_components):
                    state_stocks = results[results['state'] == state].copy()
                    
                    if len(state_stocks) > 0:
                        # 排序
                        if state == 0:  # 低波动状态
                            state_stocks = state_stocks.sort_values('alpha', ascending=False)
                            title = f"📈 状态{state}: 低波动机会 (共{len(state_stocks)}只)"
                        elif state == n_components - 1:  # 高波动状态
                            state_stocks = state_stocks.sort_values('signal_strength', ascending=False)
                            title = f"⚡ 状态{state}: 高波动机会 (共{len(state_stocks)}只)"
                        else:  # 中间状态
                            state_stocks = state_stocks.sort_values('signal_strength', ascending=False)
                            title = f"📊 状态{state}: 趋势运行 (共{len(state_stocks)}只)"
                        
                        with st.expander(title):
                            # 显示前10只
                            for _, row in state_stocks.head(10).iterrows():
                                alpha_color = "#00E676" if row['alpha'] > 0.0005 else "#FF1744"
                                alpha_class = "positive-alpha" if row['alpha'] > 0.0005 else "negative-alpha"
                                
                                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                                
                                with col1:
                                    st.markdown(f"**{row['name']}** ({row['code']})")
                                
                                with col2:
                                    st.metric("Alpha", f"{row['alpha']*10000:.1f}bps", 
                                            delta_color="normal" if row['alpha'] > 0 else "inverse")
                                
                                with col3:
                                    st.metric("信号强度", f"{row['signal_strength']:.2f}")
                                
                                with col4:
                                    st.metric("价格", f"¥{row['close']:.2f}")
                
                # 显示综合排名
                st.subheader("🏆 综合排名 Top 20")
                
                # 计算综合得分
                results['综合得分'] = (
                    results['alpha'] * 10000 * 0.4 +  # Alpha权重40%
                    results['signal_strength'] * 0.3 +  # 信号强度权重30%
                    (1 - results['state'] / (n_components - 1)) * 0.3  # 状态权重30%（低状态更好）
                )
                
                top_20 = results.sort_values('综合得分', ascending=False).head(20)
                
                for idx, (_, row) in enumerate(top_20.iterrows(), 1):
                    with st.container():
                        state_color = ['#00E676', '#FFD600', '#FF1744', '#AA00FF'][int(row['state'])]
                        
                        st.markdown(f"""
                        <div class="scanner-card" style="border-left: 4px solid {state_color};">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:1.2em; font-weight:bold;">#{idx}</span>
                                    <span style="margin-left:10px; font-weight:bold;">{row['name']}</span>
                                    <span style="color:#aaa; margin-left:5px;">({row['code']})</span>
                                </div>
                                <div style="text-align:right;">
                                    <div style="color:{state_color}; font-weight:bold;">状态 {int(row['state'])}</div>
                                    <div style="color:#ccc; font-size:0.9em;">¥{row['close']:.2f}</div>
                                </div>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                                <div>
                                    <span style="color:#00E676; margin-right:15px;">
                                        Alpha: {row['alpha']*10000:.1f}bps
                                    </span>
                                    <span style="color:#FFD600;">
                                        强度: {row['signal_strength']:.2f}
                                    </span>
                                </div>
                                <div style="color:#888;">
                                    综合得分: {row['综合得分']:.2f}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 下载结果
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载完整扫描结果",
                    data=csv,
                    file_name=f"market_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("👈 请在侧边栏配置参数并开始扫描")
    
    # ========== 模式D: 回测优化 ==========
    elif app_mode == "📊 回测优化":
        st.title("📊 回测参数优化")
        
        if optimize_btn:
            st.warning("回测优化功能正在开发中...")
            st.info("""
            计划功能：
            1. 多参数网格搜索
            2. 夏普比率最大化
            3. 最大回撤最小化
            4. 过拟合检测
            5. 参数稳定性测试
            """)
        else:
            st.info("👈 配置优化参数并开始优化")

if __name__ == "__main__":
    main()
[file content end]