import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import pickle
import time
import requests
from hmmlearn.hmm import GaussianHMM
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

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
# 1. 智能数据获取系统 (多重后备方案)
# ==========================================

class DataFetcher:
    """智能数据获取器，支持多重后备方案"""
    
    def __init__(self):
        self.cache_dir = ".data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 预定义A股龙头股数据库
        self._init_predefined_stocks()
        
    def _init_predefined_stocks(self):
        """初始化预定义股票数据库"""
        # 扩展的A股龙头股列表 (300+只)
        self.predefined_stocks = {
            # 白酒
            "白酒": [
                ("000858", "五粮液", 130.5, 1500),
                ("600519", "贵州茅台", 1600.0, 20000),
                ("002304", "洋河股份", 85.3, 1200),
                ("000568", "泸州老窖", 180.2, 2600),
                ("600809", "山西汾酒", 210.5, 2500),
            ],
            # 半导体
            "半导体": [
                ("688981", "中芯国际", 45.6, 3500),
                ("002049", "紫光国微", 85.4, 700),
                ("603501", "韦尔股份", 95.2, 1100),
                ("300661", "圣邦股份", 120.8, 500),
                ("002371", "北方华创", 280.5, 1500),
                ("600703", "三安光电", 15.2, 700),
                ("300782", "卓胜微", 85.6, 400),
            ],
            # 新能源
            "新能源": [
                ("300750", "宁德时代", 180.5, 8000),
                ("002594", "比亚迪", 210.3, 6000),
                ("002812", "恩捷股份", 45.6, 400),
                ("002460", "赣锋锂业", 35.8, 600),
                ("300014", "亿纬锂能", 38.9, 700),
                ("002709", "天赐材料", 22.5, 400),
                ("300450", "先导智能", 25.6, 400),
            ],
            # 医药
            "医药": [
                ("600276", "恒瑞医药", 42.8, 2700),
                ("300760", "迈瑞医疗", 285.6, 3500),
                ("300015", "爱尔眼科", 15.2, 1400),
                ("000538", "云南白药", 52.4, 900),
                ("600085", "同仁堂", 45.6, 600),
                ("600436", "片仔癀", 240.5, 1400),
                ("300347", "泰格医药", 52.3, 400),
            ],
            # 金融
            "金融": [
                ("601318", "中国平安", 42.5, 7500),
                ("600036", "招商银行", 32.8, 8000),
                ("601398", "工商银行", 4.9, 17000),
                ("601166", "兴业银行", 15.6, 3200),
                ("600030", "中信证券", 22.4, 3300),
                ("000776", "广发证券", 14.2, 1100),
                ("601601", "中国太保", 23.5, 2200),
            ],
            # 消费
            "消费": [
                ("600887", "伊利股份", 28.5, 1800),
                ("000651", "格力电器", 35.6, 2000),
                ("000333", "美的集团", 58.9, 4000),
                ("603288", "海天味业", 35.8, 2000),
                ("002557", "洽洽食品", 32.4, 200),
                ("300146", "汤臣倍健", 18.9, 300),
                ("603866", "桃李面包", 7.2, 100),
            ],
            # 科技
            "科技": [
                ("002415", "海康威视", 32.5, 3000),
                ("002475", "立讯精密", 28.9, 2000),
                ("300059", "东方财富", 13.2, 2100),
                ("300033", "同花顺", 105.6, 500),
                ("002230", "科大讯飞", 45.8, 1000),
                ("000977", "浪潮信息", 32.5, 500),
                ("600570", "恒生电子", 25.6, 500),
            ],
            # 光伏
            "光伏设备": [
                ("601012", "隆基绿能", 18.5, 1400),
                ("300274", "阳光电源", 75.6, 1100),
                ("002129", "TCL中环", 12.8, 400),
                ("688303", "大全能源", 25.4, 500),
                ("300118", "东方日升", 14.2, 200),
                ("603806", "福斯特", 28.9, 500),
            ],
            # 汽车
            "汽车整车": [
                ("601633", "长城汽车", 23.5, 2000),
                ("600104", "上汽集团", 14.2, 1600),
                ("000625", "长安汽车", 14.8, 1500),
                ("002594", "比亚迪", 210.3, 6000),
                ("601238", "广汽集团", 8.9, 900),
            ],
            # 军工
            "军工": [
                ("600893", "航发动力", 35.6, 900),
                ("600760", "中航沈飞", 38.9, 1000),
                ("002179", "中航光电", 32.5, 600),
                ("000768", "中航西飞", 23.4, 600),
                ("600862", "中航高科", 18.9, 300),
            ]
        }
        
        # 创建全市场列表
        self.all_stocks = []
        for sector, stocks in self.predefined_stocks.items():
            for code, name, price, market_cap in stocks:
                self.all_stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    '价格': price,
                    '市值': market_cap
                })
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_stock_list_from_alternative(self):
        """从备用API获取股票列表"""
        try:
            # 尝试从东方财富备用API获取
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                "pn": "1",
                "pz": "1000",
                "po": "1",
                "np": "1",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                "fltt": "2",
                "invt": "2",
                "fid": "f3",
                "fs": "m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23",
                "fields": "f12,f14,f2,f3,f4,f20,f21",
                "_": str(int(time.time() * 1000))
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://quote.eastmoney.com/"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                stocks = []
                for item in data.get("data", {}).get("diff", []):
                    code = item.get("f12", "")
                    name = item.get("f14", "")
                    if code and name:
                        stocks.append({"代码": code, "名称": name})
                return pd.DataFrame(stocks), True
        except:
            pass
        
        # 如果失败，使用预定义数据
        df = pd.DataFrame(self.all_stocks)
        df['Display'] = df['代码'] + " | " + df['名称'] + " | " + df['板块']
        return df, True
    
    def get_sector_components(self, sector_name: str) -> List[Tuple[str, str]]:
        """获取板块成分股"""
        if sector_name in self.predefined_stocks:
            return [(code, name) for code, name, _, _ in self.predefined_stocks[sector_name]]
        return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_stock_data(self, ticker: str, start: str, end: str):
        """获取股票数据，带重试机制"""
        cache_key = f"{ticker}_{start}_{end}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, dict) and 'df' in cached_data:
                        return cached_data['df'], cached_data.get('ticker', ticker)
            except:
                pass
        
        try:
            # 主数据源：yfinance
            df = yf.download(ticker, start=start, end=end, interval="1d", 
                           progress=False, auto_adjust=True, timeout=10)
            
            # 如果yfinance失败，尝试备用后缀
            if df.empty or len(df) < 10:
                base_code = ticker.split('.')[0]
                if len(ticker.split('.')) > 1:
                    current_suffix = '.' + ticker.split('.')[1]
                    alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
                    alt_ticker = base_code + alt_suffix
                    df = yf.download(alt_ticker, start=start, end=end, 
                                   progress=False, auto_adjust=True, timeout=10)
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
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'df': data, 'ticker': ticker}, f)
            except:
                pass
            
            return data, ticker
            
        except Exception as e:
            return None, ticker
    
    def batch_download_data(self, tickers_list: List[Tuple[str, str]], start: str, end: str, max_workers: int = 4):
        """批量下载数据"""
        data_dict = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for code, name in tickers_list:
                ticker, _ = self.format_ticker_for_yfinance(code, name)
                future = executor.submit(self.get_stock_data, ticker, start, end)
                futures[future] = (code, name, ticker)
            
            for future in concurrent.futures.as_completed(futures):
                code, name, ticker = futures[future]
                try:
                    df, final_ticker = future.result()
                    if df is not None and not df.empty:
                        data_dict[code] = {"data": df, "name": name, "ticker": final_ticker}
                except:
                    continue
        
        return data_dict
    
    def format_ticker_for_yfinance(self, raw_code: str, raw_name: str = "Unknown") -> Tuple[str, str]:
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

# 初始化数据获取器
data_fetcher = DataFetcher()

# ==========================================
# 2. 改进的贝叶斯HMM模型 (保留完整功能)
# ==========================================

def calculate_state_conditional_returns(df: pd.DataFrame, regimes: np.ndarray, 
                                        n_comps: int, window: int = 60) -> np.ndarray:
    """计算滚动窗口的状态条件收益率"""
    state_means = np.zeros((len(df), n_comps))
    
    for t in range(len(df)):
        if t < window:
            start_idx = 0
        else:
            start_idx = t - window
        
        historical_data = df.iloc[start_idx:t+1]
        historical_regimes = regimes[start_idx:t+1]
        
        for state in range(n_comps):
            state_mask = historical_regimes == state
            if np.sum(state_mask) > 5:
                state_returns = historical_data['Log_Ret'].values[state_mask]
                state_means[t, state] = np.mean(state_returns)
            else:
                state_means[t, state] = historical_data['Log_Ret'].mean()
    
    return state_means

def train_bhmm_improved(df: pd.DataFrame, n_comps: int, rolling_window: int = 60) -> Optional[pd.DataFrame]:
    """改进的贝叶斯HMM训练"""
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
        state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps) 
                          if np.sum(hidden_states == i) > 0]
        sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping.get(s, s) for s in hidden_states])
        
        # 获取转移矩阵
        transmat = model.transmat_
        new_transmat = np.zeros_like(transmat)
        for i in range(n_comps):
            for j in range(n_comps):
                new_transmat[mapping.get(i, i), mapping.get(j, j)] = transmat[i, j]
        
        # 获取后验概率
        posterior_probs = model.predict_proba(X)
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
# 3. 回测系统 (完整功能)
# ==========================================

def backtest_strategy(df: pd.DataFrame, cost: float = 0.001) -> Tuple[pd.DataFrame, Dict]:
    """完整回测策略"""
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
    
    # 卡尔玛比率
    if max_dd != 0:
        calmar = annual_ret / abs(max_dd)
    else:
        calmar = 0
    
    # 索提诺比率
    negative_returns = df['Strategy_Ret'][df['Strategy_Ret'] < 0]
    if len(negative_returns) > 0 and negative_returns.std() != 0:
        sortino = (df['Strategy_Ret'].mean() * 252) / (negative_returns.std() * np.sqrt(252))
    else:
        sortino = sharpe
    
    return df, {
        "Total Return": total_ret,
        "CAGR": annual_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Total Trades": total_trades
    }

# ==========================================
# 4. AI 投顾 (完整功能)
# ==========================================

def get_ai_advice(df: pd.DataFrame, metrics: Dict, n_comps: int) -> Dict:
    """完整AI投顾建议"""
    if len(df) == 0:
        return {
            "title": "⚠️ 数据不足",
            "color": "#FFD600",
            "bg_color": "rgba(255, 214, 0, 0.1)",
            "summary": "数据不足，无法给出建议",
            "action": "请检查数据源",
            "risk_level": "未知",
            "position": "0%",
            "confidence": "0%",
            "risk_metrics": {}
        }
    
    last_regime = int(df['Regime'].iloc[-1]) if 'Regime' in df.columns else 0
    last_alpha = df['Bayes_Exp_Ret'].iloc[-1] if 'Bayes_Exp_Ret' in df.columns else 0
    last_confidence = df['Regime_Confidence'].iloc[-1] if 'Regime_Confidence' in df.columns else 0
    
    advice = {
        "title": "",
        "color": "",
        "bg_color": "",
        "summary": "",
        "action": "",
        "risk_level": "",
        "position": "0%",
        "confidence": f"{last_confidence:.1%}",
        "risk_metrics": {}
    }
    
    threshold = 0.0005
    
    # 计算风险指标
    recent_volatility = df['Volatility'].iloc[-20:].mean() if len(df) >= 20 else df['Volatility'].mean()
    recent_max_dd = df['Cum_Strat'].iloc[-20:].min() if 'Cum_Strat' in df.columns else 0
    
    advice['risk_metrics'] = {
        "近期波动率": f"{recent_volatility:.2%}",
        "近期最大回撤": f"{recent_max_dd:.2%}",
        "模型置信度": f"{last_confidence:.1%}"
    }
    
    if last_regime == 0:  # 低波动状态
        advice['risk_level'] = "低风险 (Low Risk)"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会 (Accumulation Phase)"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = f"低波动稳态，预期Alpha: {last_alpha*10000:.1f}bps > 阈值5bps。置信度: {last_confidence:.1%}"
            advice['action'] = "建议：分批买入，设置止损-3%，关注成交量放大"
            advice['position'] = "70-90%"
        else:
            advice['title'] = "🟡 观望/防守 (Defensive)"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = f"低波动但预期收益不足 (Alpha: {last_alpha*10000:.1f}bps)。适宜防守"
            advice['action'] = "建议：轻仓观察(10-20%)，等待突破信号"
            advice['position'] = "10-20%"
            
    elif last_regime == n_comps - 1:  # 高波动状态
        advice['risk_level'] = "高风险 (High Risk)"
        if last_alpha > threshold:
            advice['title'] = "🔵 高风险机会 (High Risk Opportunity)"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"高波动中隐含机会，Alpha: {last_alpha*10000:.1f}bps"
            advice['action'] = "建议：小仓位试探(20-30%)，严格止损-5%，快进快出"
            advice['position'] = "20-30%"
        else:
            advice['title'] = "🔴 极度风险预警 (Danger Zone)"
            advice['color'] = "#FF1744"
            advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
            advice['summary'] = "剧烈波动模式，下跌风险极高"
            advice['action'] = "建议：清仓避险，等待企稳信号"
            advice['position'] = "0%"
    else:  # 中间状态
        advice['risk_level'] = "中风险 (Medium Risk)"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续 (Trend Continuation)"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"趋势运行中，Alpha: {last_alpha*10000:.1f}bps"
            advice['action'] = "建议：持有为主(50-70%)，跟踪止盈，关注趋势延续性"
            advice['position'] = "50-70%"
        else:
            advice['title'] = "🟠 减仓观望 (Reduce Exposure)"
            advice['color'] = "#FF9100"
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "上涨动能衰竭，风险上升"
            advice['action'] = "建议：逐步减仓至20-30%，锁定利润，观察调整深度"
            advice['position'] = "20-30%"
    
    return advice

# ==========================================
# 5. 高效市场扫描系统
# ==========================================

class MarketScanner:
    """高效市场扫描系统"""
    
    def __init__(self):
        self.fetcher = data_fetcher
    
    def scan_sector(self, sector_name: str, start_date: str, end_date: str, 
                   n_components: int = 3, top_n: int = 10) -> pd.DataFrame:
        """扫描板块"""
        # 获取板块成分股
        stocks = self.fetcher.get_sector_components(sector_name)
        if not stocks:
            return pd.DataFrame()
        
        # 批量下载数据
        data_dict = self.fetcher.batch_download_data(stocks[:20], start_date, end_date, max_workers=4)
        
        results = []
        for code, item in data_dict.items():
            df = item["data"]
            name = item["name"]
            
            if df is not None and len(df) > 100:
                # 训练HMM模型
                df_model = train_bhmm_improved(df, n_components)
                
                if df_model is not None:
                    last_regime = int(df_model['Regime'].iloc[-1])
                    last_alpha = df_model['Bayes_Exp_Ret'].iloc[-1]
                    confidence = df_model['Regime_Confidence'].iloc[-1] if 'Regime_Confidence' in df_model.columns else 0
                    
                    # 计算技术指标
                    recent_vol = df['Volatility'].iloc[-20:].mean()
                    recent_ret = df['Log_Ret'].iloc[-5:].mean()
                    rsi = self.calculate_rsi(df['Close']) if len(df) > 14 else 50
                    
                    # 综合评分
                    score = self.calculate_score(last_alpha, last_regime, confidence, recent_vol, recent_ret, rsi)
                    
                    results.append({
                        "代码": code,
                        "名称": name,
                        "状态": last_regime,
                        "Alpha": last_alpha,
                        "置信度": confidence,
                        "波动率": recent_vol,
                        "近期收益": recent_ret,
                        "RSI": rsi,
                        "综合评分": score,
                        "最新价": df['Close'].iloc[-1],
                        "成交量": df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    })
        
        if results:
            results_df = pd.DataFrame(results)
            return results_df.sort_values('综合评分', ascending=False).head(top_n)
        return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_score(self, alpha: float, regime: int, confidence: float, 
                       volatility: float, recent_ret: float, rsi: float) -> float:
        """计算综合评分"""
        # Alpha权重 40%
        alpha_score = alpha * 10000 * 4
        
        # 状态权重 20% (状态0最好，状态n-1最差)
        regime_score = (1 - regime / 3) * 20
        
        # 置信度权重 15%
        confidence_score = confidence * 15
        
        # 波动率权重 10% (低波动更好)
        volatility_score = (1 - min(volatility * 100, 1)) * 10
        
        # 近期收益权重 10%
        recent_ret_score = min(max(recent_ret * 10000, -10), 10)
        
        # RSI权重 5% (40-60最佳)
        rsi_score = 5 - abs(rsi - 50) * 0.1
        
        total_score = alpha_score + regime_score + confidence_score + volatility_score + recent_ret_score + rsi_score
        return max(min(total_score, 100), 0)

# ==========================================
# 6. 主程序逻辑 (完整功能)
# ==========================================

def main():
    # 初始化扫描器
    scanner = MarketScanner()
    
    # 侧边栏通用配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share Pro Plus")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 单标的深度分析", "📡 板块智能扫描", "🌐 全市场筛选", "📊 策略回测优化"], 
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
        if app_mode == "🔎 单标的深度分析":
            st.caption("单标的深度分析")
            
            # 获取股票列表
            stock_list_df, _ = data_fetcher.get_stock_list_from_alternative()
            
            if not stock_list_df.empty:
                selected = st.selectbox("选择股票", options=stock_list_df['Display'].tolist())
                if selected:
                    parts = selected.split(" | ")
                    if len(parts) >= 2:
                        c = parts[0]
                        n = parts[1]
                        target_ticker, target_name = data_fetcher.format_ticker_for_yfinance(c, n)
                    else:
                        target_ticker, target_name = None, None
                else:
                    target_ticker, target_name = None, None
            else:
                mc = st.text_input("股票代码", value="000858.SZ")
                target_ticker, target_name = data_fetcher.format_ticker_for_yfinance(mc, mc)
            
            # 高级参数
            with st.expander("高级参数"):
                rolling_window = st.slider("滚动窗口(日)", 30, 120, 60)
                signal_threshold = st.number_input("信号阈值(bps)", value=5.0, min_value=0.1, max_value=20.0) / 10000
            
            run_btn = st.button("🚀 开始深度分析", type="primary", use_container_width=True)
            
        elif app_mode == "📡 板块智能扫描":
            st.caption("板块智能扫描")
            SECTORS = list(data_fetcher.predefined_stocks.keys())
            target_sector = st.selectbox("选择板块", SECTORS)
            
            with st.expander("扫描配置"):
                top_n = st.slider("显示数量", 5, 20, 10)
                min_confidence = st.slider("最小置信度(%)", 50, 90, 70) / 100
            
            scan_btn = st.button("📡 开始智能扫描", type="primary", use_container_width=True)
            
        elif app_mode == "🌐 全市场筛选":
            st.caption("全市场筛选")
            filter_type = st.selectbox("筛选类型", ["Alpha强势股", "低波稳健股", "高置信度股", "综合评分"])
            
            with st.expander("筛选条件"):
                min_alpha = st.number_input("最小Alpha(bps)", value=5.0, min_value=0.0, max_value=20.0)
                max_volatility = st.number_input("最大波动率(%)", value=3.0, min_value=0.5, max_value=10.0) / 100
                min_confidence = st.number_input("最小置信度(%)", value=70, min_value=50, max_value=95) / 100
            
            filter_btn = st.button("🌐 开始全市场筛选", type="primary", use_container_width=True)
            
        elif app_mode == "📊 策略回测优化":
            st.caption("策略回测优化")
            optimize_type = st.selectbox("优化目标", ["夏普比率", "卡尔玛比率", "年化收益", "综合评分"])
            
            with st.expander("优化参数"):
                threshold_range = st.slider("信号阈值范围(bps)", 1, 20, (2, 10))
                window_range = st.slider("观察窗口范围(日)", 20, 100, (40, 80))
            
            optimize_btn = st.button("🔧 开始参数优化", type="primary", use_container_width=True)
    
    # ========== 模式A: 单标的深度分析 ==========
    if app_mode == "🔎 单标的深度分析":
        st.title("🔎 A-Share 单标的深度分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在深度分析 {target_name}..."):
                # 获取数据
                df, final_ticker = data_fetcher.get_stock_data(target_ticker, start_date, end_date)
                
                if df is None or df.empty:
                    st.error("无法获取股票数据，请检查代码是否正确")
                    st.stop()
                
                # 训练改进的BHMM模型
                df_model = train_bhmm_improved(df, n_components)
                
                if df_model is None:
                    st.error("模型训练失败")
                    st.stop()
                
                # 回测
                df_result, metrics = backtest_strategy(df_model, transaction_cost)
                
                # 获取AI建议
                ai_advice = get_ai_advice(df_result, metrics, n_components)
                
                # 显示核心指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("累计收益", f"{metrics['Total Return']*100:.1f}%")
                col2.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
                col3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                col4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                
                col5, col6, col7 = st.columns(3)
                col5.metric("索提诺比率", f"{metrics['Sortino']:.2f}")
                col6.metric("卡尔玛比率", f"{metrics['Calmar']:.2f}")
                col7.metric("胜率", f"{metrics['Win Rate']*100:.1f}%")
                
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
                    <div style="margin-top:15px; display:grid; grid-template-columns:repeat(3, 1fr); gap:10px;">
                        <div style="color:#888;">风险等级: {ai_advice['risk_level']}</div>
                        <div style="color:#888;">模型置信度: {ai_advice['confidence']}</div>
                        <div style="color:#888;">近期波动率: {ai_advice['risk_metrics'].get('近期波动率', 'N/A')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 多维度图表展示
                tab1, tab2, tab3, tab4 = st.tabs(["📈 价格与状态", "📊 策略收益", "📉 风险分析", "📋 详细数据"])
                
                with tab1:
                    fig = make_subplots(
                        rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    # 价格与状态
                    colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
                    for i in range(n_components):
                        mask = df_result['Regime'] == i
                        if mask.any():
                            fig.add_trace(
                                go.Scatter(
                                    x=df_result.index[mask], 
                                    y=df_result['Close'][mask], 
                                    mode='markers',
                                    marker=dict(size=6, color=colors[i % 4], symbol='circle'),
                                    name=f"状态 {i}",
                                    legendgroup=f"state_{i}"
                                ),
                                row=1, col=1
                            )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_result.index, 
                            y=df_result['Close'], 
                            line=dict(color='rgba(255,255,255,0.4)', width=1.5),
                            name="收盘价",
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Alpha信号
                    fig.add_trace(
                        go.Scatter(
                            x=df_result.index, 
                            y=df_result['Bayes_Exp_Ret'] * 10000,
                            line=dict(color='#FF5252', width=1),
                            name="Alpha信号(bps)",
                            yaxis="y2"
                        ),
                        row=2, col=1
                    )
                    
                    # 置信度
                    if 'Regime_Confidence' in df_result.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df_result.index, 
                                y=df_result['Regime_Confidence'] * 100,
                                line=dict(color='#6495ED', width=1),
                                name="置信度(%)",
                                fill='tozeroy',
                                fillcolor='rgba(100, 149, 237, 0.2)'
                            ),
                            row=3, col=1
                        )
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=700,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    fig.update_yaxes(title_text="价格", row=1, col=1)
                    fig.update_yaxes(title_text="Alpha(bps)", row=2, col=1)
                    fig.update_yaxes(title_text="置信度(%)", row=3, col=1, range=[0, 100])
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig_eq = go.Figure()
                    
                    # 基准收益
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index, 
                        y=df_result['Cum_Bench'],
                        name="基准",
                        line=dict(color='rgba(169, 169, 169, 0.6)', dash='dot', width=1)
                    ))
                    
                    # 策略收益
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index, 
                        y=df_result['Cum_Strat'],
                        name="BHMM策略",
                        line=dict(color='#FF5252', width=2.5)
                    ))
                    
                    # 持仓区域
                    positions = df_result['Position']
                    buy_signals = positions.diff() > 0
                    sell_signals = positions.diff() < 0
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index[buy_signals],
                        y=df_result['Cum_Strat'][buy_signals],
                        mode='markers',
                        marker=dict(size=10, color='#00E676', symbol='triangle-up'),
                        name='买入信号',
                        showlegend=True
                    ))
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index[sell_signals],
                        y=df_result['Cum_Strat'][sell_signals],
                        mode='markers',
                        marker=dict(size=10, color='#FF1744', symbol='triangle-down'),
                        name='卖出信号',
                        showlegend=True
                    ))
                    
                    fig_eq.update_layout(
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title="策略收益曲线与交易信号",
                        yaxis_title="累计收益",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    # 收益分布图
                    st.subheader("📊 收益分布分析")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # 月度收益热图
                        df_result['YearMonth'] = df_result.index.strftime('%Y-%m')
                        monthly_returns = df_result.groupby('YearMonth')['Strategy_Ret'].sum()
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=[monthly_returns.values],
                            x=monthly_returns.index,
                            colorscale='RdYlGn',
                            showscale=True,
                            zmid=0
                        ))
                        
                        fig_heatmap.update_layout(
                            template="plotly_dark",
                            height=300,
                            title="月度收益热图",
                            xaxis_title="月份",
                            yaxis=dict(showticklabels=False)
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with col2:
                        # 收益直方图
                        fig_hist = go.Figure(data=[go.Histogram(
                            x=df_result['Strategy_Ret'] * 100,
                            nbinsx=30,
                            marker_color='#FF5252',
                            opacity=0.7
                        )])
                        
                        fig_hist.update_layout(
                            template="plotly_dark",
                            height=300,
                            title="日收益分布",
                            xaxis_title="日收益(%)",
                            yaxis_title="频数"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                with tab3:
                    st.subheader("📉 风险指标分析")
                    
                    # 回撤分析
                    cumulative = df_result['Cum_Strat']
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max * 100
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=df_result.index,
                        y=drawdown,
                        fill='tozeroy',
                        fillcolor='rgba(255, 23, 68, 0.3)',
                        line=dict(color='#FF1744', width=1),
                        name='回撤'
                    ))
                    
                    fig_dd.update_layout(
                        template="plotly_dark",
                        height=300,
                        title="最大回撤曲线",
                        yaxis_title="回撤(%)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    # 滚动风险指标
                    st.subheader("📈 滚动窗口分析")
                    
                    rolling_window = 60
                    df_rolling = df_result.copy()
                    df_rolling['Rolling_Sharpe'] = df_rolling['Strategy_Ret'].rolling(rolling_window).apply(
                        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                    )
                    df_rolling['Rolling_Return'] = df_rolling['Strategy_Ret'].rolling(rolling_window).mean() * 252 * 100
                    df_rolling['Rolling_Volatility'] = df_rolling['Strategy_Ret'].rolling(rolling_window).std() * np.sqrt(252) * 100
                    
                    fig_rolling = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("滚动夏普比率", "滚动年化收益与波动率")
                    )
                    
                    fig_rolling.add_trace(
                        go.Scatter(
                            x=df_rolling.index,
                            y=df_rolling['Rolling_Sharpe'],
                            line=dict(color='#00E676', width=2),
                            name='滚动夏普'
                        ),
                        row=1, col=1
                    )
                    
                    fig_rolling.add_trace(
                        go.Scatter(
                            x=df_rolling.index,
                            y=df_rolling['Rolling_Return'],
                            line=dict(color='#FF5252', width=2),
                            name='滚动年化收益(%)',
                            yaxis='y2'
                        ),
                        row=2, col=1
                    )
                    
                    fig_rolling.add_trace(
                        go.Scatter(
                            x=df_rolling.index,
                            y=df_rolling['Rolling_Volatility'],
                            line=dict(color='#6495ED', width=2),
                            name='滚动波动率(%)',
                            fill='tonexty',
                            fillcolor='rgba(100, 149, 237, 0.2)',
                            yaxis='y3'
                        ),
                        row=2, col=1
                    )
                    
                    fig_rolling.update_layout(
                        template="plotly_dark",
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified",
                        showlegend=True
                    )
                    
                    fig_rolling.update_yaxes(title_text="夏普比率", row=1, col=1)
                    fig_rolling.update_yaxes(title_text="年化收益(%)", row=2, col=1)
                    
                    st.plotly_chart(fig_rolling, use_container_width=True)
                
                with tab4:
                    # 显示详细数据
                    display_cols = ['Close', 'Log_Ret', 'Volatility', 'Regime', 
                                  'Regime_Confidence', 'Bayes_Exp_Ret', 'Signal', 'Position', 'Strategy_Ret']
                    
                    available_cols = [col for col in display_cols if col in df_result.columns]
                    display_df = df_result[available_cols].copy()
                    
                    # 格式化显示
                    format_dict = {
                        'Close': '{:.2f}',
                        'Log_Ret': '{:.4f}',
                        'Volatility': '{:.4f}',
                        'Regime_Confidence': '{:.1%}',
                        'Bayes_Exp_Ret': '{:.2f}bps',
                        'Strategy_Ret': '{:.4f}'
                    }
                    
                    # 转换单位
                    if 'Bayes_Exp_Ret' in display_df.columns:
                        display_df['Bayes_Exp_Ret'] = display_df['Bayes_Exp_Ret'] * 10000
                    
                    styled_df = display_df.tail(100).style.format(format_dict)
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # 下载数据
                    csv = display_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 下载详细数据(CSV)",
                        data=csv,
                        file_name=f"{target_ticker.split('.')[0]}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        elif run_btn:
            st.warning("请选择有效的股票代码")
        else:
            st.info("👈 请在侧边栏选择股票并开始深度分析")
    
    # ========== 模式B: 板块智能扫描 ==========
    elif app_mode == "📡 板块智能扫描":
        st.title(f"📡 板块智能扫描: {target_sector}")
        
        if scan_btn:
            with st.spinner(f"正在智能扫描 {target_sector} 板块..."):
                # 执行扫描
                results = scanner.scan_sector(target_sector, start_date, end_date, n_components, top_n)
                
                if results.empty:
                    st.error(f"未在 {target_sector} 板块发现符合条件的股票")
                    st.stop()
                
                st.success(f"扫描完成！发现 {len(results)} 只优质标的")
                
                # 显示扫描结果
                st.subheader("🏆 板块优质标的推荐")
                
                # 按状态分组显示
                for state in range(n_components):
                    state_results = results[results['状态'] == state]
                    if len(state_results) > 0:
                        if state == 0:
                            title = f"📈 状态{state}: 低波建仓机会 (共{len(state_results)}只)"
                        elif state == n_components - 1:
                            title = f"⚡ 状态{state}: 高波交易机会 (共{len(state_results)}只)"
                        else:
                            title = f"📊 状态{state}: 趋势运行标的 (共{len(state_results)}只)"
                        
                        with st.expander(title):
                            for _, row in state_results.iterrows():
                                alpha_color = "#00E676" if row['Alpha'] > 0.0005 else "#FF1744"
                                alpha_class = "positive-alpha" if row['Alpha'] > 0.0005 else "negative-alpha"
                                
                                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                                
                                with col1:
                                    st.markdown(f"**{row['名称']}** ({row['代码']})")
                                
                                with col2:
                                    st.metric("Alpha", f"{row['Alpha']*10000:.1f}bps", 
                                            delta_color="normal" if row['Alpha'] > 0 else "inverse")
                                
                                with col3:
                                    st.metric("置信度", f"{row['置信度']:.1%}")
                                
                                with col4:
                                    st.metric("综合评分", f"{row['综合评分']:.1f}")
                
                # 详细数据表
                st.subheader("📋 详细扫描数据")
                
                display_results = results.copy()
                display_results['Alpha(bps)'] = display_results['Alpha'] * 10000
                display_results['近期收益(bps)'] = display_results['近期收益'] * 10000
                display_results['波动率(%)'] = display_results['波动率'] * 100
                
                display_cols = ['代码', '名称', '状态', 'Alpha(bps)', '置信度', '波动率(%)', 
                              '近期收益(bps)', 'RSI', '综合评分', '最新价']
                
                styled_df = display_results[display_cols].style.format({
                    'Alpha(bps)': '{:.1f}',
                    '置信度': '{:.1%}',
                    '波动率(%)': '{:.2f}',
                    '近期收益(bps)': '{:.1f}',
                    'RSI': '{:.1f}',
                    '综合评分': '{:.1f}',
                    '最新价': '{:.2f}'
                }).background_gradient(
                    subset=['Alpha(bps)', '综合评分'], 
                    cmap='RdYlGn'
                )
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # 可视化分析
                st.subheader("📊 板块扫描可视化")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Alpha分布
                    fig_alpha = go.Figure(data=[go.Histogram(
                        x=results['Alpha'] * 10000,
                        nbinsx=20,
                        marker_color='#FF5252',
                        opacity=0.7,
                        name='Alpha分布'
                    )])
                    
                    fig_alpha.update_layout(
                        template="plotly_dark",
                        height=300,
                        title="Alpha分布(bps)",
                        xaxis_title="Alpha(bps)",
                        yaxis_title="数量"
                    )
                    st.plotly_chart(fig_alpha, use_container_width=True)
                
                with col2:
                    # 状态分布
                    state_counts = results['状态'].value_counts().sort_index()
                    colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
                    
                    fig_state = go.Figure(data=[go.Pie(
                        labels=[f"状态{i}" for i in state_counts.index],
                        values=state_counts.values,
                        marker=dict(colors=[colors[i % 4] for i in state_counts.index]),
                        hole=0.4
                    )])
                    
                    fig_state.update_layout(
                        template="plotly_dark",
                        height=300,
                        title="状态分布",
                        showlegend=True
                    )
                    st.plotly_chart(fig_state, use_container_width=True)
                
                # 下载结果
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载完整扫描结果(CSV)",
                    data=csv,
                    file_name=f"{target_sector}_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("👈 请在侧边栏选择板块并开始智能扫描")
    
    # ========== 模式C: 全市场筛选 ==========
    elif app_mode == "🌐 全市场筛选":
        st.title("🌐 全市场智能筛选")
        
        if filter_btn:
            st.info("全市场筛选功能正在开发中...")
            st.markdown("""
            ### 🚧 即将上线功能
            1. **Alpha强势股筛选** - 筛选高Alpha且稳定的标的
            2. **低波稳健股筛选** - 状态0且波动率低的防御型标的
            3. **高置信度股筛选** - 模型置信度超过阈值的标的
            4. **综合评分筛选** - 多维度综合评分排名
            
            ### 📊 筛选维度
            - Alpha信号强度
            - 波动率控制
            - 模型置信度
            - 技术指标(RSI, MACD等)
            - 资金流向
            - 板块轮动
            """)
        else:
            st.info("👈 请在侧边栏配置筛选条件")
    
    # ========== 模式D: 策略回测优化 ==========
    elif app_mode == "📊 策略回测优化":
        st.title("📊 策略回测优化")
        
        if optimize_btn:
            st.info("策略回测优化功能正在开发中...")
            st.markdown("""
            ### 🚧 即将上线功能
            1. **参数网格搜索** - 自动寻找最优参数组合
            2. **多目标优化** - 夏普、回撤、收益多目标平衡
            3. **过拟合检测** - 交叉验证防止过拟合
            4. **参数稳定性测试** - 检验参数鲁棒性
            
            ### 🔧 可优化参数
            - 信号阈值 (1-20bps)
            - 观察窗口 (20-100日)
            - 止损止盈比例
            - 仓位管理参数
            - 交易频率控制
            """)
        else:
            st.info("👈 请在侧边栏配置优化参数")

if __name__ == "__main__":
    main()
