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
from tenacity import retry, stop_after_attempt, wait_exponential
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
# 1. 智能数据获取系统
# ==========================================

class DataFetcher:
    """智能数据获取器"""
    
    def __init__(self):
        self.cache_dir = ".data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def format_ticker_for_yfinance(self, raw_code: str, raw_name: str = "Unknown") -> Tuple[str, str]:
        """格式化股票代码为yfinance格式"""
        raw_code = str(raw_code).strip()
        
        # 移除可能的后缀
        if '.' in raw_code:
            raw_code = raw_code.split('.')[0]
        
        # 根据代码开头判断交易所
        if raw_code.startswith("6") or raw_code.startswith("9"): 
            suffix = ".SS"
        elif raw_code.startswith("0") or raw_code.startswith("3"): 
            suffix = ".SZ"
        elif raw_code.startswith("4") or raw_code.startswith("8"): 
            suffix = ".BJ"
        else: 
            suffix = ".SS"  # 默认上海交易所
        
        return f"{raw_code}{suffix}", raw_name
    
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
                        # 检查缓存是否过期（7天）
                        cache_time = cached_data.get('timestamp', 0)
                        if time.time() - cache_time < 7*24*3600:
                            return cached_data['df'], cached_data.get('ticker', ticker)
            except:
                pass
        
        try:
            # 尝试多个数据源
            df = self._try_yfinance(ticker, start, end)
            
            if df is None or df.empty or len(df) < 60:
                # 尝试切换后缀
                base_code = ticker.split('.')[0]
                if len(ticker.split('.')) > 1:
                    current_suffix = '.' + ticker.split('.')[1]
                    alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
                    alt_ticker = base_code + alt_suffix
                    df = self._try_yfinance(alt_ticker, start, end)
                    if df is not None and not df.empty and len(df) >= 60:
                        ticker = alt_ticker
            
            if df is None or df.empty or len(df) < 60:
                return None, ticker
            
            # 特征工程
            data = df[['Close', 'High', 'Low', 'Volume']].copy()
            data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
            if 'Volume' in data.columns:
                data['Vol_Change'] = (data['Volume'] - data['Volume'].rolling(window=5).mean()) / data['Volume'].rolling(window=5).mean()
            data.dropna(inplace=True)
            
            # 缓存数据
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'df': data, 
                        'ticker': ticker,
                        'timestamp': time.time()
                    }, f)
            except:
                pass
            
            return data, ticker
            
        except Exception as e:
            return None, ticker
    
    def _try_yfinance(self, ticker: str, start: str, end: str):
        """尝试yfinance数据源"""
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", 
                           progress=False, auto_adjust=True, timeout=10)
            
            if isinstance(df.columns, pd.MultiIndex):
                try: 
                    df.columns = df.columns.get_level_values(0)
                except: 
                    pass
            
            return df
        except:
            return None
    
    def get_predefined_sectors(self):
        """获取预定义板块信息"""
        sectors = {
            "白酒": ["000858", "600519", "002304", "000568", "600809"],
            "半导体": ["688981", "002049", "603501", "300661", "002371"],
            "新能源": ["300750", "002594", "002812", "002460", "300014"],
            "医药": ["600276", "300760", "300015", "000538", "600085"],
            "金融": ["601318", "600036", "601398", "601166", "600030"],
            "消费": ["600887", "000651", "000333", "603288", "002557"],
            "科技": ["002415", "002475", "300059", "300033", "002230"],
            "光伏设备": ["601012", "300274", "002129", "688303", "300118"],
            "汽车整车": ["601633", "600104", "000625", "002594", "601238"],
            "军工": ["600893", "600760", "002179", "000768", "600862"],
        }
        return sectors
    
    def get_sector_stocks(self, sector_name: str):
        """根据板块名称返回预设的股票列表"""
        sectors = self.get_predefined_sectors()
        sector_map = {
            "白酒": [("000858", "五粮液"), ("600519", "贵州茅台"), ("002304", "洋河股份"), 
                   ("000568", "泸州老窖"), ("600809", "山西汾酒")],
            "半导体": [("688981", "中芯国际"), ("002049", "紫光国微"), ("603501", "韦尔股份"), 
                     ("300661", "圣邦股份"), ("002371", "北方华创")],
            "新能源": [("300750", "宁德时代"), ("002594", "比亚迪"), ("002812", "恩捷股份"), 
                     ("002460", "赣锋锂业"), ("300014", "亿纬锂能")],
            "医药": [("600276", "恒瑞医药"), ("300760", "迈瑞医疗"), ("300015", "爱尔眼科"), 
                   ("000538", "云南白药"), ("600085", "同仁堂")],
            "金融": [("601318", "中国平安"), ("600036", "招商银行"), ("601398", "工商银行"), 
                   ("601166", "兴业银行"), ("600030", "中信证券")],
            "消费": [("600887", "伊利股份"), ("000651", "格力电器"), ("000333", "美的集团"), 
                   ("603288", "海天味业"), ("002557", "洽洽食品")],
            "科技": [("002415", "海康威视"), ("002475", "立讯精密"), ("300059", "东方财富"), 
                   ("300033", "同花顺"), ("002230", "科大讯飞")],
            "光伏设备": [("601012", "隆基绿能"), ("300274", "阳光电源"), ("002129", "TCL中环"), 
                      ("688303", "大全能源"), ("300118", "东方日升")],
            "汽车整车": [("601633", "长城汽车"), ("600104", "上汽集团"), ("000625", "长安汽车"), 
                      ("002594", "比亚迪"), ("601238", "广汽集团")],
            "军工": [("600893", "航发动力"), ("600760", "中航沈飞"), ("002179", "中航光电"), 
                   ("000768", "中航西飞"), ("600862", "中航高科")],
        }
        return sector_map.get(sector_name, [])

# 初始化数据获取器
data_fetcher = DataFetcher()

# ==========================================
# 2. 改进的贝叶斯HMM模型
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
# 3. 回测系统 (修复胜率计算)
# ==========================================

def backtest_strategy(df: pd.DataFrame, cost: float = 0.001) -> Tuple[pd.DataFrame, Dict]:
    """回测策略 - 修复胜率计算"""
    threshold = 0.0005  # 5bps
    
    # 生成信号
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    
    # 计算仓位
    df['Position'] = df['Signal'].shift(1).fillna(0)
    
    # 计算交易成本
    t_cost = df['Position'].diff().abs() * cost
    
    # 计算策略收益
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    
    # 计算累计收益
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    # === 修复胜率计算 ===
    # 正确的交易识别方式：仓位变化表示交易
    position_changes = df['Position'].diff().fillna(0)
    buy_signals = position_changes > 0  # 买入信号
    sell_signals = position_changes < 0  # 卖出信号
    
    # 计算交易结果
    trades = []
    entry_price = None
    entry_date = None
    
    for i in range(1, len(df)):
        if buy_signals.iloc[i] and entry_price is None:  # 开仓
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
        
        elif sell_signals.iloc[i] and entry_price is not None:  # 平仓
            exit_price = df['Close'].iloc[i]
            exit_date = df.index[i]
            trade_return = (exit_price - entry_price) / entry_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'return': trade_return,
                'winning': trade_return > 0
            })
            entry_price = None
            entry_date = None
    
    # 如果有未平仓的交易，按最后一天结算
    if entry_price is not None:
        exit_price = df['Close'].iloc[-1]
        exit_date = df.index[-1]
        trade_return = (exit_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'return': trade_return,
            'winning': trade_return > 0
        })
    
    # 计算胜率
    if trades:
        winning_trades = sum(1 for trade in trades if trade['winning'])
        win_rate = winning_trades / len(trades)
        total_trades = len(trades)
    else:
        win_rate = 0
        total_trades = 0
    
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
    
    # 索提诺比率
    negative_returns = df['Strategy_Ret'][df['Strategy_Ret'] < 0]
    if len(negative_returns) > 0 and negative_returns.std() != 0:
        sortino = (df['Strategy_Ret'].mean() * 252) / (negative_returns.std() * np.sqrt(252))
    else:
        sortino = sharpe
    
    # 卡尔玛比率
    if max_dd != 0:
        calmar = annual_ret / abs(max_dd)
    else:
        calmar = 0
    
    return df, {
        "Total Return": total_ret,
        "CAGR": annual_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Total Trades": total_trades,
        "Avg Trade Return": np.mean([t['return'] for t in trades]) if trades else 0,
        "Max Win": max([t['return'] for t in trades]) if trades else 0,
        "Max Loss": min([t['return'] for t in trades]) if trades else 0
    }

# ==========================================
# 4. AI 投顾
# ==========================================

def get_ai_advice(df: pd.DataFrame, metrics: Dict, n_comps: int) -> Dict:
    """获取AI投顾建议"""
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
    
    advice['risk_metrics'] = {
        "近期波动率": f"{recent_volatility:.2%}",
        "模型置信度": f"{last_confidence:.1%}",
        "Alpha信号": f"{last_alpha*10000:.1f}bps"
    }
    
    if last_regime == 0:  # 低波动状态
        advice['risk_level'] = "低风险"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = f"低波动稳态，预期Alpha: {last_alpha*10000:.1f}bps > 阈值5bps"
            advice['action'] = "建议：分批买入，设置止损-3%"
            advice['position'] = "70-90%"
        else:
            advice['title'] = "🟡 观望/防守"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = f"低波动但预期收益不足 (Alpha: {last_alpha*10000:.1f}bps)"
            advice['action'] = "建议：轻仓观察(10-20%)"
            advice['position'] = "10-20%"
            
    elif last_regime == n_comps - 1:  # 高波动状态
        advice['risk_level'] = "高风险"
        if last_alpha > threshold:
            advice['title'] = "🔵 高风险机会"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"高波动中隐含机会，Alpha: {last_alpha*10000:.1f}bps"
            advice['action'] = "建议：小仓位试探(20-30%)，严格止损-5%"
            advice['position'] = "20-30%"
        else:
            advice['title'] = "🔴 极度风险预警"
            advice['color'] = "#FF1744"
            advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
            advice['summary'] = "剧烈波动模式，下跌风险极高"
            advice['action'] = "建议：清仓避险"
            advice['position'] = "0%"
    else:  # 中间状态
        advice['risk_level'] = "中风险"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"趋势运行中，Alpha: {last_alpha*10000:.1f}bps"
            advice['action'] = "建议：持有为主(50-70%)"
            advice['position'] = "50-70%"
        else:
            advice['title'] = "🟠 减仓观望"
            advice['color'] = "#FF9100"
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "上涨动能衰竭"
            advice['action'] = "建议：逐步减仓至20-30%"
            advice['position'] = "20-30%"
    
    return advice

# ==========================================
# 5. 主程序逻辑
# ==========================================

def main():
    # 侧边栏通用配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share Pro Plus")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 自选股票分析", "📡 板块智能扫描"], 
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
        if app_mode == "🔎 自选股票分析":
            st.caption("自选股票分析")
            
            # 两种输入方式：手动输入或从预设列表选择
            input_mode = st.radio("输入方式", ["手动输入", "从列表选择"], index=0)
            
            if input_mode == "手动输入":
                # 自由输入股票代码
                stock_input = st.text_input(
                    "输入股票代码",
                    value="000858",
                    help="支持格式：000858、000858.SZ、SZ000858"
                )
                
                if stock_input:
                    # 清理输入
                    code = stock_input.strip().upper()
                    if code.startswith('SZ'):
                        code = code[2:] + '.SZ'
                    elif code.startswith('SH'):
                        code = code[2:] + '.SS'
                    elif '.' not in code:
                        # 根据开头判断
                        if code.startswith('6'):
                            code = code + '.SS'
                        else:
                            code = code + '.SZ'
                    
                    target_ticker, target_name = data_fetcher.format_ticker_for_yfinance(
                        code.split('.')[0] if '.' in code else code,
                        f"股票{code}"
                    )
                else:
                    target_ticker, target_name = None, None
                    
            else:  # 从列表选择
                # 预设的常用股票列表
                preset_stocks = [
                    ("000858", "五粮液"),
                    ("600519", "贵州茅台"),
                    ("000651", "格力电器"),
                    ("000333", "美的集团"),
                    ("300750", "宁德时代"),
                    ("002594", "比亚迪"),
                    ("601318", "中国平安"),
                    ("600036", "招商银行"),
                    ("600276", "恒瑞医药"),
                    ("300760", "迈瑞医疗"),
                    ("002415", "海康威视"),
                    ("002475", "立讯精密"),
                    ("688981", "中芯国际"),
                    ("601012", "隆基绿能"),
                    ("000002", "万科A"),
                ]
                
                stock_options = [f"{code} | {name}" for code, name in preset_stocks]
                selected_stock = st.selectbox("选择股票", options=stock_options)
                
                if selected_stock:
                    code, name = selected_stock.split(" | ")
                    target_ticker, target_name = data_fetcher.format_ticker_for_yfinance(code, name)
                else:
                    target_ticker, target_name = None, None
            
            # 高级参数
            with st.expander("高级参数"):
                rolling_window = st.slider("滚动窗口(日)", 30, 120, 60)
                signal_threshold = st.number_input("信号阈值(bps)", value=5.0, min_value=0.1, max_value=20.0) / 10000
            
            run_btn = st.button("🚀 开始深度分析", type="primary", use_container_width=True)
            
        elif app_mode == "📡 板块智能扫描":
            st.caption("板块智能扫描")
            SECTORS = list(data_fetcher.get_predefined_sectors().keys())
            target_sector = st.selectbox("选择板块", SECTORS)
            
            with st.expander("扫描配置"):
                top_n = st.slider("显示数量", 5, 20, 10)
                min_confidence = st.slider("最小置信度(%)", 50, 90, 70) / 100
            
            scan_btn = st.button("📡 开始智能扫描", type="primary", use_container_width=True)
    
    # ========== 模式A: 自选股票分析 ==========
    if app_mode == "🔎 自选股票分析":
        st.title("🔎 A-Share 自选股票深度分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在深度分析 {target_name if target_name else target_ticker}..."):
                # 获取数据
                df, final_ticker = data_fetcher.get_stock_data(target_ticker, start_date, end_date)
                
                if df is None or df.empty:
                    st.error(f"无法获取股票数据，请检查代码 {target_ticker} 是否正确")
                    st.stop()
                
                if len(df) < 100:
                    st.warning(f"数据量较少({len(df)}天)，分析结果可能不准确")
                
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
                col5.metric("胜率", f"{metrics['Win Rate']*100:.1f}%")
                col6.metric("交易次数", f"{metrics['Total Trades']}")
                col7.metric("平均收益", f"{metrics['Avg Trade Return']*100:.1f}%")
                
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
                        <div style="color:#888;">Alpha信号: {ai_advice['risk_metrics'].get('Alpha信号', 'N/A')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 多维度图表展示
                tab1, tab2, tab3 = st.tabs(["📈 价格与状态", "📊 策略收益", "📉 风险分析"])
                
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
                                    name=f"状态 {i}"
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
                            name="Alpha信号(bps)"
                        ),
                        row=2, col=1
                    )
                    
                    # 添加阈值线
                    fig.add_hline(y=5, line=dict(color="white", width=1, dash="dash"), 
                                 row=2, col=1, annotation_text="阈值 5bps")
                    
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
                        showlegend=True
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
                    
                    # 识别交易信号点
                    position_changes = df_result['Position'].diff().fillna(0)
                    buy_points = position_changes > 0
                    sell_points = position_changes < 0
                    
                    # 买入信号
                    if buy_points.any():
                        fig_eq.add_trace(go.Scatter(
                            x=df_result.index[buy_points],
                            y=df_result['Cum_Strat'][buy_points],
                            mode='markers',
                            marker=dict(size=10, color='#00E676', symbol='triangle-up'),
                            name='买入信号',
                            showlegend=True
                        ))
                    
                    # 卖出信号
                    if sell_points.any():
                        fig_eq.add_trace(go.Scatter(
                            x=df_result.index[sell_points],
                            y=df_result['Cum_Strat'][sell_points],
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
                    
                    # 交易统计
                    st.subheader("📊 交易统计")
                    
                    if metrics['Total Trades'] > 0:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("最大盈利", f"{metrics['Max Win']*100:.1f}%")
                        col2.metric("最大亏损", f"{metrics['Max Loss']*100:.1f}%")
                        col3.metric("盈亏比", 
                                  f"{(metrics['Avg Trade Return'] if metrics['Avg Trade Return'] > 0 else 0) / abs(metrics['Max Loss']) if metrics['Max Loss'] < 0 else 'N/A':.2f}")
                    
                    # 月度收益分析
                    st.subheader("📅 月度收益分析")
                    
                    df_monthly = df_result.copy()
                    df_monthly['YearMonth'] = df_monthly.index.strftime('%Y-%m')
                    monthly_returns = df_monthly.groupby('YearMonth')['Strategy_Ret'].sum()
                    
                    fig_monthly = go.Figure(data=[go.Bar(
                        x=monthly_returns.index,
                        y=monthly_returns.values * 100,
                        marker_color=np.where(monthly_returns.values > 0, '#00E676', '#FF1744'),
                        text=[f'{x:.1f}%' for x in monthly_returns.values * 100],
                        textposition='auto',
                    )])
                    
                    fig_monthly.update_layout(
                        template="plotly_dark",
                        height=400,
                        title="月度收益",
                        xaxis_title="月份",
                        yaxis_title="收益(%)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                with tab3:
                    st.subheader("📉 风险分析")
                    
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
                    st.subheader("📈 滚动窗口风险指标")
                    
                    rolling_window = 60
                    df_rolling = df_result.copy()
                    
                    # 滚动夏普
                    df_rolling['Rolling_Sharpe'] = df_rolling['Strategy_Ret'].rolling(rolling_window).apply(
                        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                    )
                    
                    # 滚动最大回撤
                    df_rolling['Rolling_Cum'] = (1 + df_rolling['Strategy_Ret']).rolling(rolling_window).apply(lambda x: x.prod())
                    df_rolling['Rolling_Max'] = df_rolling['Rolling_Cum'].rolling(rolling_window, min_periods=1).max()
                    df_rolling['Rolling_DD'] = (df_rolling['Rolling_Cum'] - df_rolling['Rolling_Max']) / df_rolling['Rolling_Max'] * 100
                    
                    fig_rolling = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("滚动夏普比率", "滚动最大回撤")
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
                            y=df_rolling['Rolling_DD'],
                            line=dict(color='#FF5252', width=2),
                            name='滚动最大回撤(%)',
                            fill='tozeroy',
                            fillcolor='rgba(255, 82, 82, 0.2)'
                        ),
                        row=2, col=1
                    )
                    
                    fig_rolling.update_layout(
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified"
                    )
                    
                    fig_rolling.update_yaxes(title_text="夏普比率", row=1, col=1)
                    fig_rolling.update_yaxes(title_text="回撤(%)", row=2, col=1)
                    
                    st.plotly_chart(fig_rolling, use_container_width=True)
        
        elif run_btn:
            st.warning("请输入有效的股票代码")
        else:
            st.info("👈 请在侧边栏输入或选择股票代码并开始分析")
    
    # ========== 模式B: 板块智能扫描 ==========
    elif app_mode == "📡 板块智能扫描":
        st.title(f"📡 板块智能扫描: {target_sector}")
        
        if scan_btn:
            with st.spinner(f"正在扫描 {target_sector} 板块..."):
                # 获取板块股票
                sector_stocks = data_fetcher.get_sector_stocks(target_sector)
                
                if not sector_stocks:
                    st.error(f"未找到 {target_sector} 板块数据")
                    st.stop()
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, (code, name) in enumerate(sector_stocks):
                    try:
                        ticker, _ = data_fetcher.format_ticker_for_yfinance(code, name)
                        df, _ = data_fetcher.get_stock_data(ticker, start_date, end_date)
                        
                        if df is not None and len(df) > 100:
                            df_model = train_bhmm_improved(df, n_components)
                            
                            if df_model is not None:
                                last_regime = int(df_model['Regime'].iloc[-1])
                                last_alpha = df_model['Bayes_Exp_Ret'].iloc[-1]
                                confidence = df_model['Regime_Confidence'].iloc[-1] if 'Regime_Confidence' in df_model.columns else 0
                                
                                # 计算技术指标
                                recent_vol = df['Volatility'].iloc[-20:].mean() if len(df) >= 20 else df['Volatility'].mean()
                                recent_ret = df['Log_Ret'].iloc[-5:].mean() if len(df) >= 5 else 0
                                
                                # 综合评分
                                score = last_alpha * 10000  # 基础分
                                if last_regime == 0:
                                    score += 20  # 低波动加分
                                if confidence > 0.7:
                                    score += 10  # 高置信度加分
                                if recent_vol < 0.02:
                                    score += 5  # 低波动率加分
                                
                                results.append({
                                    "代码": code,
                                    "名称": name,
                                    "状态": last_regime,
                                    "Alpha(bps)": last_alpha * 10000,
                                    "置信度": confidence,
                                    "波动率": recent_vol,
                                    "近期收益(bps)": recent_ret * 10000,
                                    "综合评分": score,
                                    "最新价": df['Close'].iloc[-1] if 'Close' in df.columns else 0
                                })
                    except:
                        continue
                    
                    progress_bar.progress((idx + 1) / len(sector_stocks))
                
                progress_bar.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('综合评分', ascending=False)
                    
                    st.success(f"扫描完成！发现 {len(results_df)} 只标的")
                    
                    # 显示结果
                    st.subheader("🏆 优质标的推荐")
                    
                    for _, row in results_df.iterrows():
                        state_color = ['#00E676', '#FFD600', '#FF1744', '#AA00FF'][int(row['状态']) % 4]
                        
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                        
                        with col1:
                            st.markdown(f"**{row['名称']}** ({row['代码']})")
                        
                        with col2:
                            st.metric("Alpha", f"{row['Alpha(bps)']:.1f}bps", 
                                    delta_color="normal" if row['Alpha(bps)'] > 0 else "inverse")
                        
                        with col3:
                            st.metric("状态", f"{int(row['状态'])}", 
                                    delta_color="normal" if row['状态'] == 0 else "off")
                        
                        with col4:
                            st.metric("评分", f"{row['综合评分']:.1f}")
                    
                    # 详细数据
                    st.subheader("📋 详细数据")
                    styled_df = results_df.style.format({
                        'Alpha(bps)': '{:.1f}',
                        '置信度': '{:.1%}',
                        '波动率': '{:.4f}',
                        '近期收益(bps)': '{:.1f}',
                        '综合评分': '{:.1f}',
                        '最新价': '{:.2f}'
                    }).background_gradient(
                        subset=['Alpha(bps)', '综合评分'], 
                        cmap='RdYlGn'
                    )
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                else:
                    st.warning("未发现符合条件的标的")
        else:
            st.info("👈 请在侧边栏选择板块并开始扫描")

if __name__ == "__main__":
    main()
