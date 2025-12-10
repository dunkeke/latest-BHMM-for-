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
from tenacity import retry, stop_after_attempt, wait_exponential

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
    .alert-high { border-left: 4px solid #FF1744 !important; background: rgba(255, 23, 68, 0.1) !important; }
    .alert-medium { border-left: 4px solid #FF9100 !important; background: rgba(255, 145, 0, 0.1) !important; }
    .alert-low { border-left: 4px solid #00E676 !important; background: rgba(0, 230, 118, 0.1) !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 智能数据获取系统 (稳定版)
# ==========================================

class DataFetcher:
    """智能数据获取器"""
    
    def __init__(self):
        self.cache_dir = ".data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 预定义股票数据库 (避免网络请求)
        self._init_stock_database()
    
    def _init_stock_database(self):
        """初始化股票数据库"""
        # 核心股票池 (200+只A股龙头)
        self.stock_database = {
            # 白酒
            "白酒": [
                ("000858", "五粮液", 130.5),
                ("600519", "贵州茅台", 1600.0),
                ("002304", "洋河股份", 85.3),
                ("000568", "泸州老窖", 180.2),
                ("600809", "山西汾酒", 210.5),
            ],
            # 半导体
            "半导体": [
                ("688981", "中芯国际", 45.6),
                ("002049", "紫光国微", 85.4),
                ("603501", "韦尔股份", 95.2),
                ("300661", "圣邦股份", 120.8),
                ("002371", "北方华创", 280.5),
            ],
            # 新能源
            "新能源": [
                ("300750", "宁德时代", 180.5),
                ("002594", "比亚迪", 210.3),
                ("002812", "恩捷股份", 45.6),
                ("002460", "赣锋锂业", 35.8),
                ("300014", "亿纬锂能", 38.9),
            ],
            # 医药
            "医药": [
                ("600276", "恒瑞医药", 42.8),
                ("300760", "迈瑞医疗", 285.6),
                ("300015", "爱尔眼科", 15.2),
                ("000538", "云南白药", 52.4),
                ("600085", "同仁堂", 45.6),
            ],
            # 金融
            "金融": [
                ("601318", "中国平安", 42.5),
                ("600036", "招商银行", 32.8),
                ("601398", "工商银行", 4.9),
                ("601166", "兴业银行", 15.6),
                ("600030", "中信证券", 22.4),
            ],
            # 消费
            "消费": [
                ("600887", "伊利股份", 28.5),
                ("000651", "格力电器", 35.6),
                ("000333", "美的集团", 58.9),
                ("603288", "海天味业", 35.8),
                ("002557", "洽洽食品", 32.4),
            ],
            # 科技
            "科技": [
                ("002415", "海康威视", 32.5),
                ("002475", "立讯精密", 28.9),
                ("300059", "东方财富", 13.2),
                ("300033", "同花顺", 105.6),
                ("002230", "科大讯飞", 45.8),
            ],
        }
        
        # 创建全市场列表
        self.all_stocks = []
        for sector, stocks in self.stock_database.items():
            for code, name, price in stocks:
                self.all_stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    '参考价': price
                })
    
    def format_ticker(self, raw_code: str, raw_name: str = "Unknown") -> Tuple[str, str]:
        """格式化股票代码"""
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
            suffix = ".SS"
        
        return f"{raw_code}{suffix}", raw_name
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_stock_data(self, ticker: str, start: str, end: str):
        """获取股票数据（带重试）"""
        cache_key = f"{ticker}_{start}_{end}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # 检查缓存是否过期（3天）
                    cache_time = cached_data.get('timestamp', 0)
                    if time.time() - cache_time < 3*24*3600:
                        return cached_data['df'], cached_data.get('ticker', ticker)
            except:
                pass
        
        try:
            df = yf.download(ticker, start=start, end=end, 
                           progress=False, auto_adjust=True, timeout=15)
            
            if df.empty or len(df) < 10:
                return None, ticker
            
            if isinstance(df.columns, pd.MultiIndex):
                try: 
                    df.columns = df.columns.get_level_values(0)
                except: 
                    pass
            
            if len(df) < 60:
                return None, ticker
            
            # 特征工程
            data = df[['Close', 'Volume']].copy()
            data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
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
    
    def get_market_stocks(self, limit: int = 200) -> pd.DataFrame:
        """获取市场股票列表"""
        return pd.DataFrame(self.all_stocks).head(limit)
    
    def get_sector_stocks(self, sector_name: str):
        """获取板块成分股"""
        return self.stock_database.get(sector_name, [])

# 初始化数据获取器
data_fetcher = DataFetcher()

# ==========================================
# 2. BHMM模型 (稳定版)
# ==========================================

def train_bhmm_simple(df: pd.DataFrame, n_comps: int = 3) -> Optional[pd.DataFrame]:
    """简化的BHMM训练（稳定优先）"""
    if len(df) < 100:
        return None
    
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(
            n_components=n_comps, 
            covariance_type="diag",  # 使用对角矩阵更稳定
            n_iter=500, 
            random_state=88, 
            tol=0.01
        )
        model.fit(X)
        
        hidden_states = model.predict(X)
        
        # 状态排序（按波动率）
        state_vol_means = []
        for i in range(n_comps):
            if np.sum(hidden_states == i) > 0:
                state_vol_means.append((i, X[hidden_states == i, 1].mean()))
        
        if not state_vol_means:
            return None
        
        sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping.get(s, s) for s in hidden_states])
        
        # 计算状态条件收益率
        state_returns = []
        for i in range(n_comps):
            state_data = df[df['Regime'] == i]['Log_Ret']
            if len(state_data) > 5:
                state_returns.append(state_data.mean())
            else:
                state_returns.append(df['Log_Ret'].mean())
        
        # 简单贝叶斯预期收益
        df['Bayes_Exp_Ret'] = 0
        for i in range(1, len(df)):
            prev_state = df['Regime'].iloc[i-1]
            df.loc[df.index[i], 'Bayes_Exp_Ret'] = state_returns[int(prev_state)]
        
        return df
    except:
        return None

# ==========================================
# 3. 回测系统 (修复胜率)
# ==========================================

def backtest_strategy_simple(df: pd.DataFrame, cost: float = 0.001):
    """简化的回测策略"""
    threshold = 0.0005
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    
    df['Position'] = df['Signal'].shift(1).fillna(0)
    
    # 计算交易
    position_changes = df['Position'].diff().fillna(0)
    trades = []
    
    in_position = False
    entry_idx = None
    
    for i in range(1, len(df)):
        if position_changes.iloc[i] > 0 and not in_position:  # 买入
            in_position = True
            entry_idx = i
        elif position_changes.iloc[i] < 0 and in_position:  # 卖出
            if entry_idx is not None:
                trade_return = (df['Close'].iloc[i] - df['Close'].iloc[entry_idx]) / df['Close'].iloc[entry_idx]
                trades.append({
                    'entry': df.index[entry_idx],
                    'exit': df.index[i],
                    'return': trade_return,
                    'winning': trade_return > 0
                })
            in_position = False
            entry_idx = None
    
    # 计算胜率
    if trades:
        winning_trades = sum(1 for t in trades if t['winning'])
        win_rate = winning_trades / len(trades)
        total_trades = len(trades)
        avg_return = np.mean([t['return'] for t in trades])
    else:
        win_rate = 0
        total_trades = 0
        avg_return = 0
    
    # 计算策略收益
    t_cost = df['Position'].diff().abs() * cost
    df['Strategy_Ret'] = (df['Position'] * df['Log_Ret']) - t_cost
    df['Cum_Bench'] = (1 + df['Log_Ret']).cumprod()
    df['Cum_Strat'] = (1 + df['Strategy_Ret']).cumprod()
    
    total_ret = df['Cum_Strat'].iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / len(df)) - 1
    
    # 最大回撤
    cumulative = df['Cum_Strat']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() if not drawdown.empty else 0
    
    # 夏普比率
    if df['Strategy_Ret'].std() != 0:
        sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252))
    else:
        sharpe = 0
    
    return df, {
        "Total Return": total_ret,
        "CAGR": annual_ret,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Total Trades": total_trades,
        "Avg Trade Return": avg_return
    }

# ==========================================
# 4. 全市场Alpha扫描系统
# ==========================================

class MarketAlphaScanner:
    """全市场Alpha扫描器"""
    
    def __init__(self, data_fetcher):
        self.fetcher = data_fetcher
        self.scan_cache = {}
    
    def scan_market_alpha(self, n_components: int = 3, sample_size: int = 50, 
                         lookback_days: int = 365):
        """扫描全市场Alpha"""
        # 获取市场股票
        market_stocks = self.fetcher.get_market_stocks(sample_size)
        
        if market_stocks.empty:
            return pd.DataFrame()
        
        results = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for idx, row in market_stocks.iterrows():
            code = row['代码']
            name = row['名称']
            
            progress_text.text(f"扫描中: {name}({code}) [{idx+1}/{len(market_stocks)}]")
            
            try:
                # 获取数据
                ticker, _ = self.fetcher.format_ticker(code, name)
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                df, _ = self.fetcher.get_stock_data(ticker, start_date, end_date)
                
                if df is None or len(df) < 100:
                    continue
                
                # 训练HMM
                df_model = train_bhmm_simple(df, n_components)
                
                if df_model is None:
                    continue
                
                # 计算Alpha分数
                alpha_score = self._calculate_alpha_score(df_model)
                
                # 生成信号
                signal = self._generate_signal(df_model, alpha_score)
                
                results.append({
                    '代码': code,
                    '名称': name,
                    '板块': row['板块'],
                    'Alpha分数': alpha_score['total'],
                    '动量分数': alpha_score['momentum'],
                    '价值分数': alpha_score['value'],
                    '质量分数': alpha_score['quality'],
                    '最新状态': int(df_model['Regime'].iloc[-1]) if 'Regime' in df_model.columns else 0,
                    '最新Alpha': df_model['Bayes_Exp_Ret'].iloc[-1] if 'Bayes_Exp_Ret' in df_model.columns else 0,
                    '信号': signal['direction'],
                    '信号强度': signal['strength'],
                    '推荐仓位': signal['position'],
                    '最新价': df['Close'].iloc[-1] if 'Close' in df.columns else 0,
                    '扫描时间': datetime.now().strftime("%H:%M:%S")
                })
                
            except Exception as e:
                continue
            
            progress_bar.progress((idx + 1) / len(market_stocks))
        
        progress_text.empty()
        progress_bar.empty()
        
        if results:
            results_df = pd.DataFrame(results)
            # 缓存结果
            cache_key = f"scan_{datetime.now().strftime('%Y%m%d')}"
            self.scan_cache[cache_key] = results_df
            
            return results_df.sort_values('Alpha分数', ascending=False)
        
        return pd.DataFrame()
    
    def _calculate_alpha_score(self, df):
        """计算Alpha分数"""
        scores = {}
        
        # 1. 动量分数 (20日收益)
        if len(df) > 20:
            momentum_20d = df['Log_Ret'].tail(20).mean() * 20
            scores['momentum'] = self._normalize_score(momentum_20d * 100, -20, 20)
        else:
            scores['momentum'] = 50
        
        # 2. 价值分数 (波动率倒数)
        volatility = df['Volatility'].iloc[-1] if 'Volatility' in df.columns else df['Log_Ret'].std()
        if volatility > 0:
            value_score = 1 / (volatility * 10)  # 低波动得分高
            scores['value'] = self._normalize_score(value_score * 100, 0, 100)
        else:
            scores['value'] = 50
        
        # 3. 质量分数 (夏普比率)
        if df['Log_Ret'].std() > 0:
            sharpe = df['Log_Ret'].mean() / df['Log_Ret'].std() * np.sqrt(252)
            scores['quality'] = self._normalize_score(sharpe * 20, -20, 20)
        else:
            scores['quality'] = 50
        
        # 4. 状态分数 (Regime 0最好)
        if 'Regime' in df.columns:
            last_regime = df['Regime'].iloc[-1]
            regime_score = 100 - (last_regime / 3 * 100)  # Regime 0得100分，Regime 3得0分
            scores['regime'] = regime_score
        else:
            scores['regime'] = 50
        
        # 总分 (加权平均)
        weights = {'momentum': 0.3, 'value': 0.2, 'quality': 0.3, 'regime': 0.2}
        total_score = sum(scores[k] * weights[k] for k in scores.keys())
        
        scores['total'] = total_score
        return scores
    
    def _generate_signal(self, df, alpha_score):
        """生成交易信号"""
        signal = {
            'direction': '持有',
            'strength': 0,
            'position': '观望'
        }
        
        total_score = alpha_score['total']
        last_alpha = df['Bayes_Exp_Ret'].iloc[-1] if 'Bayes_Exp_Ret' in df.columns else 0
        
        if total_score > 70 and last_alpha > 0.0005:
            signal['direction'] = '强烈买入'
            signal['strength'] = 0.9
            signal['position'] = '70-90%'
        elif total_score > 60 and last_alpha > 0.0003:
            signal['direction'] = '买入'
            signal['strength'] = 0.7
            signal['position'] = '50-70%'
        elif total_score > 50:
            signal['direction'] = '谨慎买入'
            signal['strength'] = 0.5
            signal['position'] = '30-50%'
        elif total_score > 40:
            signal['direction'] = '持有'
            signal['strength'] = 0.3
            signal['position'] = '10-30%'
        elif total_score > 30:
            signal['direction'] = '减持'
            signal['strength'] = 0.7
            signal['position'] = '0-10%'
        else:
            signal['direction'] = '卖出'
            signal['strength'] = 0.9
            signal['position'] = '0%'
        
        return signal
    
    def _normalize_score(self, value, min_val, max_val):
        """归一化到0-100分"""
        if max_val == min_val:
            return 50
        normalized = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, normalized))

# ==========================================
# 5. 交易提示系统
# ==========================================

class TradingAlertSystem:
    """交易提示系统"""
    
    def __init__(self):
        self.alerts = []
        self.alert_levels = {
            'critical': '🔴 紧急',
            'high': '🟠 重要',
            'medium': '🟡 关注',
            'low': '🟢 提示'
        }
    
    def generate_alerts(self, scan_results: pd.DataFrame, top_n: int = 10):
        """从扫描结果生成交易提示"""
        if scan_results.empty:
            return []
        
        alerts = []
        
        # 1. 高Alpha机会
        high_alpha = scan_results[scan_results['Alpha分数'] > 70].head(5)
        for _, row in high_alpha.iterrows():
            alerts.append({
                'level': 'high',
                'title': f"高Alpha机会: {row['名称']}({row['代码']})",
                'message': f"Alpha分数: {row['Alpha分数']:.1f}, 信号: {row['信号']}, 推荐仓位: {row['推荐仓位']}",
                'stock_code': row['代码'],
                'stock_name': row['名称'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # 2. 低波动价值股
        low_vol = scan_results[(scan_results['价值分数'] > 70) & (scan_results['Alpha分数'] > 60)]
        low_vol = low_vol.head(3)
        for _, row in low_vol.iterrows():
            alerts.append({
                'level': 'medium',
                'title': f"低波动价值股: {row['名称']}({row['代码']})",
                'message': f"价值分数: {row['价值分数']:.1f}, Alpha分数: {row['Alpha分数']:.1f}",
                'stock_code': row['代码'],
                'stock_name': row['名称'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # 3. 动量突破股
        momentum = scan_results[scan_results['动量分数'] > 70].head(3)
        for _, row in momentum.iterrows():
            alerts.append({
                'level': 'medium',
                'title': f"动量突破: {row['名称']}({row['代码']})",
                'message': f"动量分数: {row['动量分数']:.1f}, 最新状态: {row['最新状态']}",
                'stock_code': row['代码'],
                'stock_name': row['名称'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # 4. 风险预警 (低分股)
        risk_stocks = scan_results[scan_results['Alpha分数'] < 30].head(3)
        for _, row in risk_stocks.iterrows():
            alerts.append({
                'level': 'critical',
                'title': f"风险预警: {row['名称']}({row['代码']})",
                'message': f"Alpha分数: {row['Alpha分数']:.1f}, 建议回避或减仓",
                'stock_code': row['代码'],
                'stock_name': row['名称'],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        return alerts[:top_n]
    
    def display_alerts(self, alerts):
        """显示交易提示"""
        if not alerts:
            st.info("📊 当前无交易提示")
            return
        
        st.subheader("🚨 交易提示")
        
        for alert in alerts:
            level_class = f"alert-{alert['level']}"
            
            st.markdown(f"""
            <div class="scanner-card {level_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0;">{self.alert_levels[alert['level']]} {alert['title']}</h4>
                    <span style="color:#aaa; font-size:0.9em;">{alert['timestamp']}</span>
                </div>
                <p style="color:#ccc; margin-top:10px;">{alert['message']}</p>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <span style="color:#888;">股票: {alert['stock_name']} ({alert['stock_code']})</span>
                    <button onclick="alert('分析功能开发中')" style="background:#FF5252; color:white; border:none; padding:5px 10px; border-radius:4px; cursor:pointer;">
                        查看详情
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 6. 主程序逻辑
# ==========================================

def main():
    # 初始化系统
    scanner = MarketAlphaScanner(data_fetcher)
    alert_system = TradingAlertSystem()
    
    # 侧边栏配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share Pro")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 自选股票分析", "🌐 全市场Alpha扫描", "🚨 交易提示中心"], 
            index=0
        )
        st.divider()
        
        # 通用参数
        n_components = st.slider("隐藏状态数", 2, 4, 3)
        lookback_years = st.slider("回看年限", 1, 3, 2)
        trans_cost_bps = st.number_input("交易成本(bps)", value=10, min_value=0, max_value=50)
        transaction_cost = trans_cost_bps / 10000
        
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        st.divider()
        
        # 模式特定配置
        if app_mode == "🔎 自选股票分析":
            st.caption("自选股票分析")
            
            # 股票输入
            input_mode = st.radio("输入方式", ["手动输入", "常用股票"], index=0)
            
            if input_mode == "手动输入":
                stock_input = st.text_input("股票代码", value="000858", 
                                          help="输入股票代码，如：000858、600519、300750")
                if stock_input:
                    target_ticker, target_name = data_fetcher.format_ticker(stock_input, f"股票{stock_input}")
                else:
                    target_ticker, target_name = None, None
            else:
                common_stocks = [
                    ("000858", "五粮液"),
                    ("600519", "贵州茅台"),
                    ("300750", "宁德时代"),
                    ("601318", "中国平安"),
                    ("600036", "招商银行"),
                    ("002415", "海康威视"),
                    ("600276", "恒瑞医药"),
                    ("002594", "比亚迪"),
                ]
                
                stock_options = [f"{code} | {name}" for code, name in common_stocks]
                selected = st.selectbox("选择股票", options=stock_options)
                
                if selected:
                    code, name = selected.split(" | ")
                    target_ticker, target_name = data_fetcher.format_ticker(code, name)
                else:
                    target_ticker, target_name = None, None
            
            run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
            
        elif app_mode == "🌐 全市场Alpha扫描":
            st.caption("全市场扫描设置")
            
            sample_size = st.slider("扫描数量", 20, 100, 50)
            min_alpha_score = st.slider("最低Alpha分数", 0, 100, 50)
            
            scan_btn = st.button("🌐 开始扫描", type="primary", use_container_width=True)
            
        elif app_mode == "🚨 交易提示中心":
            st.caption("交易提示设置")
            
            alert_count = st.slider("显示提示数量", 5, 20, 10)
            refresh_btn = st.button("🔄 刷新提示", type="primary", use_container_width=True)
    
    # ========== 模式A: 自选股票分析 ==========
    if app_mode == "🔎 自选股票分析":
        st.title("🔎 自选股票深度分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在分析 {target_name if '股票' not in target_name else target_ticker}..."):
                # 获取数据
                df, final_ticker = data_fetcher.get_stock_data(target_ticker, start_date, end_date)
                
                if df is None or df.empty:
                    st.error(f"无法获取股票数据: {target_ticker}")
                    st.stop()
                
                # 训练模型
                df_model = train_bhmm_simple(df, n_components)
                
                if df_model is None:
                    st.error("模型训练失败")
                    st.stop()
                
                # 回测
                df_result, metrics = backtest_strategy_simple(df_model, transaction_cost)
                
                # 显示结果
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("累计收益", f"{metrics['Total Return']*100:.1f}%")
                col2.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
                col3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                col4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                
                col5, col6 = st.columns(2)
                col5.metric("胜率", f"{metrics['Win Rate']*100:.1f}%")
                col6.metric("交易次数", f"{metrics['Total Trades']}")
                
                # 图表
                tab1, tab2 = st.tabs(["📈 价格与状态", "📊 策略收益"])
                
                with tab1:
                    fig = go.Figure()
                    
                    # 价格线
                    fig.add_trace(go.Scatter(
                        x=df_result.index,
                        y=df_result['Close'],
                        line=dict(color='rgba(255,255,255,0.4)', width=1.5),
                        name="收盘价"
                    ))
                    
                    # 状态点
                    colors = ['#00E676', '#FFD600', '#FF1744', '#AA00FF']
                    for i in range(n_components):
                        mask = df_result['Regime'] == i
                        if mask.any():
                            fig.add_trace(go.Scatter(
                                x=df_result.index[mask],
                                y=df_result['Close'][mask],
                                mode='markers',
                                marker=dict(size=6, color=colors[i % 4], symbol='circle'),
                                name=f"状态 {i}"
                            ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=500,
                        title="价格走势与隐藏状态",
                        yaxis_title="价格"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig_eq = go.Figure()
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index,
                        y=df_result['Cum_Bench'],
                        line=dict(color='gray', dash='dot'),
                        name="基准"
                    ))
                    
                    fig_eq.add_trace(go.Scatter(
                        x=df_result.index,
                        y=df_result['Cum_Strat'],
                        line=dict(color='#FF5252', width=2),
                        name="策略"
                    ))
                    
                    fig_eq.update_layout(
                        template="plotly_dark",
                        height=400,
                        title="策略收益曲线",
                        yaxis_title="累计收益"
                    )
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
        
        elif run_btn:
            st.warning("请输入股票代码")
        else:
            st.info("👈 请在侧边栏输入股票代码并开始分析")
    
    # ========== 模式B: 全市场Alpha扫描 ==========
    elif app_mode == "🌐 全市场Alpha扫描":
        st.title("🌐 全市场Alpha扫描")
        st.info("💡 基于BHMM模型的多维度Alpha评分系统")
        
        if scan_btn:
            with st.spinner("正在扫描全市场Alpha..."):
                # 执行扫描
                scan_results = scanner.scan_market_alpha(
                    n_components=n_components,
                    sample_size=sample_size,
                    lookback_days=lookback_years*365
                )
                
                if scan_results.empty:
                    st.error("扫描失败，请重试")
                    st.stop()
                
                # 筛选结果
                filtered_results = scan_results[scan_results['Alpha分数'] >= min_alpha_score]
                
                st.success(f"扫描完成！发现 {len(filtered_results)} 只符合条件的股票")
                
                # 显示Top 10
                st.subheader("🏆 Alpha评分Top 10")
                
                top_10 = filtered_results.head(10)
                for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                    
                    with col1:
                        st.markdown(f"**#{idx} {row['名称']}** ({row['代码']})")
                        st.caption(f"{row['板块']}")
                    
                    with col2:
                        color = "#00E676" if row['Alpha分数'] > 60 else "#FFD600" if row['Alpha分数'] > 40 else "#FF1744"
                        st.metric("Alpha分数", f"{row['Alpha分数']:.1f}", delta_color="normal")
                    
                    with col3:
                        st.metric("信号", row['信号'])
                    
                    with col4:
                        st.metric("推荐仓位", row['推荐仓位'])
                
                # 详细数据表
                st.subheader("📋 完整扫描结果")
                
                # 简化显示，避免使用background_gradient
                display_cols = ['代码', '名称', '板块', 'Alpha分数', '动量分数', 
                              '价值分数', '质量分数', '最新状态', '信号', '推荐仓位', '最新价']
                
                display_df = filtered_results[display_cols].copy()
                
                # 使用简单格式化
                format_dict = {
                    'Alpha分数': '{:.1f}',
                    '动量分数': '{:.1f}',
                    '价值分数': '{:.1f}',
                    '质量分数': '{:.1f}',
                    '最新价': '{:.2f}'
                }
                
                for col, fmt in format_dict.items():
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: fmt.format(x))
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # 生成交易提示
                st.divider()
                alerts = alert_system.generate_alerts(filtered_results, top_n=5)
                alert_system.display_alerts(alerts)
                
                # 下载功能
                csv = filtered_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载扫描结果",
                    data=csv,
                    file_name=f"market_alpha_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("👈 配置扫描参数并开始扫描")
    
    # ========== 模式C: 交易提示中心 ==========
    elif app_mode == "🚨 交易提示中心":
        st.title("🚨 交易提示中心")
        
        # 检查是否有缓存扫描结果
        cache_key = f"scan_{datetime.now().strftime('%Y%m%d')}"
        scan_results = scanner.scan_cache.get(cache_key, pd.DataFrame())
        
        if scan_results.empty:
            st.warning("暂无扫描数据，请先运行全市场扫描")
            
            if st.button("🔄 立即运行扫描"):
                with st.spinner("正在扫描..."):
                    scan_results = scanner.scan_market_alpha(
                        n_components=n_components,
                        sample_size=50,
                        lookback_days=lookback_years*365
                    )
                    
                    if not scan_results.empty:
                        st.success("扫描完成！")
                    else:
                        st.error("扫描失败")
        else:
            st.success(f"使用今日扫描数据 ({len(scan_results)} 只股票)")
        
        # 显示交易提示
        if not scan_results.empty:
            alerts = alert_system.generate_alerts(scan_results, top_n=alert_count)
            alert_system.display_alerts(alerts)
            
            # 实时监控面板
            st.divider()
            st.subheader("📊 实时监控面板")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_alpha_count = len(scan_results[scan_results['Alpha分数'] > 70])
                st.metric("高Alpha机会", f"{high_alpha_count}只")
            
            with col2:
                buy_signals = len(scan_results[scan_results['信号'].str.contains('买入')])
                st.metric("买入信号", f"{buy_signals}只")
            
            with col3:
                risk_count = len(scan_results[scan_results['Alpha分数'] < 30])
                st.metric("风险预警", f"{risk_count}只")
            
            # 板块分布
            st.subheader("📈 板块Alpha分布")
            
            if '板块' in scan_results.columns:
                sector_stats = scan_results.groupby('板块').agg({
                    'Alpha分数': 'mean',
                    '代码': 'count'
                }).rename(columns={'代码': '数量'})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(
                        sector_stats.sort_values('Alpha分数', ascending=False).style.format({
                            'Alpha分数': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    # 简单的板块柱状图
                    fig_sector = go.Figure(data=[go.Bar(
                        x=sector_stats.index,
                        y=sector_stats['Alpha分数'],
                        marker_color='#FF5252'
                    )])
                    
                    fig_sector.update_layout(
                        template="plotly_dark",
                        height=300,
                        title="各板块平均Alpha分数",
                        yaxis_title="Alpha分数"
                    )
                    
                    st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("暂无数据，请先运行全市场扫描")

if __name__ == "__main__":
    main()
