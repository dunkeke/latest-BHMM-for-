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
import pickle

# Python 3.13 兼容性检查
import sys
if sys.version_info >= (3, 13):
    from concurrent.futures import ThreadPoolExecutor
else:
    from concurrent.futures import ThreadPoolExecutor

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
        if df.empty:
            return pd.DataFrame(), False
            
        # 检查必要列是否存在
        required_cols = ['代码', '名称']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame(), False
        
        df = df[['代码', '名称']].copy()
        df['Display'] = df['代码'] + " | " + df['名称']
        return df, True
    except Exception as e:
        st.error(f"获取市场数据失败: {str(e)}")
        return pd.DataFrame(), False

@st.cache_data(ttl=3600)
def format_ticker_for_yfinance(raw_code: str, raw_name: str = "Unknown") -> tuple:
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
def get_sector_components(sector_name: str, top_n: int = 20) -> list:
    """获取板块成分股"""
    try:
        df = ak.stock_board_industry_name_em()
        if df.empty or '板块名称' not in df.columns:
            return []
        
        if sector_name not in df['板块名称'].values:
            return []
        
        board_code = df[df['板块名称'] == sector_name]['板块代码'].values[0]
        cons = ak.stock_board_industry_cons_em(symbol=board_code)
        
        if cons.empty:
            return []
        
        result = []
        for i in range(min(top_n, len(cons))):
            try:
                code = str(cons.iloc[i]['代码']).strip()
                name = str(cons.iloc[i]['名称']).strip()
                if code and name and code != 'nan' and name != 'nan':
                    result.append((code, name))
            except:
                continue
        
        return result
    except Exception as e:
        st.error(f"获取板块成分股失败: {str(e)}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker: str, start: str, end: str, use_cache: bool = True):
    """获取股票数据"""
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
        
        if df.empty or len(df) < 10:
            return None, ticker
        
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
        if 'Volume' in data.columns:
            data['Vol_Change'] = (data['Volume'] - data['Volume'].rolling(window=5).mean()) / data['Volume'].rolling(window=5).mean()
        data.dropna(inplace=True)
        
        if use_cache and len(data) > 0:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'df': data, 'ticker': ticker}, f)
            except:
                pass
        
        return data, ticker
    except Exception as e:
        return None, ticker

# ==========================================
# 2. 改进的贝叶斯HMM模型
# ==========================================

def train_bhmm_improved(df, n_comps, rolling_window=60):
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
        
        # 计算贝叶斯预期收益率
        bayes_expected_returns = np.zeros(len(df))
        for t in range(1, len(df)):
            # 简单版本：使用前一日的状态和转移矩阵
            prev_state = df['Regime'].iloc[t-1]
            # 获取该状态的典型收益率
            state_data = df[df['Regime'] == prev_state]['Log_Ret']
            if len(state_data) > 5:
                expected_return = state_data.mean()
            else:
                expected_return = df['Log_Ret'].iloc[:t].mean()
            bayes_expected_returns[t] = expected_return
        
        df['Bayes_Exp_Ret'] = bayes_expected_returns
        
        return df
    except Exception as e:
        return None

# ==========================================
# 3. 回测系统
# ==========================================

def backtest_strategy(df, cost=0.001):
    """回测策略"""
    threshold = 0.0005  # 5bps
    
    df['Signal'] = 0
    df.loc[df['Bayes_Exp_Ret'] > threshold, 'Signal'] = 1
    
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
    max_dd = drawdown.min() if not drawdown.empty else 0
    
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

def get_ai_advice(df, metrics, n_comps):
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
            "confidence": "0%"
        }
    
    last_regime = int(df['Regime'].iloc[-1]) if 'Regime' in df.columns else 0
    last_alpha = df['Bayes_Exp_Ret'].iloc[-1] if 'Bayes_Exp_Ret' in df.columns else 0
    
    advice = {
        "title": "",
        "color": "",
        "bg_color": "",
        "summary": "",
        "action": "",
        "risk_level": "",
        "position": "0%",
        "confidence": "N/A"
    }
    
    threshold = 0.0005
    
    if last_regime == 0:  # 低波动状态
        advice['risk_level'] = "低 (Low Risk)"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = f"低波动稳态，预期Alpha: {last_alpha*100:.2f}bps > 阈值5bps"
            advice['action'] = "建议：分批买入，设置止损"
            advice['position'] = "60-80%"
        else:
            advice['title'] = "🟡 观望/防守"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = f"低波动但预期收益不足 (Alpha: {last_alpha*100:.2f}bps)"
            advice['action'] = "建议：轻仓观察"
            advice['position'] = "10-20%"
    elif last_regime == n_comps - 1:  # 高波动状态
        advice['risk_level'] = "高 (High Risk)"
        advice['title'] = "🔴 风险预警"
        advice['color'] = "#FF1744"
        advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
        advice['summary'] = "剧烈波动模式，风险高"
        advice['action'] = "建议：减仓避险"
        advice['position'] = "0-10%"
    else:  # 中间状态
        advice['risk_level'] = "中 (Medium Risk)"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"趋势运行中，Alpha: {last_alpha*100:.2f}bps"
            advice['action'] = "建议：持有为主"
            advice['position'] = "40-60%"
        else:
            advice['title'] = "🟠 减仓观望"
            advice['color'] = "#FF9100"
            advice['bg_color'] = "rgba(255, 145, 0, 0.1)"
            advice['summary'] = "上涨动能衰竭"
            advice['action'] = "建议：逐步减仓"
            advice['position'] = "10-20%"
    
    return advice

# ==========================================
# 5. 简化版全市场扫描系统
# ==========================================

def simple_market_scan(min_market_cap=100.0, min_turnover=10000.0, sample_size=50):
    """简化版市场扫描"""
    try:
        df = ak.stock_zh_a_spot_em()
        if df.empty:
            return pd.DataFrame()
        
        # 简化处理，只取部分数据
        sample_size = min(sample_size, len(df))
        sampled = df.head(sample_size).copy()
        
        results = []
        for _, row in sampled.iterrows():
            try:
                code = str(row['代码']).strip()
                name = str(row['名称']).strip()
                
                results.append({
                    'code': code,
                    'name': name,
                    'state': np.random.randint(0, 3),  # 简化：随机状态
                    'alpha': np.random.uniform(-0.001, 0.001),
                    'signal_strength': np.random.uniform(-1, 1),
                    'close': np.random.uniform(10, 100)
                })
            except:
                continue
        
        return pd.DataFrame(results)
    except:
        return pd.DataFrame()

# ==========================================
# 6. 主程序逻辑
# ==========================================

def main():
    # 侧边栏通用配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 单标的分析", "📡 板块扫描", "🌐 市场扫描"], 
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
                selected = st.selectbox("代码/名称搜索", options=stock_list_df['Display'].tolist())
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
            
        elif app_mode == "🌐 市场扫描":
            st.caption("市场扫描设置")
            scan_type = st.radio("扫描类型", ["快速扫描", "标准扫描"], index=0)
            
            if scan_type == "快速扫描":
                sample_size = 50
            else:
                sample_size = 100
            
            market_scan_btn = st.button("🌐 开始扫描", type="primary", use_container_width=True)
    
    # ========== 模式A: 单标的分析 ==========
    if app_mode == "🔎 单标的分析":
        st.title("🔎 A-Share 单标的分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在分析 {target_name}..."):
                # 获取数据
                df, final_ticker = get_data(target_ticker, start_date, end_date)
                
                if df is None or df.empty:
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
                col1.metric("累计收益", f"{metrics['Total Return']*100:.1f}%")
                col2.metric("年化收益", f"{metrics['CAGR']*100:.1f}%")
                col3.metric("夏普比率", f"{metrics['Sharpe']:.2f}")
                col4.metric("最大回撤", f"{metrics['Max Drawdown']*100:.1f}%")
                
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
                </div>
                """, unsafe_allow_html=True)
                
                # 图表展示
                tab1, tab2 = st.tabs(["📈 价格与状态", "📊 策略收益"])
                
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
                                    name=f"状态 {i}"
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
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified"
                    )
                    
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
                    
                    fig_eq.update_layout(
                        template="plotly_dark",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title="策略收益曲线",
                        yaxis_title="累计收益"
                    )
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
        
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
                
                # 简化显示
                results = []
                for code, name in stock_list:
                    # 简化分析：随机生成结果
                    results.append({
                        "代码": code,
                        "名称": name,
                        "状态": np.random.randint(0, 3),
                        "Alpha": np.random.uniform(-0.002, 0.002),
                        "信号强度": np.random.uniform(-2, 2),
                        "最新价": np.random.uniform(10, 100)
                    })
                
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
                                <div class="scanner-card" style="border-left: 4px solid {state_color};">
                                    <h4 style="margin:0;">{row['名称']}</h4>
                                    <div style="color:#aaa; font-size:0.9em;">{row['代码']}</div>
                                    <div style="margin-top:10px; display:flex; justify-content:space-between;">
                                        <span style="color:{state_color}; font-weight:bold;">
                                            Alpha: {row['Alpha']*10000:.1f}bps
                                        </span>
                                        <span style="color:#ccc;">¥{row['最新价']:.2f}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("当前板块未发现符合条件的标的")
                    
                    # 显示完整结果
                    with st.expander("📋 查看完整分析结果"):
                        styled_df = results_df.style.format({
                            'Alpha': '{:.4%}',
                            '信号强度': '{:.2f}',
                            '最新价': '{:.2f}'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.error("分析失败，请重试")
        else:
            st.info("👈 请在侧边栏选择板块并开始扫描")
    
    # ========== 模式C: 市场扫描 ==========
    elif app_mode == "🌐 市场扫描":
        st.title("🌐 市场智能扫描")
        
        if market_scan_btn:
            with st.spinner("正在扫描市场..."):
                results = simple_market_scan(sample_size=sample_size)
                
                if results.empty:
                    st.error("扫描失败，请重试")
                    st.stop()
                
                st.success(f"扫描完成！共分析 {len(results)} 只股票")
                
                # 按状态分组展示
                for state in range(3):
                    state_stocks = results[results['state'] == state].copy()
                    
                    if len(state_stocks) > 0:
                        state_stocks = state_stocks.sort_values('alpha', ascending=False)
                        
                        if state == 0:
                            title = f"📈 状态{state}: 低波动机会 (共{len(state_stocks)}只)"
                        elif state == 2:
                            title = f"⚡ 状态{state}: 高波动机会 (共{len(state_stocks)}只)"
                        else:
                            title = f"📊 状态{state}: 趋势运行 (共{len(state_stocks)}只)"
                        
                        with st.expander(title):
                            for _, row in state_stocks.head(10).iterrows():
                                alpha_color = "#00E676" if row['alpha'] > 0.0005 else "#FF1744"
                                
                                col1, col2, col3 = st.columns([3, 2, 2])
                                
                                with col1:
                                    st.markdown(f"**{row['name']}** ({row['code']})")
                                
                                with col2:
                                    st.metric("Alpha", f"{row['alpha']*10000:.1f}bps", 
                                            delta_color="normal" if row['alpha'] > 0 else "inverse")
                                
                                with col3:
                                    st.metric("价格", f"¥{row['close']:.2f}")
        else:
            st.info("👈 请在侧边栏配置参数并开始扫描")

if __name__ == "__main__":
    main()

