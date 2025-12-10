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

# 尝试导入akshare，如果失败则使用备用方案
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    st.warning("akshare不可用，将使用备用数据源")

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(
    page_title="BHMM A-Share Pro",
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 基础工具函数 (使用yfinance为主)
# ==========================================

def format_ticker_for_yfinance(raw_code: str, raw_name: str = "Unknown") -> tuple:
    """格式化股票代码为yfinance格式"""
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

def get_data_with_retry(ticker: str, start: str, end: str, max_retries: int = 3):
    """带重试机制的数据获取"""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            
            if df.empty or len(df) < 10:
                # 尝试切换后缀
                base_code = ticker.split('.')[0]
                if len(ticker.split('.')) > 1:
                    current_suffix = '.' + ticker.split('.')[1]
                    alt_suffix = '.SZ' if current_suffix == '.SS' else '.SS'
                    alt_ticker = base_code + alt_suffix
                    df = yf.download(alt_ticker, start=start, end=end, progress=False, auto_adjust=True)
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
            data = df[['Close', 'Volume']].copy()
            data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Log_Ret'].rolling(window=20).std()
            if 'Volume' in data.columns:
                data['Vol_Change'] = (data['Volume'] - data['Volume'].rolling(window=5).mean()) / data['Volume'].rolling(window=5).mean()
            data.dropna(inplace=True)
            
            return data, ticker
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                return None, ticker

@st.cache_data(ttl=3600)
def get_data(ticker: str, start: str, end: str):
    """缓存的数据获取函数"""
    return get_data_with_retry(ticker, start, end)

@st.cache_data(ttl=24*3600)
def get_a_share_list_from_cache():
    """获取A股列表（使用缓存或预设列表）"""
    # 预定义的A股龙头股列表（各行业代表）
    default_stocks = [
        ("000858", "五粮液"),
        ("000651", "格力电器"),
        ("000333", "美的集团"),
        ("000002", "万科A"),
        ("000001", "平安银行"),
        ("600519", "贵州茅台"),
        ("600036", "招商银行"),
        ("600887", "伊利股份"),
        ("600276", "恒瑞医药"),
        ("600900", "长江电力"),
        ("600309", "万华化学"),
        ("601318", "中国平安"),
        ("601857", "中国石油"),
        ("601988", "中国银行"),
        ("601398", "工商银行"),
        ("601668", "中国建筑"),
        ("002415", "海康威视"),
        ("002475", "立讯精密"),
        ("300750", "宁德时代"),
        ("300059", "东方财富"),
        ("300760", "迈瑞医疗"),
        ("300015", "爱尔眼科"),
        ("688981", "中芯国际"),
        ("688599", "天合光能"),
        ("688111", "金山办公"),
    ]
    
    df = pd.DataFrame(default_stocks, columns=['代码', '名称'])
    df['Display'] = df['代码'] + " | " + df['名称']
    return df, True

def get_sector_stocks(sector_name: str):
    """根据板块名称返回预设的股票列表"""
    sector_map = {
        "白酒": [
            ("000858", "五粮液"),
            ("600519", "贵州茅台"),
            ("002304", "洋河股份"),
            ("000568", "泸州老窖"),
            ("600809", "山西汾酒"),
        ],
        "半导体": [
            ("688981", "中芯国际"),
            ("002049", "紫光国微"),
            ("603501", "韦尔股份"),
            ("300661", "圣邦股份"),
            ("002371", "北方华创"),
        ],
        "新能源": [
            ("300750", "宁德时代"),
            ("002594", "比亚迪"),
            ("002812", "恩捷股份"),
            ("002460", "赣锋锂业"),
            ("300014", "亿纬锂能"),
        ],
        "医药": [
            ("600276", "恒瑞医药"),
            ("300760", "迈瑞医疗"),
            ("300015", "爱尔眼科"),
            ("000538", "云南白药"),
            ("600085", "同仁堂"),
        ],
        "金融": [
            ("601318", "中国平安"),
            ("600036", "招商银行"),
            ("601398", "工商银行"),
            ("601166", "兴业银行"),
            ("600030", "中信证券"),
        ],
        "消费": [
            ("600887", "伊利股份"),
            ("000651", "格力电器"),
            ("000333", "美的集团"),
            ("603288", "海天味业"),
            ("002557", "洽洽食品"),
        ],
        "科技": [
            ("002415", "海康威视"),
            ("002475", "立讯精密"),
            ("300059", "东方财富"),
            ("300033", "同花顺"),
            ("002230", "科大讯飞"),
        ],
    }
    
    return sector_map.get(sector_name, [])

# ==========================================
# 2. HMM模型训练
# ==========================================

def train_bhmm(df, n_comps):
    """训练贝叶斯HMM模型"""
    scale = 100.0
    X = df[['Log_Ret', 'Volatility']].values * scale
    
    try:
        model = GaussianHMM(
            n_components=n_comps, 
            covariance_type="full", 
            n_iter=1000, 
            random_state=88, 
            tol=0.01, 
            min_covar=0.001
        )
        model.fit(X)
        
        hidden_states = model.predict(X)
        
        # 状态排序
        state_vol_means = [(i, X[hidden_states == i, 1].mean()) for i in range(n_comps) 
                          if np.sum(hidden_states == i) > 0]
        sorted_stats = sorted(state_vol_means, key=lambda x: x[1])
        mapping = {old: new for new, (old, _) in enumerate(sorted_stats)}
        
        df['Regime'] = np.array([mapping.get(s, s) for s in hidden_states])
        
        # 计算贝叶斯预期收益率
        bayes_expected_returns = np.zeros(len(df))
        for t in range(1, len(df)):
            prev_state = df['Regime'].iloc[t-1]
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
            "position": "0%"
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
        "position": "0%"
    }
    
    threshold = 0.0005
    
    if last_regime == 0:  # 低波动状态
        advice['risk_level'] = "低风险"
        if last_alpha > threshold:
            advice['title'] = "🟢 积极建仓机会"
            advice['color'] = "#00E676"
            advice['bg_color'] = "rgba(0, 230, 118, 0.1)"
            advice['summary'] = f"低波动稳态，预期Alpha: {last_alpha*10000:.1f}bps > 5bps"
            advice['action'] = "建议：分批买入，设置止损"
            advice['position'] = "60-80%"
        else:
            advice['title'] = "🟡 观望/防守"
            advice['color'] = "#FFD600"
            advice['bg_color'] = "rgba(255, 214, 0, 0.1)"
            advice['summary'] = f"低波动但预期收益不足 (Alpha: {last_alpha*10000:.1f}bps)"
            advice['action'] = "建议：轻仓观察"
            advice['position'] = "10-20%"
    elif last_regime == n_comps - 1:  # 高波动状态
        advice['risk_level'] = "高风险"
        advice['title'] = "🔴 风险预警"
        advice['color'] = "#FF1744"
        advice['bg_color'] = "rgba(255, 23, 68, 0.1)"
        advice['summary'] = "剧烈波动模式，风险高"
        advice['action'] = "建议：减仓避险"
        advice['position'] = "0-10%"
    else:  # 中间状态
        advice['risk_level'] = "中风险"
        if last_alpha > threshold:
            advice['title'] = "🔵 趋势延续"
            advice['color'] = "#2962FF"
            advice['bg_color'] = "rgba(41, 98, 255, 0.1)"
            advice['summary'] = f"趋势运行中，Alpha: {last_alpha*10000:.1f}bps"
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
# 5. 主程序逻辑
# ==========================================

def main():
    # 侧边栏通用配置
    with st.sidebar:
        st.title("🇨🇳 BHMM A-Share Pro")
        app_mode = st.radio(
            "功能模式", 
            ["🔎 单股票分析", "📡 板块扫描"], 
            index=0
        )
        st.divider()
        
        # 通用参数
        n_components = st.slider("隐藏状态数", 2, 4, 3)
        lookback_years = st.slider("回看年限", 1, 5, 2)
        trans_cost_bps = st.number_input("交易成本(bps)", value=10, min_value=0, max_value=50)
        transaction_cost = trans_cost_bps / 10000
        
        start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        st.divider()
        
        # 模式特定配置
        if app_mode == "🔎 单股票分析":
            st.caption("单股票分析")
            
            # 使用缓存的股票列表
            stock_list_df, is_online = get_a_share_list_from_cache()
            
            if not stock_list_df.empty:
                selected = st.selectbox("选择股票", options=stock_list_df['Display'].tolist())
                if selected:
                    parts = selected.split(" | ")
                    if len(parts) >= 2:
                        c = parts[0]
                        n = parts[1]
                        target_ticker, target_name = format_ticker_for_yfinance(c, n)
                    else:
                        target_ticker, target_name = None, None
                else:
                    target_ticker, target_name = None, None
            else:
                mc = st.text_input("股票代码", value="000858.SZ")
                target_ticker, target_name = format_ticker_for_yfinance(mc, mc)
            
            run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
            
        elif app_mode == "📡 板块扫描":
            st.caption("板块扫描设置")
            SECTORS = ["白酒", "半导体", "新能源", "医药", "金融", "消费", "科技"]
            target_sector = st.selectbox("选择板块", SECTORS)
            
            scan_btn = st.button("📡 开始扫描", type="primary", use_container_width=True)
    
    # ========== 模式A: 单股票分析 ==========
    if app_mode == "🔎 单股票分析":
        st.title("🔎 A-Share 单股票分析")
        
        if run_btn and target_ticker:
            with st.spinner(f"正在分析 {target_name}..."):
                # 获取数据
                df, final_ticker = get_data(target_ticker, start_date, end_date)
                
                if df is None or df.empty:
                    st.error("无法获取股票数据，请检查代码是否正确")
                    st.stop()
                
                # 训练HMM模型
                df = train_bhmm(df, n_components)
                
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
                        风险等级: {ai_advice['risk_level']}
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
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified"
                    )
                    
                    fig.update_yaxes(title_text="价格", row=1, col=1)
                    if 'Volume' in df.columns:
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
            st.info("👈 请在侧边栏选择股票并开始分析")
    
    # ========== 模式B: 板块扫描 ==========
    elif app_mode == "📡 板块扫描":
        st.title(f"📡 板块扫描: {target_sector}")
        
        if scan_btn:
            # 获取板块股票列表
            sector_stocks = get_sector_stocks(target_sector)
            
            if not sector_stocks:
                st.error(f"未找到{target_sector}板块的股票数据")
                st.stop()
            
            st.success(f"获取到 {len(sector_stocks)} 只{target_sector}板块股票")
            
            # 进度条
            progress_bar = st.progress(0)
            results = []
            
            for idx, (code, name) in enumerate(sector_stocks):
                with st.spinner(f"正在分析 {name}({code})..."):
                    ticker, _ = format_ticker_for_yfinance(code, name)
                    df, _ = get_data(ticker, start_date, end_date)
                    
                    if df is not None and not df.empty and len(df) > 100:
                        df_model = train_bhmm(df, n_components)
                        
                        if df_model is not None:
                            last_regime = int(df_model['Regime'].iloc[-1]) if 'Regime' in df_model.columns else 0
                            last_alpha = df_model['Bayes_Exp_Ret'].iloc[-1] if 'Bayes_Exp_Ret' in df_model.columns else 0
                            
                            # 计算信号强度
                            if 'Volatility' in df.columns and df['Volatility'].iloc[-1] > 0:
                                signal_strength = last_alpha / df['Volatility'].iloc[-1]
                            else:
                                signal_strength = 0
                            
                            results.append({
                                "代码": code,
                                "名称": name,
                                "状态": last_regime,
                                "Alpha": last_alpha,
                                "信号强度": signal_strength,
                                "最新价": df['Close'].iloc[-1] if 'Close' in df.columns else 0
                            })
                
                progress_bar.progress((idx + 1) / len(sector_stocks))
            
            progress_bar.empty()
            
            if results:
                results_df = pd.DataFrame(results)
                
                # 筛选推荐标的（状态0且Alpha>0）
                recommendation_df = results_df[
                    (results_df['状态'] == 0) & 
                    (results_df['Alpha'] > 0.0005)
                ].sort_values('Alpha', ascending=False)
                
                if not recommendation_df.empty:
                    st.success(f"🎯 发现 {len(recommendation_df)} 只潜在建仓标的")
                    
                    # 显示推荐标的
                    cols = st.columns(3)
                    for idx, row in recommendation_df.iterrows():
                        with cols[idx % 3]:
                            state_color = ['#00E676', '#FFD600', '#FF1744', '#AA00FF'][int(row['状态']) % 4]
                            alpha_color = "#00E676" if row['Alpha'] > 0.0005 else "#FF1744"
                            
                            st.markdown(f"""
                            <div class="scanner-card state-{int(row['状态'])}">
                                <h4 style="margin:0;">{row['名称']}</h4>
                                <div style="color:#aaa; font-size:0.9em;">{row['代码']}</div>
                                <div style="margin-top:10px; display:flex; justify-content:space-between;">
                                    <span style="color:{alpha_color}; font-weight:bold;">
                                        Alpha: {row['Alpha']*10000:.1f}bps
                                    </span>
                                    <span style="color:#ccc;">¥{row['最新价']:.2f}</span>
                                </div>
                                <div style="font-size:0.8em; color:#888; margin-top:5px;">
                                    信号强度: {row['信号强度']:.2f}
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
                    }).background_gradient(
                        subset=['Alpha', '信号强度'], 
                        cmap='RdYlGn'
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # 下载结果
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载扫描结果",
                        data=csv,
                        file_name=f"{target_sector}_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("分析失败，请重试")
        else:
            st.info("👈 请在侧边栏选择板块并开始扫描")

if __name__ == "__main__":
    main()

