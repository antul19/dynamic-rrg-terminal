import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sector & Equity Rotation Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Disc.",
    "XLC": "Communication",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

QUADRANT_META = {
    "LEADING":   {"color": "#00E676", "bg": "rgba(0, 230, 118, 0.15)"},
    "WEAKENING": {"color": "#FFD600", "bg": "rgba(255, 214, 0, 0.15)"},
    "LAGGING":   {"color": "#FF5252", "bg": "rgba(255, 82, 82, 0.15)"},
    "IMPROVING": {"color": "#40C4FF", "bg": "rgba(64, 196, 255, 0.15)"},
}


# -----------------------------------------------------------------------------
# 2. DATA FETCHING, RRG MATH & VECTOR KINEMATICS ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_data(tickers: tuple, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Batch-downloads Adjusted Close prices for all tickers in one request."""
    raw = yf.download(
        list(tickers),
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.dropna(how="all", axis=1).ffill()


def get_quadrant(rs_ratio: float, rs_mom: float) -> str:
    """Classifies (RS-Ratio, RS-Momentum) coordinates into RRG quadrants."""
    if rs_ratio >= 100 and rs_mom >= 100:
        return "LEADING"
    elif rs_ratio >= 100 and rs_mom < 100:
        return "WEAKENING"
    elif rs_ratio < 100 and rs_mom < 100:
        return "LAGGING"
    else:
        return "IMPROVING"


def get_heading_direction(angle_deg: float) -> str:
    """Maps Cartesian polar angle (0°=E, 90°=N) to RRG rotation phase."""
    if 0 <= angle_deg <= 90:
        return "↗ NE (Power)"
    elif 90 < angle_deg <= 180:
        return "↖ NW (Recovery)"
    elif 180 < angle_deg <= 270:
        return "↙ SW (Breakdown)"
    else:
        return "↘ SE (Rollover)"


def compute_rrg_metrics(close_df: pd.DataFrame, benchmark: str, window: int = 14):
    """Calculates JdK-style RS-Ratio and RS-Momentum centered around 100."""
    if benchmark not in close_df.columns:
        return None, None

    bench_series = close_df[benchmark]
    assets_df = close_df.drop(columns=[benchmark], errors="ignore")

    # 1. Raw Relative Strength smoothed
    rs = (assets_df.div(bench_series, axis=0)) * 100
    rs_smooth = rs.ewm(span=max(2, window // 3), adjust=False).mean()

    # 2. RS-Ratio (Rolling Z-score scaled around 100)
    rs_mean = rs_smooth.rolling(window=window).mean()
    rs_std = rs_smooth.rolling(window=window).std().replace(0, np.nan)
    rs_ratio = 100 + ((rs_smooth - rs_mean) / rs_std) * 2.0
    rs_ratio = rs_ratio.ewm(span=3, adjust=False).mean()

    # 3. RS-Momentum (Rolling Z-score of RS-Ratio ROC scaled around 100)
    rs_roc = rs_ratio.pct_change(periods=1) * 100
    mom_mean = rs_roc.rolling(window=window).mean()
    mom_std = rs_roc.rolling(window=window).std().replace(0, np.nan)
    rs_mom = 100 + ((rs_roc - mom_mean) / mom_std) * 2.0
    rs_mom = rs_mom.ewm(span=3, adjust=False).mean()

    return rs_ratio.dropna(how="all"), rs_mom.dropna(how="all")


def compute_tail_kinematics(
    close_df: pd.DataFrame,
    rs_ratio: pd.DataFrame,
    rs_mom: pd.DataFrame,
    tail_len: int,
    custom_tickers: list
) -> pd.DataFrame:
    """
    Computes position, velocity, acceleration, mean tail speed, and heading angle
    for every active ticker on the RRG.
    """
    rows = []
    for sym in rs_ratio.columns:
        if sym not in close_df.columns:
            continue

        r_ser = rs_ratio[sym].dropna().tail(max(tail_len, 3))
        m_ser = rs_mom[sym].dropna().tail(max(tail_len, 3))
        s_close = close_df[sym].dropna()

        if len(r_ser) < 3 or len(m_ser) < 3 or len(s_close) < 2:
            continue

        # Current & previous step displacements
        dx_curr = r_ser.iloc[-1] - r_ser.iloc[-2]
        dy_curr = m_ser.iloc[-1] - m_ser.iloc[-2]
        dx_prev = r_ser.iloc[-2] - r_ser.iloc[-3]
        dy_prev = m_ser.iloc[-2] - m_ser.iloc[-3]

        # Instantaneous 1-bar velocity & prior 1-bar velocity
        vel_curr = float(np.hypot(dx_curr, dy_curr))
        vel_prev = float(np.hypot(dx_prev, dy_prev))
        accel = vel_curr - vel_prev

        # Mean velocity across the selected tail window
        r_tail = r_ser.tail(tail_len)
        m_tail = m_ser.tail(tail_len)
        step_distances = np.hypot(r_tail.diff().dropna(), m_tail.diff().dropna())
        avg_tail_vel = float(step_distances.mean()) if not step_distances.empty else vel_curr

        # Radial distance from benchmark center (100, 100)
        dist_from_center = float(np.hypot(r_ser.iloc[-1] - 100.0, m_ser.iloc[-1] - 100.0))

        # Heading angle in degrees [0, 360) where 0°=East (+RS-Ratio), 90°=North (+RS-Mom)
        heading_deg = float((np.degrees(np.arctan2(dy_curr, dx_curr)) + 360.0) % 360.0)
        direction_tag = get_heading_direction(heading_deg)

        pct_chg = ((s_close.iloc[-1] - s_close.iloc[-2]) / s_close.iloc[-2]) * 100.0
        r_val = float(r_ser.iloc[-1])
        m_val = float(m_ser.iloc[-1])

        rows.append({
            "Ticker": sym,
            "Group": "Custom Target" if sym in custom_tickers else f"Sector ({SECTOR_ETFS.get(sym, 'ETF')})",
            "Last Price": float(s_close.iloc[-1]),
            "Bar Chg %": float(pct_chg),
            "Quadrant": get_quadrant(r_val, m_val),
            "Direction": direction_tag,
            "Heading (°)": round(heading_deg, 1),
            "Velocity (1B)": round(vel_curr, 2),
            "Acceleration": round(accel, 2),
            "Avg Tail Speed": round(avg_tail_vel, 2),
            "Center Dist": round(dist_from_center, 2),
            "RS-Ratio": round(r_val, 2),
            "RS-Momentum": round(m_val, 2),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. TOP SCROLLING HTML/CSS TICKER TAPE (CONSTANT PACE)
# -----------------------------------------------------------------------------
def render_scrolling_ticker(kinematics_df: pd.DataFrame, seconds_per_ticker: int = 8):
    """Renders a continuous, hover-pausable HTML/CSS scrolling ticker bar with constant speed."""
    if kinematics_df.empty:
        return

    items_html = []
    for _, row in kinematics_df.iterrows():
        sym = row["Ticker"]
        last_price = row["Last Price"]
        pct_chg = row["Bar Chg %"]
        quad = row["Quadrant"]
        vel = row["Velocity (1B)"]
        dir_arrow = row["Direction"].split(" ")[0]

        chg_color = "#00E676" if pct_chg >= 0 else "#FF5252"
        chg_sign = "+" if pct_chg >= 0 else ""
        q_color = QUADRANT_META[quad]["color"]
        q_bg = QUADRANT_META[quad]["bg"]

        item = (
            f'<div class="ticker-item">'
            f'<span class="t-sym">{sym}</span>'
            f'<span class="t-price">${last_price:,.2f}</span>'
            f'<span class="t-chg" style="color:{chg_color};">{chg_sign}{pct_chg:.2f}%</span>'
            f'<span class="t-badge" style="color:{q_color}; background:{q_bg}; border:1px solid {q_color};">'
            f'{quad} {dir_arrow} v:{vel:.2f}</span>'
            f'</div>'
        )
        items_html.append(item)

    num_tickers = max(len(items_html), 1)
    repeat_pairs = max(2, 16 // num_tickers)
    # Ensure even number of repeats so the -50% keyframe loops seamlessly
    if repeat_pairs % 2 != 0:
        repeat_pairs += 1

    total_items_in_half = num_tickers * (repeat_pairs // 2)
    scroll_duration = max(45, total_items_in_half * seconds_per_ticker)
    track_content = "".join(items_html) * repeat_pairs

    ticker_css_html = f"""
    <style>
    .ticker-wrap {{
        width: 100%;
        overflow: hidden;
        background: #0E1117;
        border-top: 1px solid #262730;
        border-bottom: 1px solid #262730;
        padding: 8px 0;
        margin-bottom: 14px;
        white-space: nowrap;
        box-sizing: border-box;
    }}
    .ticker-track {{
        display: inline-flex;
        width: max-content;
        animation: ticker-scroll {scroll_duration}s linear infinite;
    }}
    .ticker-wrap:hover .ticker-track {{
        animation-play-state: paused;
    }}
    .ticker-item {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0 22px;
        border-right: 1px solid #262730;
        font-family: 'Source Code Pro', monospace;
        font-size: 0.88rem;
    }}
    .t-sym {{ font-weight: 700; color: #FAFAFA; }}
    .t-price {{ color: #B0B8C4; }}
    .t-chg {{ font-weight: 600; }}
    .t-badge {{
        font-size: 0.70rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        letter-spacing: 0.4px;
    }}
    @keyframes ticker-scroll {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    </style>
    <div class="ticker-wrap" title="Hover to pause ticker">
        <div class="ticker-track">
            {track_content}
        </div>
    </div>
    """
    st.markdown(ticker_css_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. INTERACTIVE PLOTLY RRG CHART BUILDER
# -----------------------------------------------------------------------------
def build_rrg_figure(
    rs_ratio: pd.DataFrame,
    rs_mom: pd.DataFrame,
    kinematics_df: pd.DataFrame,
    tail_len: int,
    custom_tickers: list,
    mode: str,
    benchmark: str
) -> go.Figure:
    fig = go.Figure()
    kin_map = kinematics_df.set_index("Ticker").to_dict(orient="index")

    recent_r = rs_ratio.tail(tail_len)
    recent_m = rs_mom.tail(tail_len)
    max_dev_x = max(abs(recent_r.max().max() - 100), abs(100 - recent_r.min().min()), 2.5) * 1.18
    max_dev_y = max(abs(recent_m.max().max() - 100), abs(100 - recent_m.min().min()), 2.5) * 1.18

    x_min, x_max = 100 - max_dev_x, 100 + max_dev_x
    y_min, y_max = 100 - max_dev_y, 100 + max_dev_y

    quad_shapes = [
        dict(type="rect", x0=100, x1=x_max, y0=100, y1=y_max, fillcolor="rgba(0, 230, 118, 0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=100, x1=x_max, y0=y_min, y1=100, fillcolor="rgba(255, 214, 0, 0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=x_min, x1=100, y0=y_min, y1=100, fillcolor="rgba(255, 82, 82, 0.06)", line_width=0, layer="below"),
        dict(type="rect", x0=x_min, x1=100, y0=100, y1=y_max, fillcolor="rgba(64, 196, 255, 0.06)", line_width=0, layer="below"),
        dict(type="line", x0=100, x1=100, y0=y_min, y1=y_max, line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dash")),
        dict(type="line", x0=x_min, x1=x_max, y0=100, y1=100, line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dash")),
    ]
    fig.update_layout(shapes=quad_shapes)

    annotations = [
        dict(x=x_max - 0.03 * max_dev_x, y=y_max - 0.05 * max_dev_y, text="LEADING", showarrow=False, xanchor="right", font=dict(color="rgba(0,230,118,0.45)", size=15, weight="bold")),
        dict(x=x_max - 0.03 * max_dev_x, y=y_min + 0.05 * max_dev_y, text="WEAKENING", showarrow=False, xanchor="right", font=dict(color="rgba(255,214,0,0.45)", size=15, weight="bold")),
        dict(x=x_min + 0.03 * max_dev_x, y=y_min + 0.05 * max_dev_y, text="LAGGING", showarrow=False, xanchor="left", font=dict(color="rgba(255,82,82,0.45)", size=15, weight="bold")),
        dict(x=x_min + 0.03 * max_dev_x, y=y_max - 0.05 * max_dev_y, text="IMPROVING", showarrow=False, xanchor="left", font=dict(color="rgba(64,196,255,0.45)", size=15, weight="bold")),
    ]
    fig.update_layout(annotations=annotations)

    for sym in rs_ratio.columns:
        r_tail = rs_ratio[sym].dropna().tail(tail_len)
        m_tail = rs_mom[sym].dropna().tail(tail_len)
        if r_tail.empty or m_tail.empty or sym not in kin_map:
            continue

        is_custom = sym in custom_tickers
        k_info = kin_map[sym]
        quad = k_info["Quadrant"]
        q_color = QUADRANT_META[quad]["color"]

        if mode == "Sectors + Custom" and not is_custom:
            line_width = 1.6
            opacity = 0.42
            head_size = 8
            label_text = sym
        else:
            line_width = 3.5 if is_custom else 2.5
            opacity = 1.0
            head_size = 13 if is_custom else 10
            label_text = f"<b>{sym}</b>" if is_custom else sym

        # Trail spline
        fig.add_trace(
            go.Scatter(
                x=r_tail.values,
                y=m_tail.values,
                mode="lines+markers",
                name=sym,
                opacity=opacity,
                line=dict(width=line_width, color=q_color, shape="spline", smoothing=0.7),
                marker=dict(size=5, color=q_color),
                customdata=np.stack([r_tail.index.strftime("%Y-%m-%d")], axis=-1),
                hovertemplate=(
                    f"<b>{sym}</b> (%{{customdata[0]}})<br>"
                    "RS-Ratio: %{x:.2f}<br>"
                    "RS-Mom: %{y:.2f}<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Head marker with full kinematics hover tooltip
        fig.add_trace(
            go.Scatter(
                x=[r_tail.iloc[-1]],
                y=[m_tail.iloc[-1]],
                mode="markers+text",
                text=[label_text],
                textposition="top right",
                textfont=dict(size=12 if is_custom else 10, color="#FFFFFF" if is_custom else "#B0B8C4"),
                opacity=opacity,
                marker=dict(
                    size=head_size,
                    color=q_color,
                    line=dict(width=2, color="#FFFFFF" if is_custom else "#1E1E1E"),
                    symbol="diamond" if is_custom else "circle",
                ),
                hovertemplate=(
                    f"<b>{sym} — Current Head</b><br>"
                    f"Quadrant: {quad}<br>"
                    f"Heading: {k_info['Heading (°)']}° ({k_info['Direction']})<br>"
                    f"Velocity (1B): {k_info['Velocity (1B)']:.2f}<br>"
                    f"Acceleration: {k_info['Acceleration']:+.2f}<br>"
                    "RS-Ratio: %{x:.2f} | RS-Mom: %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=660,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text=f"Relative Rotation Graph (vs. <b>{benchmark}</b>) — Diamonds = Custom Tickers | Circles = Sector ETFs",
            font=dict(size=15)
        ),
        xaxis=dict(title="JdK RS-Ratio (Relative Trend)", range=[x_min, x_max], zeroline=False, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="JdK RS-Momentum (Rate of Change)", range=[y_min, y_max], zeroline=False, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------------------------------------------------------------
# 5. MAIN UI & CONTROL BAR
# -----------------------------------------------------------------------------
st.title("⚡ Dynamic Sector & Equity RRG Terminal")

c1, c2, c3, c4, c5 = st.columns([2.6, 1.5, 1.0, 1.1, 1.2])

with c1:
    custom_input = st.text_input(
        "Custom Ticker Line (comma-separated)",
        value="RKLB, GEV, PLTR",
        help="Type any stock or ETF symbols to overlay directly onto the RRG."
    )
with c2:
    view_mode = st.selectbox(
        "Display Layer",
        options=["Sectors + Custom", "Custom Only", "Sectors Only"],
        index=0
    )
with c3:
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
with c4:
    timeframe = st.selectbox("Timeframe", options=["Weekly (1wk)", "Daily (1d)"], index=0)
with c5:
    tail_length = st.slider("Tail Length (Bars)", min_value=3, max_value=16, value=6)

custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
sector_tickers = list(SECTOR_ETFS.keys())

if view_mode == "Sectors + Custom":
    active_tickers = list(dict.fromkeys(sector_tickers + custom_tickers))
elif view_mode == "Custom Only":
    active_tickers = custom_tickers
else:
    active_tickers = sector_tickers

active_tickers = [t for t in active_tickers if t != benchmark]
all_download_tickers = tuple(sorted(set(active_tickers + [benchmark])))

interval_code = "1wk" if "1wk" in timeframe else "1d"
lookback_window = 14 if interval_code == "1wk" else 21

if not active_tickers:
    st.warning("Please enter at least one valid ticker symbol.")
    st.stop()

with st.spinner("Fetching market data & computing Relative Rotation vectors..."):
    close_df = fetch_price_data(all_download_tickers, period="2y", interval=interval_code)

if close_df.empty or benchmark not in close_df.columns:
    st.error(f"Could not load benchmark `{benchmark}` from Yahoo Finance. Check the ticker symbol.")
    st.stop()

rs_ratio, rs_mom = compute_rrg_metrics(close_df, benchmark=benchmark, window=lookback_window)
kinematics_df = compute_tail_kinematics(close_df, rs_ratio, rs_mom, tail_length, custom_tickers)

# Render Component 1: Top Scrolling Status Ticker Tape (Constant Slow Pace)
render_scrolling_ticker(kinematics_df, seconds_per_ticker=8)

# Render Component 2: Interactive Plotly RRG Chart
rrg_fig = build_rrg_figure(
    rs_ratio=rs_ratio,
    rs_mom=rs_mom,
    kinematics_df=kinematics_df,
    tail_len=tail_length,
    custom_tickers=custom_tickers,
    mode=view_mode,
    benchmark=benchmark
)
st.plotly_chart(rrg_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. KINEMATICS SCANNER & QUADRANT TABLE
# -----------------------------------------------------------------------------
st.subheader("🚀 Tail Kinematics Scanner & Quadrant Breakdown")

f1, f2, f3 = st.columns([1.8, 1.5, 1.5])
with f1:
    direction_filter = st.multiselect(
        "Filter by Heading Phase",
        options=["↗ NE (Power)", "↖ NW (Recovery)", "↘ SE (Rollover)", "↙ SW (Breakdown)"],
        default=[],
        placeholder="All Headings (Select ↗ NE for Power Breakouts)"
    )
with f2:
    only_accelerating = st.checkbox("⚡ Positive Acceleration Only (Accel > 0)", value=False)
with f3:
    sort_col = st.selectbox(
        "Rank Table By",
        options=["Velocity (1B)", "Acceleration", "Avg Tail Speed", "RS-Ratio", "RS-Momentum", "Center Dist"],
        index=0
    )

filtered_df = kinematics_df.copy()
if direction_filter:
    filtered_df = filtered_df[filtered_df["Direction"].isin(direction_filter)]
if only_accelerating:
    filtered_df = filtered_df[filtered_df["Acceleration"] > 0]

filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)

max_vel = float(kinematics_df["Velocity (1B)"].max()) if not kinematics_df.empty else 5.0

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Last Price": st.column_config.NumberColumn("Last Price", format="$%.2f"),
        "Bar Chg %": st.column_config.NumberColumn("Bar Chg %", format="%+.2f%%"),
        "Heading (°)": st.column_config.NumberColumn(
            "Heading (°)",
            help="0°–90° (NE) = Gaining both RS-Ratio & RS-Momentum",
            format="%.1f°"
        ),
        "Velocity (1B)": st.column_config.ProgressColumn(
            "Velocity (1B)",
            help="1-Bar vector displacement magnitude in RRG space",
            format="%.2f",
            min_value=0.0,
            max_value=max(max_vel, 0.1),
        ),
        "Acceleration": st.column_config.NumberColumn(
            "Acceleration",
            help="Change in tail velocity vs. previous bar (+ = speeding up)",
            format="%+.2f"
        ),
        "Avg Tail Speed": st.column_config.NumberColumn("Avg Tail Speed", format="%.2f"),
        "Center Dist": st.column_config.NumberColumn(
            "Center Dist",
            help="Radial distance from benchmark origin (100, 100)",
            format="%.2f"
        ),
    }
)
