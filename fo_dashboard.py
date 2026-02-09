import streamlit as st
import pandas as pd
from datetime import datetime, time
import glob
import os
from streamlit_autorefresh import st_autorefresh
import matplotlib.pyplot as plt


MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

DATA_DIR = "/Users/mayuri/Desktop/Stocks_code/option/FO"
EOD_file = "/Users/mayuri/Desktop/Stocks_code/option/FO/alerts/all_alerts_master.csv"
date_str = datetime.now().strftime("%Y%m%d")

# ================== INDEX UNIVERSE ==================
INDEX_SYMBOLS = ("NIFTY", "BANKNIFTY")


# ================== CONFIG ==================
st.set_page_config(
    page_title="F&O Smart Money Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    @keyframes blink {
        0% { background-color: #fff3cd; }
        50% { background-color: #ffe082; }
        100% { background-color: #fff3cd; }
    }
    .blink-row {
        animation: blink 1.2s ease-in-out infinite;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ================== AUTO REFRESH ==================
st_autorefresh(
    interval=3 * 60 * 1000,  # 5 minutes
    key="auto_refresh"
)

# ================== LOAD DATA (NO CACHE) ==================
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ================== SIDEBAR ==================
st.sidebar.title("📅 Scan Date")

# csv_files = sorted(
#     glob.glob(f"{DATA_DIR}/fo_futures_full_scan_{date_str}.csv")
# )

csv_files = sorted(
    glob.glob(f"{DATA_DIR}/fo_futures_full_scan_*.csv")
)

if not csv_files:
    st.error("No scan CSV files found")
    st.stop()


if not csv_files:
    st.error("No scan CSV found for today")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Scan File",
    csv_files,
    index=len(csv_files) - 1,
    format_func=lambda x: os.path.basename(x)
)


df = load_data(selected_file)

# ================== DERIVE TRADE DATE (MUST BE EARLY) ==================
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

df["trade_date"] = df["timestamp"].dt.date

# ================== DATE SELECTION ==================
st.sidebar.markdown("---")
st.sidebar.title("📅 Trade Date Selection")

available_dates = sorted(df["trade_date"].unique())
default_date = max(available_dates)

selected_trade_date = st.sidebar.date_input(
    "Select Trade Date",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates)
)
###================== TIME SELECTION ==================

st.sidebar.markdown("---")
st.sidebar.title("⏰ Trade Time Selection (5-min)")

# ---------------- Market hours ----------------
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30

# ---------------- Hour options ----------------
hours = list(range(MARKET_OPEN_HOUR, MARKET_CLOSE_HOUR + 1))

# ---------------- Minute options (15-min TF) ----------------
#minute_options = [0, 15, 30, 45]
minute_options = list(range(0, 60, 5))

# ---------------- Sensible defaults ----------------
# now = datetime.now()

# default_hour = now.hour
# default_min = (now.minute // 15) * 15

# ---------------- Sensible defaults ----------------
now = datetime.now()

default_hour = now.hour
default_min = (now.minute // 5) * 5


# Clamp defaults to market hours
if (default_hour < MARKET_OPEN_HOUR) or (
    default_hour == MARKET_OPEN_HOUR and default_min < MARKET_OPEN_MIN
):
    default_hour, default_min = MARKET_OPEN_HOUR, MARKET_OPEN_MIN

if (default_hour > MARKET_CLOSE_HOUR) or (
    default_hour == MARKET_CLOSE_HOUR and default_min > MARKET_CLOSE_MIN
):
    default_hour, default_min = MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN

# ---------------- UI ----------------
col1, col2 = st.sidebar.columns(2)

with col1:
    selected_hour = st.selectbox(
        "HH",
        hours,
        index=hours.index(default_hour),
        key="hh_select"
    )

with col2:
    selected_minute = st.selectbox(
        "MM",
        minute_options,
        index=minute_options.index(default_min),
        key="mm_select"
    )

# ---------------- Enforce market open ----------------
if (
    selected_hour == MARKET_OPEN_HOUR
    and selected_minute < MARKET_OPEN_MIN
):
    selected_minute = MARKET_OPEN_MIN
    st.sidebar.info("⏰ Market opens at 09:15")

# ---------------- Enforce market close ----------------
if (
    selected_hour == MARKET_CLOSE_HOUR
    and selected_minute > MARKET_CLOSE_MIN
):
    selected_minute = MARKET_CLOSE_MIN
    st.sidebar.info("⏰ Market closes at 15:30")

# ---------------- Final time ----------------
selected_trade_time = time(selected_hour, selected_minute)

selected_datetime = datetime.combine(
    selected_trade_date,
    selected_trade_time
)


# ================== ALLOWED TRIGGER ALERTS ==================
ALLOWED_ALERTS = [
    "FRESH_BREAKOUT",
    "FRESH_BREAKDOWN",
    "BREAKOUT+ATH",
    "BREAKOUT_MONEY_SURGE",
    "BREAKDOWN_MONEY_SURGE",
    "BREAKOUT_MONEY_SPIKE",
    "BREAKDOWN_MONEY_SPIKE",
    "BREAKOUT+UMM",
    "INSTITUTIONAL_ACCUMULATION",
    "IMA_SURGE",
    "ATH",
    "ATL",
    "BREAKDOWN+ATL"
]

# ================== NORMALIZE TRIGGER ALERT ==================
df["trigger_alert"] = (
    df["trigger_alert"]
    .astype(str)
    .str.strip()
    .str.upper()
)

def breakout_breakdown_style(val):
    if pd.isna(val):
        return ""

    v = str(val).upper()

    if "BREAKOUT" in v:
        return "background-color:#e8f5e9; color:#1b5e20; font-weight:700" #green 
    elif "BREAKDOWN" in v:
        return "background-color:#fdecea; color:#b71c1c; font-weight:700" #red
    elif "ATL" in v:
        return "background-color:#fdecea; color:#b71c1c; font-weight:700" #red
    elif "ATH" in v:
        return "background-color:#e8f5e9; color:#1b5e20; font-weight:700" #green
    return ""


# ================== LOAD EOD SMART MONEY DATA ==================
@st.cache_data(ttl=300)
def load_eod_data(EOD_file):
    eod = pd.read_csv(EOD_file)
    eod["date"] = pd.to_datetime(eod["date"])
    return eod

if os.path.exists(EOD_file):
    df_eod = load_eod_data(EOD_file)
else:
    df_eod = pd.DataFrame()

def build_smart_money_alerts(row):
    alerts = []

    if row.get("is_ath", 0) == 1:
        alerts.append("S_ATH")

    if row.get("rolling_high", 0) == 1:
        alerts.append("ROLLING_HIGH")

    if row.get("is_breakout", 0) == 1:
        alerts.append("BREAKOUT")

    if row.get("is_ath", 0) == 1 and row.get("is_breakout", 0) == 1:
        alerts.append("ATH_BREAKOUT")

    if row.get("fresh_breakout", 0) == 1 and row.get("posvol_above_ema21", 0) == 1:
        alerts.append("FRESH_BREAKOUT_VOLUME")

    if row.get("is_breakout", 0) == 1 and row.get("ravi_buy", 0) == 1:
        alerts.append("BREAKOUT_RAVI_BUY")

    return ",".join(alerts)


if not df_eod.empty:
    df_eod["smart_money_alerts"] = df_eod.apply(
        build_smart_money_alerts, axis=1
    )
    df_eod["has_smart_alert"] = df_eod["smart_money_alerts"] != ""

PIVOT_SCHEMA = [
    "symbol",
    "cnt_breakout_strength",
    "SUPER_TRADE_SIGNAL",
    "rolling_alert_score_5",
    "rolling_alert_signal",
    "rolling_alert_signal_old",
    "HLC_TREND_STRENGTH",
    "superbuy_count",
    "supersell_count",
]

def empty_alert_pivot():
    return pd.DataFrame({c: [] for c in PIVOT_SCHEMA})




# =========================================================
# 🔹 TRADE DATE (SAFE — HANDLES HOLIDAYS)
# =========================================================
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])

# df["trade_date"] = df["timestamp"].dt.date
# today = df["trade_date"].max()   # safest way

def compute_smart_score(row):
    score = 0

    mf = pd.to_numeric(row.get("money_change_%", 0), errors="coerce") or 0
    di = pd.to_numeric(row.get("daily_inflow_change_%", 0), errors="coerce") or 0

    if mf > 150:
        score += 40
    elif mf > 80:
        score += 25

    if di > 50:
        score += 30
    elif di > 20:
        score += 15

    if row.get("eod_confirm", 0) == 1:
        score += 30

    return min(score, 100)


if not df_eod.empty:
    latest_eod = (
        df_eod.sort_values("date")
              .groupby("symbol")
              .tail(1)[["symbol", "smart_money_alerts", "has_smart_alert"]]
    )

    df = df.merge(
        latest_eod,
        on="symbol",
        how="left"
    )

    df["eod_confirm"] = df["has_smart_alert"].fillna(False).astype(int)
else:
    df["eod_confirm"] = 0


df["smart_score"] = df.apply(compute_smart_score, axis=1)

# ================== NORMALIZE SUPER TRADE SIGNAL (SAFE) ==================
if "SUPER_TRADE_SIGNAL" in df.columns:
    df["SUPER_TRADE_SIGNAL"] = (
        df["SUPER_TRADE_SIGNAL"]
        .fillna("NONE")
        .astype(str)
        .str.strip()
    )
else:
    df["SUPER_TRADE_SIGNAL"] = "NONE"


# ================== FILE UPDATE CONFIRMATION ==================
last_updated = datetime.fromtimestamp(os.path.getmtime(selected_file))
st.caption(f"🕒 CSV Last Updated: {last_updated}")

# ================== FILTERS ==================
st.sidebar.title("🔍 Filters")

symbols = sorted(df["symbol1"].unique())

alerts = [a for a in ALLOWED_ALERTS if a in df["trigger_alert"].unique()]


selected_symbols = st.sidebar.multiselect("symbol", symbols)
selected_alerts = st.sidebar.multiselect("Trigger Alert", alerts)

# ================== SUPER TRADE FILTER ==================
super_trade_options = sorted(
    df["SUPER_TRADE_SIGNAL"].unique().tolist()
)

selected_super_trades = st.sidebar.multiselect(
    "Super Trade Signal",
    super_trade_options,
    default=[]
)


if selected_symbols:
    df = df[df["symbol1"].isin(selected_symbols)]

if selected_alerts:
    df = df[df["trigger_alert"].isin(selected_alerts)]

# ================== APPLY SUPER TRADE FILTER ==================
if selected_super_trades:
    df = df[df["SUPER_TRADE_SIGNAL"].isin(selected_super_trades)]


# ================== DATE SELECTION (PAST + TODAY) ==================
# st.sidebar.markdown("---")
# st.sidebar.title("📅 Trade Date Selection")

# available_dates = sorted(df["trade_date"].unique())

# default_date = max(available_dates)  # latest available safely

# selected_trade_date = st.sidebar.date_input(
#     "Select Trade Date",
#     value=default_date,
#     min_value=min(available_dates),
#     max_value=max(available_dates)
# )

if selected_trade_date not in available_dates:
    st.warning("No data available for selected date")
    df_selected = pd.DataFrame()
else:
    # df_selected = df[df["trade_date"] == selected_trade_date].copy()
    df_selected = df[
    (df["trade_date"] == selected_trade_date) &
    (df["timestamp"] <= selected_datetime)
    ].copy()


if df_selected.empty:
    df_latest = pd.DataFrame()
else:
    df_latest = (
        df_selected
        .sort_values("timestamp")
        .groupby("symbol", as_index=False)
        .tail(1)
    )


latest_ts = df_latest["timestamp"].max() if not df_latest.empty else None

df_latest_ts = df_selected[
    df_selected["timestamp"] == latest_ts
].copy()

if latest_ts is not None and "supertrend_dir" in df_selected.columns:
    st_dir = pd.to_numeric(
        df_selected.loc[df_selected["timestamp"] == latest_ts, "supertrend_dir"],
        errors="coerce"
    ).fillna(0)

    bull_cnt = int((st_dir == 1).sum())
    bear_cnt = int((st_dir == -1).sum())
else:
    bull_cnt = bear_cnt = 0


df_selected = df_selected.copy()

###=== MOMENTUM CALCULATION FUNCTIONS ===###



def build_alert_strength_pivot(df_selected):
    """
    Builds a pivot table with:
    - Rows: symbol
    - Columns: alert type counts (FRESH_BREAKOUT, ATH, etc.)
    - Extra column: cnt_breakout_strength (SUM per symbol)

    Input:
        df_selected : DataFrame (already filtered for selected_trade_date)

    Output:
        Pivoted DataFrame
    """

    # if df_selected.empty:
    #     return pd.DataFrame()
    if df_selected.empty:
        return empty_alert_pivot()


    # -------------------------------
    # Define alert buckets
    # -------------------------------
    ALERT_BUCKETS = {
        "FRESH_BREAKOUT": "FRESH_BREAKOUT",
        "ATH": "ATH",
        "ATL": "ATL",
        "BREAKOUT_MONEY_SURGE": "BREAKOUT_MONEY_SURGE",
        "BREAKDOWN_MONEY_SURGE": "BREAKDOWN_MONEY_SURGE",
        "BREAKOUT_MONEY_SPIKE": "BREAKOUT_MONEY_SPIKE",
        "BREAKDOWN_MONEY_SPIKE": "BREAKDOWN_MONEY_SPIKE",
        "BREAKOUT+ATH": "BREAKOUT+ATH",
        "BREAKOUT+UMM": "BREAKOUT+UMM",
        "BREAKDOWN+ATL": "BREAKDOWN+ATL",
        "FRESH_BREAKDOWN": "FRESH_BREAKDOWN"
    }

    # -------------------------------
    # Normalize trigger_alert → bucket
    # -------------------------------
    def map_alert_bucket(val):
        if pd.isna(val):
            return None

        v = str(val).upper()
        for k in ALERT_BUCKETS:
            if k in v:
                return ALERT_BUCKETS[k]
        return None

    df = df_selected.copy()

    # if "trigger_alert" not in df.columns:
    #     return pd.DataFrame()
    if "trigger_alert" not in df.columns:
        return empty_alert_pivot()


    df["alert_bucket"] = df["trigger_alert"].apply(map_alert_bucket)

    df = df[df["alert_bucket"].notna()]

    # if df.empty:
    #     return pd.DataFrame()
    if df.empty:
        return empty_alert_pivot()


    # -------------------------------
    # Pivot: COUNT of alerts
    # -------------------------------
    pivot_counts = (
        df
        .groupby(["symbol", "alert_bucket"])
        .size()
        .unstack(fill_value=0)
    )

        # -------------------------------
    # Bring latest rolling columns from CSV (NO recompute)
    # -------------------------------
    # -------------------------------
    # Bring latest rolling columns (GUARANTEED)
    # -------------------------------
    extra_cols = [
        "rolling_alert_score_5",
         "SUPER_TRADE_SIGNAL",
        "HLC_TREND_STRENGTH",
         "superbuy_count",
         "supersell_count",
        "rolling_alert_signal",
        "rolling_alert_signal_old"
       
    ]

    latest_extra = (
        df.sort_values("timestamp")
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol"] + [c for c in extra_cols if c in df.columns]]
        .set_index("symbol")
    )

    # merge whatever exists
    pivot_counts = pivot_counts.merge(
        latest_extra,
        left_index=True,
        right_index=True,
        how="left"
    )

    # 🔒 GUARANTEE missing rolling columns exist
    for c in extra_cols:
        if c not in pivot_counts.columns:
            pivot_counts[c] = 0


    # -------------------------------
    # Add cnt_breakout_strength (SUM)
    # -------------------------------
    if "cnt_breakout_strength" in df.columns:
        strength_sum = (
            df.groupby("symbol")["cnt_breakout_strength"]
            .sum()
        )
    else:
        strength_sum = pd.Series(0, index=pivot_counts.index)

    pivot_counts["cnt_breakout_strength"] = strength_sum

    # -------------------------------
    # Final formatting
    # -------------------------------
        # -------------------------------
    # Final formatting (FIX COLUMN ORDER)
    # -------------------------------
    pivot_counts = pivot_counts.reset_index()

    # Desired column order:
    # symbol | cnt_breakout_strength | alert counts...
    base_cols = [
    "symbol",
    "cnt_breakout_strength",
    "SUPER_TRADE_SIGNAL",
    "rolling_alert_score_5",
    "rolling_alert_signal",
]


    alert_cols = [
        c for c in pivot_counts.columns
        if c not in base_cols
    ]

    # Reorder columns
    pivot_counts = pivot_counts[base_cols + alert_cols]

    # Sort strongest first
    pivot_counts = pivot_counts.sort_values(
        "cnt_breakout_strength", ascending=False
    )

    return pivot_counts

###===

# alert_pivot = build_alert_strength_pivot(df_selected)
# alert_pivot_latest = build_alert_strength_pivot(df_latest_ts)

alert_pivot = build_alert_strength_pivot(df_selected)
# 🔒 HARD GUARANTEE PIVOT SCHEMA
for c in PIVOT_SCHEMA:
    if c not in alert_pivot.columns:
        alert_pivot[c] = 0


# derive latest FROM SAME pivot
# alert_pivot_latest = (
#     alert_pivot
#     .merge(
#         df_latest_ts[["symbol"]],
#         on="symbol",
#         how="inner"
#     )
# )

# ================= SAFE LATEST PIVOT =================
if (
    alert_pivot.empty or
    df_latest_ts.empty or
    "symbol" not in alert_pivot.columns or
    "symbol" not in df_latest_ts.columns
):
    alert_pivot_latest = empty_alert_pivot()
else:
    alert_pivot_latest = alert_pivot.merge(
        df_latest_ts[["symbol"]],
        on="symbol",
        how="inner"
    )

# 🔒 HARD ENFORCE SCHEMA (ABSOLUTE SAFETY)
for c in PIVOT_SCHEMA:
    if c not in alert_pivot_latest.columns:
        alert_pivot_latest[c] = 0





# # ================== LATEST ONLY DATA ==================
# if df_latest.empty:
#     alert_pivot_latest = pd.DataFrame()
# else:
#     alert_pivot_latest = build_alert_strength_pivot(df_latest)


positive_trend = alert_pivot[
    alert_pivot["cnt_breakout_strength"] > 0
]

negative_trend = alert_pivot[
    alert_pivot["cnt_breakout_strength"] < 0
]

# ================== NEW SIGNAL TRIGGERS ==================
# buy_triggers = alert_pivot[
#     (alert_pivot["rolling_alert_signal"] == 1) &
#     (alert_pivot["rolling_alert_signal_old"] == 0)
# ].copy()

# sell_triggers = alert_pivot[
#     (alert_pivot["rolling_alert_signal"] == -1) &
#     (alert_pivot["rolling_alert_signal_old"] == 0)
# ].copy()

buy_triggers = alert_pivot_latest[
    (alert_pivot_latest["rolling_alert_signal"] == 1) &
    (alert_pivot_latest["rolling_alert_signal_old"] == 0)
].copy()

sell_triggers = alert_pivot_latest[
    (alert_pivot_latest["rolling_alert_signal"] == -1) &
    (alert_pivot_latest["rolling_alert_signal_old"] == 0)
].copy()

buy_cnt = len(buy_triggers)
sell_cnt = len(sell_triggers)



def highlight_trend(row):
    sig = row.get("rolling_alert_signal", 0)

    if sig == 1:
        return ["background-color:#e8f5e9; font-weight:bold"] * len(row)
    elif sig == -1:
        return ["background-color:#fdecea; font-weight:bold"] * len(row)
    return [""] * len(row)





POSITIVE_COLS = [
    "symbol",
    "cnt_breakout_strength",
    "SUPER_TRADE_SIGNAL",
    "superbuy_count",
    "HLC_TREND_STRENGTH",
    "rolling_alert_score_5",
    "rolling_alert_signal",
    "ATH",
    "BREAKOUT_MONEY_SURGE",
    "BREAKOUT_MONEY_SPIKE",
    "FRESH_BREAKOUT"
]

NEGATIVE_COLS = [
    "symbol",
    "cnt_breakout_strength",
    "SUPER_TRADE_SIGNAL",
    "HLC_TREND_STRENGTH",
    "supersell_count",
    "rolling_alert_score_5",
    "rolling_alert_signal",
    "ATL",
    "BREAKDOWN_MONEY_SURGE",
    "BREAKDOWN_MONEY_SPIKE",
    "FRESH_BREAKDOWN"
]



pos_cols = [c for c in POSITIVE_COLS if c in alert_pivot.columns]
neg_cols = [c for c in NEGATIVE_COLS if c in alert_pivot.columns]


TRIGGER_COLS = [
    "symbol",
    "cnt_breakout_strength",
    "SUPER_TRADE_SIGNAL",
    "HLC_TREND_STRENGTH",
    "rolling_alert_score_5",
    "rolling_alert_signal_old",
    "rolling_alert_signal",
    "FRESH_BREAKOUT",
    "FRESH_BREAKDOWN",
    "BREAKOUT_MONEY_SURGE",
    "BREAKDOWN_MONEY_SURGE",
]

# trigger_cols = [c for c in TRIGGER_COLS if c in alert_pivot.columns]
trigger_cols = [c for c in TRIGGER_COLS if c in alert_pivot_latest.columns]



def compute_market_trend_from_supertrend(df, ts, threshold=120):
    """
    Computes market-wide trend using supertrend_dir
    at a specific timestamp across ALL symbols
    """
    if "supertrend_dir" not in df.columns:
        return "NA", 0, 0

    snap = df[df["timestamp"] == ts]

    if snap.empty:
        return "NA", 0, 0

    st_dir = pd.to_numeric(
        snap["supertrend_dir"], errors="coerce"
    ).fillna(0)

    bull_cnt = int((st_dir == 1).sum())
    bear_cnt = int((st_dir == -1).sum())

    if bull_cnt >= threshold:
        trend = f"BULLISH ({bull_cnt})"
    elif bear_cnt >= threshold:
        trend = f"BEARISH ({bear_cnt})"
    else:
        trend = f"NEUTRAL ({bull_cnt}/{bear_cnt})"

    return trend, bull_cnt, bear_cnt

def build_index_smart_money_table(day_df):
    """
    Index Smart Money table
    - Index rows only
    - Breakout / Breakdown only
    - Market-wide Trend from supertrend_dir (breadth)
    """
    if day_df.empty:
        return pd.DataFrame()

    idx_df = day_df[day_df["symbol"].str.startswith(INDEX_SYMBOLS)].copy()

    if idx_df.empty:
        return pd.DataFrame()

    idx_df = idx_df[
        idx_df["trigger_alert"].str.contains(
            "FRESH_BREAKOUT|FRESH_BREAKDOWN|ATL|ATH|BREAKOUT+ATH|BREAKDOWN+ATL|BREAKOUT_MONEY_SURGE|BREAKDOWN_MONEY_SURGE|BREAKOUT_MONEY_SPIKE|BREAKDOWN_MONEY_SPIKE",
            na=False
        )
    ]

    if idx_df.empty:
        return pd.DataFrame()

    rows = []

    for sym in idx_df["symbol"].unique():
        sym_df = idx_df[idx_df["symbol"] == sym]

        # ✅ latest candle FOR THIS SYMBOL
        latest_row = (
            sym_df.sort_values("timestamp")
                  .tail(1)
                  .iloc[0]
        )

        ts = latest_row["timestamp"]

        # ✅ Market-wide trend at SAME timestamp
        trend, bull_cnt, bear_cnt = compute_market_trend_from_supertrend(
            day_df, ts, threshold=120
        )

        rows.append({
            "symbol": sym,
            "timestamp": ts,
            "close": latest_row["close"],
            "trigger_alert": latest_row["trigger_alert"],
            "Market_Trend": trend,
            # "Bull_Count": bull_cnt,
            # "Bear_Count": bear_cnt,
            "SUPER_TRADE_SIGNAL": latest_row.get("SUPER_TRADE_SIGNAL"),
            "Bull_Count": latest_row.get("superbuy_count"),
            "Bear_Count": latest_row.get("supersell_count"),
            "cnt_breakout_strength": latest_row.get("cnt_breakout_strength"),
            "money_change_%": latest_row.get("money_change_%"),
            "daily_inflow_change_%": latest_row.get("daily_inflow_change_%"),
            "direction_bias": latest_row.get("direction_bias"),
            "flow_signal": latest_row.get("flow_signal"),
            "fo_buildup": latest_row.get("fo_buildup"),
        })

    out = pd.DataFrame(rows)

    sort_col = (
        "cnt_breakout_strength"
        if "cnt_breakout_strength" in out.columns
        else "money_change_%"
    )

    return out.sort_values(sort_col, ascending=False)

# ================== INDEX SMART MONEY TABLE ==================
st.subheader("📊 Smart Money Activity — Index Futures")

# index_smart_money = build_index_smart_money_table(df)
index_smart_money = build_index_smart_money_table(df_selected)




if index_smart_money.empty:
    st.info("No index breakout / breakdown activity")
else:
    st.dataframe(
    index_smart_money.style
        .applymap(breakout_breakdown_style, subset=["trigger_alert"])
        .format({
            "money_change_%": "{:.2f}",
            "daily_inflow_change_%": "{:.2f}",
        }),
    width="stretch",
    height=260
    )

NIFTY50_WEIGHTS = { "HDFCBANK": 0.11, "RELIANCE": 0.10, "ICICIBANK": 0.07, "INFY": 0.06, "TCS": 0.04, "ITC": 0.04, "BHARTIARTL": 0.04, "KOTAKBANK": 0.03, "SBIN": 0.03, "HINDUNILVR": 0.03, }
DEFAULT_NIFTY_WT = 0.005
SENSEX_WEIGHTS = { "RELIANCE": 0.12, "HDFCBANK": 0.10, "ICICIBANK": 0.07, "INFY": 0.06, "TCS": 0.05, "HINDUNILVR": 0.04 }
DEFAULT_SENSEX_WT = 0.008
BANKNIFTY_WEIGHTS = { "HDFCBANK": 0.28, "ICICIBANK": 0.23, "KOTAKBANK": 0.13, "AXISBANK": 0.12, "SBIN": 0.12, "INDUSINDBK": 0.06, "BANKBARODA": 0.03, "FEDERALBNK": 0.02, "PNB": 0.02, "IDFCFIRSTB": 0.015, "BANDHANBNK": 0.01, }
DEFAULT_BANKNIFTY_WT = 0.005

def compute_index_supertrend_pie(df_latest, weight_map, default_weight):
    pos = neg = neu = 0.0

    for stock, wt in weight_map.items():
        row = df_latest[df_latest["symbol1"] == stock]

        if row.empty:
            st_dir = 0
            wt = default_weight
        else:
            #st_dir = int(row.iloc[0]["supertrend_dir"])
            st_dir = int(row.iloc[0]["HLC_TREND_STRENGTH"]) #HLC_TREND_STRENGTH HLC_TREND_DIR
            #st_dir = int(row.iloc[0]["rolling_alert_signal"])

        if st_dir >= 1:
            pos += wt
        elif st_dir <= -1:
            neg += wt
        else:
            neu += wt

    total = pos + neg + neu
    if total == 0:
        return [0, 0, 100]  # fully neutral fallback

    return [
        round(pos / total * 100, 1),
        round(neg / total * 100, 1),
        round(neu / total * 100, 1),
    ]

nifty_pie = compute_index_supertrend_pie(
    df_latest,
    NIFTY50_WEIGHTS,
    DEFAULT_NIFTY_WT
)

banknifty_pie = compute_index_supertrend_pie(
    df_latest,
    BANKNIFTY_WEIGHTS,
    DEFAULT_BANKNIFTY_WT
)

sensex_pie = compute_index_supertrend_pie(
    df_latest,
    SENSEX_WEIGHTS,
    DEFAULT_SENSEX_WT
)

def draw_pie(data, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        data,
        labels=["Positive", "Negative", "Neutral"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2e7d32", "#c62828", "#f9a825"],
        wedgeprops={"edgecolor": "white"}
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("📊 Index Supertrend Breadth (Weighted)")

c1, c2, c3 = st.columns(3)

with c1:
    draw_pie(nifty_pie, "NIFTY 50")

with c2:
    draw_pie(banknifty_pie, "BANK NIFTY")

with c3:
    draw_pie(sensex_pie, "SENSEX")



# ================== UI ==================

st.subheader(f"📊 Momentum Strength — {selected_trade_date}")

# ================== METRICS ==================
c1, c2 = st.columns(2)
c1.metric("📈 Positive Symbols", len(positive_trend))
c2.metric("📉 Negative Symbols", len(negative_trend))

if alert_pivot.empty:
    st.info("No alert activity for selected date")

else:
    positive_trend = alert_pivot[
        alert_pivot["cnt_breakout_strength"] > 0
    ].sort_values("cnt_breakout_strength", ascending=False)

    negative_trend = alert_pivot[
        alert_pivot["cnt_breakout_strength"] < 0
    ].sort_values("cnt_breakout_strength")

    buy_cnt = len(buy_triggers)
    sell_cnt = len(sell_triggers)

    trigger_tab_label = f"🚨 New Signal Triggers (🟢 {buy_cnt} | 🔴 {sell_cnt})"

    # ================== SUPER TRADE DATA ==================
    super_buy = df_latest[
        df_latest["SUPER_TRADE_SIGNAL"] == "SuperBuy"
    ].copy()

    super_sell = df_latest[
        df_latest["SUPER_TRADE_SIGNAL"] == "SuperSell"
    ].copy()

    super_buy_cnt = len(super_buy)
    super_sell_cnt = len(super_sell)



    # ================== HLC TREND STRENGTH (LATEST ONLY) ==================

    # Hard safety
    if "HLC_TREND_STRENGTH" not in df_latest.columns:
        df_latest["HLC_TREND_STRENGTH"] = 0

    hlc_bull = df_latest[
        df_latest["HLC_TREND_STRENGTH"] >= 2
    ].copy()

    hlc_bear = df_latest[
        df_latest["HLC_TREND_STRENGTH"] <= -2
    ].copy()

    hlc_bull_cnt = len(hlc_bull)
    hlc_bear_cnt = len(hlc_bear)



    super_tab_label = f"🧨 Super Trades (🟢 {super_buy_cnt} | 🔴 {super_sell_cnt})"

    hlc_tab_label = f"📐 HLC Trend Strength (🟢 {hlc_bull_cnt} | 🔴 {hlc_bear_cnt})"

    tab_up, tab_down, tab_trigger, tab_super, tab_hlc = st.tabs([
        "📈 Positive Trend",
        "📉 Down Trend",
        trigger_tab_label,
        super_tab_label,
        hlc_tab_label
    ])




    with tab_up:
        st.subheader("📈 Bullish Momentum")
        if positive_trend.empty:
            st.info("No bullish symbols")
        else:
            st.dataframe(
                positive_trend[pos_cols]
                    .sort_values("cnt_breakout_strength", ascending=False)
                    .style.apply(highlight_trend, axis=1),
                height=500,
                use_container_width=True
            )


    with tab_down:       
        st.subheader("📉 Bearish Momentum")
        if negative_trend.empty:
            st.info("No bearish symbols")
        else:
            st.dataframe(
                negative_trend[neg_cols]
                    .sort_values("cnt_breakout_strength")
                    .style.apply(highlight_trend, axis=1),
                height=500,
                use_container_width=True
            )

    with tab_trigger:
        st.subheader("🚨 Fresh Buy / Sell Triggers")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🟢 BUY Triggers")
            if buy_triggers.empty:
                st.info("No fresh BUY triggers")
            else:
                st.dataframe(
                    buy_triggers[trigger_cols]
                        .sort_values("cnt_breakout_strength", ascending=False)
                        .style.apply(highlight_trend, axis=1),
                    height=400,
                    use_container_width=True
                )

        with c2:
            st.markdown("### 🔴 SELL Triggers")
            if sell_triggers.empty:
                st.info("No fresh SELL triggers")
            else:
                st.dataframe(
                    sell_triggers[trigger_cols]
                        .sort_values("cnt_breakout_strength")
                        .style.apply(highlight_trend, axis=1),
                    height=400,
                    use_container_width=True
                )
    with tab_super:
        st.subheader("🧨 Super Trades (OI + Alert Confirmation)")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🟢 SuperBuy")
            if super_buy.empty:
                st.info("No SuperBuy signals")
            else:
                st.dataframe(
                    super_buy[
                        [
                            "symbol",
                            "timestamp",
                            "close",
                            "superbuy_count",
                            "rolling_alert_score_5",
                            "fo_buildup",
                            "money_change_%"
                        ]
                    ].sort_values("rolling_alert_score_5", ascending=False),
                    height=400,
                    use_container_width=True
                )

        with c2:
            st.markdown("### 🔴 SuperSell")
            if super_sell.empty:
                st.info("No SuperSell signals")
            else:
                st.dataframe(
                    super_sell[
                        [
                            "symbol",
                            "timestamp",
                            "close",
                             "supersell_count",
                            "rolling_alert_score_5",
                            "fo_buildup",
                            "money_change_%"
                        ]
                    ].sort_values("rolling_alert_score_5"),
                    height=400,
                    use_container_width=True
                )
    with tab_hlc:

        st.subheader("📐 Current HLC Trend Strength (Pure Price Structure)")

        c1, c2 = st.columns(2)

        # ================== BULLISH HLC ==================
        with c1:
            st.markdown("### 🟢 Strong Bullish Structure (+2 / +3)")

            if hlc_bull.empty:
                st.info("No strong bullish HLC trends")
            else:
                st.dataframe(
                    hlc_bull[
                        [
                            "symbol",
                            "timestamp",
                            "close",
                            "HLC_TREND_STRENGTH",
                            "SUPER_TRADE_SIGNAL",
                            "superbuy_count",
                            "rolling_alert_score_5",
                            "fo_buildup",
                            "money_change_%"
                        ]
                    ].sort_values(
                        "HLC_TREND_STRENGTH", ascending=False
                    ),
                    height=400,
                    use_container_width=True
                )

        # ================== BEARISH HLC ==================
        with c2:
            st.markdown("### 🔴 Strong Bearish Structure (-2 / -3)")

            if hlc_bear.empty:
                st.info("No strong bearish HLC trends")
            else:
                st.dataframe(
                    hlc_bear[
                        [
                            "symbol",
                            "timestamp",
                            "close",
                            "HLC_TREND_STRENGTH",
                            "SUPER_TRADE_SIGNAL",
                            "supersell_count",
                            "rolling_alert_score_5",
                            "fo_buildup",
                            "money_change_%"
                        ]
                    ].sort_values(
                        "HLC_TREND_STRENGTH"
                    ),
                    height=400,
                    use_container_width=True
                )

        # ================== TABS ==================


    # with tab_up:
    #     st.subheader("📈 Bullish Momentum")
    #     if positive_trend.empty:
    #         st.info("No bullish symbols")
    #     else:
    #         st.dataframe(
    #             positive_trend.style.apply(highlight_trend, axis=1),
    #             height=500,
    #             use_container_width=True
    #         )

    # with tab_down:
    #     st.subheader("📉 Bearish Momentum")
    #     if negative_trend.empty:
    #         st.info("No bearish symbols")
    #     else:
    #         st.dataframe(
    #             negative_trend.style.apply(highlight_trend, axis=1),
    #             height=500,
    #             use_container_width=True
    #         )





# st.dataframe(positive_trend.style.apply(highlight_trend, axis=1))



# ================== HEADER ==================
st.title("📊 F&O Futures – Smart Money Dashboard")
st.caption(f"Source: {selected_file}")

# ================== KPI ROW ==================
c1, c2, c3, c4, c5 = st.columns(5) 
threshold=120

c1.metric("symbols", df_latest["symbol"].nunique())
c2.metric("Breakouts", int((df_selected["is_breakout"] == 1).sum()))
c3.metric("Breakdowns", int((df_selected["is_breakdown"] == 1).sum()))

# c1.metric("symbols", df["symbol"].nunique())
# c2.metric("Breakouts", int((df["is_breakout"] == 1).sum()))
# c3.metric("Breakdowns", int((df["is_breakdown"] == 1).sum()))
# c4.metric("Smart Money", int((df["money_change_spike"] == 1).sum()))
if bull_cnt > threshold:
    c4.metric(
        "Market Trend",
        f"BULLISH",
        # delta=f"+{bull_cnt - bear_cnt}"
        delta=f"+{bull_cnt}/{bear_cnt}"
    )
elif bear_cnt > threshold:
    c4.metric(
        "Market Trend",
        f"BEARISH",
        delta=f"+{bull_cnt}/{bear_cnt}"
    )
else:
    c4.metric(
        "Market Trend",
        "NEUTRAL",
        # delta="0"
        delta=f"+{bull_cnt}/{bear_cnt}"
    )

c5.metric(
    "Smart Score > 70",
    int((df_selected["smart_score"] >= 70).sum())
)



# ================== CNT BREAKOUT STRENGTH BUCKET ==================
# =========================================================
# 🔹 STRENGTH BUCKET FUNCTION (DEFINE FIRST — VERY IMPORTANT)
# =========================================================
def strength_bucket(val):
    try:
        v = int(val)
    except:
        return "UNKNOWN"

    if v >= 20:
        return "🔥 +20 to +25 (Very Strong BO)"
    elif v >= 10:
        return "🟢 +10 to +19 (Strong BO)"
    elif v >= 1:
        return "🟡 +1 to +9 (Weak BO)"
    elif v == 0:
        return "⚪ 0 (Neutral)"
    elif v >= -9:
        return "🟠 -1 to -9 (Weak BD)"
    elif v >= -19:
        return "🔴 -10 to -19 (Strong BD)"
    else:
        return "💀 ≤ -20 (Very Strong BD)"


# =========================================================
# 🔹 GUARANTEE REQUIRED COLUMNS EXIST
# =========================================================
required_cols = ["timestamp", "symbol", "cnt_breakout_strength"]

for c in required_cols:
    if c not in df.columns:
        st.error(f"Required column missing: {c}")
        st.stop()





# =========================================================
# 🔹 TODAY ONLY DATA
# =========================================================
# df_today = df[df["trade_date"] == today].copy()

# if df_today.empty:
#     st.warning("No data available for today")
#     df_today_latest = pd.DataFrame()
# else:
    # =====================================================
    # 🔹 LATEST CANDLE PER SYMBOL (TODAY)
    # =====================================================
    # df_today_latest = (
    #     df_today
    #     .sort_values("timestamp")
    #     .groupby("symbol", as_index=False)
    #     .tail(1)
    # )
    #df_today_latest = df_latest.copy()

# =========================================================
# 🔹 SELECTED DAY DATA (LATEST PER SYMBOL)
# =========================================================
df_today_latest = df_latest.copy()

if df_today_latest.empty:
    st.warning("No data available for selected date")


# =========================================================
# 🔹 MAIN TABS
# =========================================================
tab_latest, tab_strength, tab_smart = st.tabs([
    "📌 Latest Breakout / Breakdown / ATH",
    "📊 Bucket Strength (Today)",
    "💰 Smart Money Activity"
])




# =========================================================
# 🔹 TAB 3 — SMART MONEY ACTIVITY
# =========================================================
with tab_smart:
    st.subheader("💰 Smart Money Activity")

    # smart_money = df_selected[
    #     df_selected["trigger_alert"].isin([
    #         "FRESH_BREAKDOWN",
    #         "BREAKOUT_MONEY_SURGE",
    #         "BREAKDOWN_MONEY_SURGE",
    #         "BREAKOUT_MONEY_SPIKE",
    #         "BREAKDOWN_MONEY_SPIKE",
    #         "FRESH_BREAKOUT",
    #         "BREAKOUT+ATH",
    #         "ATH",
    #         "ATL",
    #         "BREAKDOWN+ATL",
    #         "BREAKDOWN+UMM",
    #         "BREAKOUT+ATH+UMM",
    #         "BREAKDOWN+UMM",
    #         "STACKED_BREAKOUT_MONEY_ATH",
    #         "STACKED_BREAKDOWN_MONEY_ATL"
    #     ])
    # ].sort_values("timestamp", ascending=False)
    # 🔒 Safety: guarantee column exists
    if "HLC_TREND_STRENGTH" not in df_selected.columns:
        df_selected["HLC_TREND_STRENGTH"] = 0

    smart_money = df_selected[
        (df_selected["trigger_alert"].isin([
            "FRESH_BREAKDOWN",
            "BREAKOUT_MONEY_SURGE",
            "BREAKDOWN_MONEY_SURGE",
            "BREAKOUT_MONEY_SPIKE",
            "BREAKDOWN_MONEY_SPIKE",
            "FRESH_BREAKOUT",
            "BREAKOUT+ATH",
            "ATH",
            "ATL",
            "BREAKDOWN+ATL",
            "BREAKDOWN+UMM",
            "BREAKOUT+ATH+UMM",
            "STACKED_BREAKOUT_MONEY_ATH",
            "STACKED_BREAKDOWN_MONEY_ATL"
        ]))
        |
        (
            (df_selected["HLC_TREND_STRENGTH"] >= 2) |
            (df_selected["HLC_TREND_STRENGTH"] <= -2)
        )
        |
        (
            (df_selected["SUPER_TRADE_SIGNAL"] == "SuperBuy") |
            (df_selected["SUPER_TRADE_SIGNAL"] == "SuperSell")
        )
        ].sort_values("timestamp", ascending=False)


    if smart_money.empty:
        st.info("No smart money activity detected")
    else:
        st.dataframe(
            smart_money[
                [
                    "timestamp", "symbol", "close",
                    "trigger_alert" , "SUPER_TRADE_SIGNAL","HLC_TREND_STRENGTH", "cnt_breakout_strength","superbuy_count","supersell_count",
                    "money_change_%", "daily_inflow_change_%",
                    "fo_buildup"
                ]
            ].style.applymap(
                breakout_breakdown_style, subset=["trigger_alert"]
            ),
            width="stretch",
            height=600
        )



# =========================================================
# 🔹 TAB 1 — LATEST EVENTS (COLOURED + BLINKING)
# =========================================================
with tab_latest:
    st.subheader("📌 Latest Breakout / Breakdown / ATH")
    latest_df = df_latest.copy()


    # latest_df = (
    #     df.sort_values("timestamp")
    #       .groupby("symbol")
    #       .tail(1)
    #       .copy()
    # )

    # ---------------- EVENT TYPE ----------------
    def latest_event(row):
        if row.get("is_breakout", 0) == 1:
            return "BREAKOUT"
        if row.get("is_breakdown", 0) == 1:
            return "BREAKDOWN"
        if row.get("is_ath", 0) == 1:
            return "ATH"
        if row.get("is_atl", 0) == 1:
            return "ATL"
        return None

    latest_df["event"] = latest_df.apply(latest_event, axis=1)
    event_df = latest_df[latest_df["event"].notna()].copy()

    if event_df.empty:
        st.info("No latest breakout / breakdown / ATH events")
    else:
        # ---------------- NEW EVENT DETECTION ----------------
        event_df["event_key"] = event_df["symbol"] + "|" + event_df["event"]

        if "prev_events" not in st.session_state:
            st.session_state.prev_events = set()

        event_df["is_new"] = ~event_df["event_key"].isin(
            st.session_state.prev_events
        )
        st.session_state.prev_events = set(event_df["event_key"])

        # ---------------- ROW STYLE ----------------
        def highlight_event_row(row):
            alert = str(row.get("event", "")).upper()
            is_new = row.get("is_new", False)

            if "BREAKOUT" in alert or "ATH" in alert:
                base = "background-color:#e8f5e9; color:#1b5e20; font-weight:700"
            elif "BREAKDOWN" in alert or "ATL" in alert:
                base = "background-color:#fdecea; color:#b71c1c; font-weight:700"
            else:
                base = ""

            if is_new and base:
                base += "; animation: blink 1.2s ease-in-out infinite"

            return [base] * len(row)

        display_cols = [
            "symbol",
            "timestamp",
            "close",
            "event",
            "SUPER_TRADE_SIGNAL", 
            "HLC_TREND_STRENGTH",
            "cnt_breakout_strength",
            "fo_buildup",
            "money_change_%",
            "daily_inflow_change_%",
            "price_change_%"
        ]

        styled_df = (
            event_df
            .sort_values("money_change_%", ascending=False)[display_cols]
            .style
            .apply(highlight_event_row, axis=1)
        )

        st.dataframe(
            styled_df,
            width="stretch",
            height=600
        )
# =========================================================
# 🔹 TAB 2 — STRENGTH BUCKETS (TODAY ONLY | SAFE)
# =========================================================
with tab_strength:
    st.subheader(f"📊 Bucket Strength — {selected_trade_date}")

    if df_today_latest.empty:
        st.info("No strength data available for today")
    else:
        # -------------------------------------------------
        # GUARANTEE strength_bucket EXISTS (CRITICAL FIX)
        # -------------------------------------------------
        if "strength_bucket" not in df_today_latest.columns:
            df_today_latest["strength_bucket"] = (
                df_today_latest["cnt_breakout_strength"]
                .apply(strength_bucket)
            )

        bucket_order = [
            "🔥 +20 to +25 (Very Strong BO)",
            "🟢 +10 to +19 (Strong BO)",
            "🟡 +1 to +9 (Weak BO)",
            "⚪ 0 (Neutral)",
            "🟠 -1 to -9 (Weak BD)",
            "🔴 -10 to -19 (Strong BD)",
            "💀 ≤ -20 (Very Strong BD)"
        ]

        for bucket in bucket_order:
            sub = df_today_latest.loc[
                df_today_latest["strength_bucket"] == bucket
            ].sort_values(
                "cnt_breakout_strength", ascending=False
            )

            if sub.empty:
                continue

            with st.expander(
                f"{bucket}  •  {len(sub)} symbols",
                expanded=bucket.startswith("🔥") or bucket.startswith("🟢")
            ):
                st.dataframe(
                    sub[
                        [
                            "symbol",
                            "timestamp",
                            "close",
                            "trigger_alert",
                            "cnt_breakout_strength",
                            "money_change_%",
                            "daily_inflow_change_%",
                            "fo_buildup"
                        ]
                    ],
                    width="stretch",
                    height=400
                )




# # ================== LATEST EVENTS ==================
# st.subheader("📌 Latest Breakout / Breakdown / ATH")

# latest_df = (
#     df.sort_values("timestamp") 
#       .groupby("symbol")
#       .tail(1)
#     #   .sort_values("cnt_breakout_strength", ascending=False)
#       .copy()
# )

# def latest_event(row):
#     if row["is_breakout"] == 1:
#         return "BREAKOUT"
#     if row["is_breakdown"] == 1:
#         return "BREAKDOWN"
#     if row["is_ath"] == 1:
#         return "ATH"
#     if row["is_atl"] == 1:
#         return "ATL"
#     return None

# latest_df["event"] = latest_df.apply(latest_event, axis=1)

# event_df = latest_df[latest_df["event"].notna()].copy()

# # ================== BLINK / NEW EVENT LOGIC ==================
# event_df["event_key"] = event_df["symbol"] + "|" + event_df["event"]

# if "prev_events" not in st.session_state:
#     st.session_state.prev_events = set()

# event_df["is_new"] = ~event_df["event_key"].isin(st.session_state.prev_events)
# st.session_state.prev_events = set(event_df["event_key"])

# def highlight_new_event(row):
#     styles = []

#     alert = str(row.get("event", "")).upper()
#     is_new = row.get("is_new", False)

#     if "BREAKOUT" in alert:
#         base = "background-color:#e8f5e9; color:#1b5e20; font-weight:700" #green
#     elif "BREAKDOWN" in alert:
#         base = "background-color:#fdecea; color:#b71c1c; font-weight:700" #red
#     elif "ATL" in alert:
#         base = "background-color:#fdecea; color:#b71c1c; font-weight:700" #red
#     elif "ATH" in alert:
#         base = "background-color:#e8f5e9; color:#1b5e20; font-weight:700" #green
#     else:
#         base = ""

#     if is_new and base:
#         base += "; animation: blink 1.2s ease-in-out infinite"

#     return [base] * len(row)
#     return styles


# display_cols = [
#     "symbol",
#     "timestamp",
#     "close",
#     "event",
#     "cnt_breakout_strength",
#     "fo_buildup",
#     "money_change_%",
#     "daily_inflow_change_%",
#     "price_change_%"
#     #"volume"
# ]

# # Sort & select columns FIRST
# event_display_df = (
#     event_df.assign(
#         event_key=event_df["symbol"] + "|" + event_df["event"]
#     )
#     # .sort_values("timestamp", ascending=False)
#     .sort_values("money_change_%", ascending=False)
# )


# # Style AFTER final column selection
# styled_df = event_display_df[display_cols].style.apply(
#     highlight_new_event, axis=1
# )

# st.dataframe(
#     styled_df,
#     width="stretch",
#     height=600
# )

# ####====


# # ================== SMART MONEY TABLE ==================
# st.subheader("💰 Smart Money Activity")

# smart_money = df_selected[
#     df_selected["trigger_alert"].isin([
#         "FRESH_BREAKDOWN",
#         "BREAKOUT_MONEY_SURGE",
#         "FRESH_BREAKOUT",
#         "BREAKOUT+ATH",
#         "ATH",
#         "ATL",
#         # "MONEY_SPIKE_COVERING",
#         "BREAKDOWN+ATL",
#         "BREAKDOWN+UMM",
#         "BREAKOUT+ATH+UMM",
#         "BREAKDOWN+UMM"

#     ])
# ].sort_values("timestamp", ascending=False)

# st.dataframe(
#     smart_money[
#         [
#             "timestamp", "symbol", "close",
#             "trigger_alert", "cnt_breakout_strength", "money_change_%",
#             "daily_inflow_change_%", 
#             "fo_buildup", "flow_signal"
#         ]
#     ].style
#      .applymap(breakout_breakdown_style, subset=["trigger_alert"]),
#     width="stretch"
# )


# ================== BUILD EOD SMART MONEY VIEW ==================




# ================== BUILD HISTORICAL EOD SMART MONEY VIEW ==================
if not df_eod.empty:

    # Keep ONLY rows where EOD smart alert exists
    eod_view = df_eod[df_eod["has_smart_alert"] == True].copy()

    # Merge live confirmation + smart score from intraday df (latest per symbol)
    live_cols = [
    c for c in [
        "symbol",
        "smart_score",
        "eod_confirm",
        "money_change_%",
        "daily_inflow_change_%"
    ]
    if c in df.columns
    ]

    live_confirm = (
        df.sort_values("timestamp")
        .groupby("symbol")
        .tail(1)[live_cols]
    )


    eod_view = eod_view.merge(
        live_confirm,
        on="symbol",
        how="left"
    )

    # Sort by EOD date (latest first)
    eod_view = eod_view.sort_values("date", ascending=False)

else:
    eod_view = pd.DataFrame()

if selected_symbols:
    eod_view = eod_view[eod_view["symbol"].isin(selected_symbols)]





# ================== DAILY (EOD) FILTERS — FULL SAFE BLOCK ==================
st.sidebar.markdown("---")
st.sidebar.title("📅 Daily (EOD) Filters")

# ---- Symbol selection (Daily only) ----
daily_symbols = (
    sorted(eod_view["symbol"].unique())
    if not eod_view.empty and "symbol" in eod_view.columns
    else []
)

selected_daily_symbols = st.sidebar.multiselect(
    "Daily Symbols",
    daily_symbols,
    key="daily_symbols"
)

use_daily_filter = st.sidebar.checkbox(
    "Enable Daily Scan Filter",
    value=False
)

daily_smart_only = st.sidebar.checkbox(
    "Only Daily Smart Money"
)

min_ath_10d = st.sidebar.slider(
    "Min ATH Count (10D)",
    0, 10, 0
)

# ------------------------------------------------------------------
# GUARANTEE ath_count_10d EXISTS (NO KEYERROR EVER)
# ------------------------------------------------------------------
if "ath_count_10d" not in eod_view.columns:
    eod_view["ath_count_10d"] = 0
else:
    eod_view["ath_count_10d"] = (
        pd.to_numeric(eod_view["ath_count_10d"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

# ------------------------------------------------------------------
# MERGE LATEST ATH COUNT FROM df_eod (ONLY IF AVAILABLE)
# ------------------------------------------------------------------
if (
    not df_eod.empty
    and "ath_count_10d" in df_eod.columns
    and "symbol" in df_eod.columns
):
    latest_ath = (
        df_eod.sort_values("date")
              .groupby("symbol", as_index=False)
              .tail(1)[["symbol", "ath_count_10d"]]
    )

    eod_view = eod_view.merge(
        latest_ath,
        on="symbol",
        how="left",
        suffixes=("", "_new")
    )

    # Prefer merged value if present
    if "ath_count_10d_new" in eod_view.columns:
        eod_view["ath_count_10d"] = (
            eod_view["ath_count_10d_new"]
            .fillna(eod_view["ath_count_10d"])
            .astype(int)
        )
        eod_view.drop(columns=["ath_count_10d_new"], inplace=True)

# ------------------------------------------------------------------
# APPLY DAILY FILTERS
# ------------------------------------------------------------------
if selected_daily_symbols:
    eod_view = eod_view[
        eod_view["symbol"].isin(selected_daily_symbols)
    ]

if use_daily_filter:
    if daily_smart_only and "has_smart_alert" in eod_view.columns:
        eod_view = eod_view[eod_view["has_smart_alert"] == True]

    if min_ath_10d > 0:
        eod_view = eod_view[
            eod_view["ath_count_10d"] >= min_ath_10d
        ]

# ------------------------------------------------------------------
# OPTIONAL DEBUG (SAFE TO REMOVE LATER)
# ------------------------------------------------------------------
# st.sidebar.write("Daily rows:", len(eod_view))
# st.sidebar.write("Columns:", list(eod_view.columns))




# ================== SMART SCORE TABLE ==================

# smart_df = df[
#     df["trigger_alert"].isin([
#         "FRESH_BREAKDOWN",
#         "FRESH_BREAKOUT",
#         "BREAKOUT+ATH",
#         "ATH",
#         "MONEY_SPIKE_COVERING"
#     ])
# ].sort_values("timestamp", ascending=False)

st.subheader("🧠 Smart Money Activity (DAILY + Live Confirmation)")

final_cols = [
    c for c in [
        "date",
        "symbol",
        "close",
        "smart_money_alerts",
        "ath_count_10d",
        "smart_score",
        "money_change_%",
        "daily_inflow_change_%",
        "eod_confirm"
    ]
    if c in eod_view.columns
]

st.dataframe(eod_view[final_cols], width="stretch", height=500)


# ================== NORMALIZE FLAGS ==================
for col in ["is_breakout", "is_breakdown", "is_ath"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)



# ================== FULL DATA ==================
with st.expander("📂 Full Scanner Data (No Styling)"):
    st.dataframe(df, width="stretch")



# # =========================================================
# # 🔹 APPLY STRENGTH BUCKET (TODAY ONLY)
# # =========================================================
# if not df_today_latest.empty and "cnt_breakout_strength" in df_today_latest.columns:
#     df_today_latest["strength_bucket"] = (
#         df_today_latest["cnt_breakout_strength"]
#         .apply(strength_bucket)
#     )
# else:
#     df_today_latest["strength_bucket"] = "UNKNOWN"


# # =========================================================
# # 🔹 ALSO ADD BUCKET TO FULL DF (FOR SIDEBAR FILTER USE)
# # =========================================================
# if "cnt_breakout_strength" in df.columns:
#     df["strength_bucket"] = df["cnt_breakout_strength"].apply(strength_bucket)
# else:
#     df["strength_bucket"] = "UNKNOWN"


# # =========================================================
# # 🔹 TABS
# # =========================================================
# tab1, tab2, tab3, tab4 = st.tabs([
#     "📌 Latest Events",
#     "📊 Strength Buckets",
#     "💰 Smart Money",
#     "🧠 Daily + EOD"
# ])

# # =========================================================
# # 🔹 TAB 2 — STRENGTH BUCKETS (TODAY ONLY | EXPANDERS)
# # =========================================================
# with tab2:
#     st.subheader(f"📊 Breakout / Breakdown Strength (Today Only – {today})")

#     if df_today_latest.empty:
#         st.info("No strength data for today")
#     else:
#         # Ensure sorted order of buckets (Strong → Weak)
#         bucket_order = [
#             "🔥 +20 to +25 (Very Strong BO)",
#             "🟢 +10 to +19 (Strong BO)",
#             "🟡 +1 to +9 (Weak BO)",
#             "⚪ 0 (Neutral)",
#             "🟠 -1 to -9 (Weak BD)",
#             "🔴 -10 to -19 (Strong BD)",
#             "💀 ≤ -20 (Very Strong BD)"
#         ]

#         for bucket in bucket_order:
#             sub = df_today_latest[
#                 df_today_latest["strength_bucket"] == bucket
#             ].sort_values(
#                 "cnt_breakout_strength", ascending=False
#             )

#             if sub.empty:
#                 continue

#             with st.expander(
#                 f"{bucket}  •  {len(sub)} symbols",
#                 expanded=False
#             ):
#                 st.dataframe(
#                     sub[
#                         [
#                             "symbol",
#                             "timestamp",
#                             "close",
#                             "trigger_alert",
#                             "cnt_breakout_strength",
#                             "money_change_%",
#                             "daily_inflow_change_%",
#                             "fo_buildup"
#                         ]
#                     ],
#                     width="stretch",
#                     height=400
#                 )

# # =========================================================
# # 🔹 SIDEBAR — STRENGTH FILTER (OPTIONAL)
# # =========================================================
# st.sidebar.markdown("---")
# st.sidebar.title("📊 Strength Filter")

# strength_levels = sorted(
#     df["strength_bucket"].dropna().unique().tolist()
# )

# selected_strengths = st.sidebar.multiselect(
#     "Strength Buckets",
#     strength_levels
# )

# if selected_strengths:
#     df = df[df["strength_bucket"].isin(selected_strengths)]












