# ============================================================
# FO FUTURES — UNIFIED INTRADAY + SMART MONEY SCANNER
# ============================================================

import os
from zipfile import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import requests
from collections import deque
from pathlib import Path

# ===================== CONFIG =====================
API_KEY = "cj1a5xy951aule0t"
API_SECRET = "rtmfamrchkhalabya8m06smjs41i2o0y"
REQUEST_TOKEN = "Q3eiPHUp753GNf51At1fRZaX8cZK0ixd"

BOT_TOKEN = "6455237987:AAGwEBM552bZSwYIURSSOxJsh4kF3v9awLM"
CHAT_ID = "-1001984107299"

INTERVAL = "5minute"
DAYS = 5
BASE_DIR = Path(__file__).resolve().parent
FO_DIR=BASE_DIR / ""   
os.chdir(os.path.dirname(os.path.abspath(FO_DIR)))
SUPER_BUY_ALERTS = []
SUPER_SELL_ALERTS = []


# ===================== KITE INIT =====================
def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    token_file = "access_token.txt"

    if os.path.exists(token_file):
        kite.set_access_token(open(token_file).read().strip())
        try:
            kite.profile()
            return kite
        except:
            pass

    data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    kite.set_access_token(data["access_token"])
    open(token_file, "w").write(data["access_token"])
    return kite


def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("Telegram error:", e)


# ===================== UTILITIES =====================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_rsi_generic(series, period=14):
    """
    Generic RSI function
    Works for:
    - price
    - volume
    - money_flow
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_max_oi(df, window=10):
    df = df.copy()
    df["max_OI"] = df["oi"].rolling(window, min_periods=1).max()
    return df

def add_volume_spike(df):
    df = df.copy()
    df["vol_spike_%"] = df["volume"].pct_change() * 100
    return df

def add_anomaly_score(df, window=20):
    df = df.copy()

    vol_z = (df["volume"] - df["volume"].rolling(window).mean()) / \
            (df["volume"].rolling(window).std() + 1e-9)

    oi_z = (df["oi_change_%"] - df["oi_change_%"].rolling(window).mean()) / \
           (df["oi_change_%"].rolling(window).std() + 1e-9)

    df["anomaly_score"] = (0.6 * vol_z) + (0.4 * oi_z)
    return df

def add_supertrend_dir(df, period=10, multiplier=3.0):
    """
    Proper stateful Supertrend implementation
    Returns:
        supertrend
        supertrend_dir  (+1 bullish, -1 bearish)
        atr
    """
    df = df.copy()

    # --- ATR (your existing logic) ---
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()

    # --- Basic Bands ---
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * df["atr"]
    basic_lower = hl2 - multiplier * df["atr"]

    # --- Final Bands (stateful) ---
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df["close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df["close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    # --- Supertrend Direction ---
    supertrend = np.zeros(len(df))
    supertrend_dir = np.zeros(len(df))

    for i in range(1, len(df)):
        if supertrend_dir[i-1] == 1:
            if df["close"].iloc[i] < final_lower.iloc[i]:
                supertrend_dir[i] = -1
                supertrend[i] = final_upper.iloc[i]
            else:
                supertrend_dir[i] = 1
                supertrend[i] = final_lower.iloc[i]
        else:
            if df["close"].iloc[i] > final_upper.iloc[i]:
                supertrend_dir[i] = 1
                supertrend[i] = final_lower.iloc[i]
            else:
                supertrend_dir[i] = -1
                supertrend[i] = final_upper.iloc[i]

    df["supertrend"] = supertrend
    df["supertrend_dir"] = supertrend_dir.astype(int)

    return df


# def add_supertrend_dir(df, period=10, multiplier=3.0):
#     df = df.copy()

#     # --- ATR ---
#     prev_close = df["close"].shift(1)
#     tr = pd.concat([
#         df["high"] - df["low"],
#         (df["high"] - prev_close).abs(),
#         (df["low"] - prev_close).abs()
#     ], axis=1).max(axis=1)

#     df["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()

#     # --- Basic Bands ---
#     hl2 = (df["high"] + df["low"]) / 2
#     basic_upper = hl2 + multiplier * df["atr"]
#     basic_lower = hl2 - multiplier * df["atr"]

#     # --- Final Bands ---
#     final_upper = basic_upper.copy()
#     final_lower = basic_lower.copy()

#     for i in range(1, len(df)):
#         if (
#             basic_upper.iloc[i] < final_upper.iloc[i-1]
#             or df["close"].iloc[i-1] > final_upper.iloc[i-1]
#         ):
#             final_upper.iloc[i] = basic_upper.iloc[i]
#         else:
#             final_upper.iloc[i] = final_upper.iloc[i-1]

#         if (
#             basic_lower.iloc[i] > final_lower.iloc[i-1]
#             or df["close"].iloc[i-1] < final_lower.iloc[i-1]
#         ):
#             final_lower.iloc[i] = basic_lower.iloc[i]
#         else:
#             final_lower.iloc[i] = final_lower.iloc[i-1]

#     # --- Direction (+1 / -1 / 0) ---
#     supertrend_dir = np.zeros(len(df), dtype=int)
#     supertrend = np.full(len(df), np.nan)

#     for i in range(1, len(df)):
#         close = df["close"].iloc[i]

#         # Bullish breakout
#         if close > final_upper.iloc[i]:
#             supertrend_dir[i] = 1
#             supertrend[i] = final_lower.iloc[i]

#         # Bearish breakdown
#         elif close < final_lower.iloc[i]:
#             supertrend_dir[i] = -1
#             supertrend[i] = final_upper.iloc[i]

#         # Inside bands → NEUTRAL
#         else:
#             supertrend_dir[i] = 0
#             supertrend[i] = supertrend[i-1]

#     df["supertrend"] = supertrend
#     df["supertrend_dir"] = supertrend_dir

#     return df

def add_composite_trend_dir(
    df,
    smart_score_col="smart_score",
    hlc_col="HLC_TREND_DIR",
    st_col="supertrend_dir",
    smart_threshold=60
):
    """
    Composite Trend Direction (CTD)

    Combines:
    - HLC_TREND_DIR  → structure
    - supertrend_dir → volatility regime
    - smart_score    → institutional intent

    Output:
        COMPOSITE_TREND_DIR
            +1 → Strong Bullish Alignment
            -1 → Strong Bearish Alignment
             0 → Neutral / No Consensus
    """

    df = df.copy()

    # ---------------------------
    # Safety: ensure columns exist
    # ---------------------------
    for col in [smart_score_col, hlc_col, st_col]:
        if col not in df.columns:
            df[col] = 0

    # ---------------------------
    # Smart Money Intent
    # ---------------------------
    smart_bull = df[smart_score_col] >= smart_threshold
    smart_bear = df[smart_score_col] <= -smart_threshold

    # ---------------------------
    # Structure + Volatility alignment
    # ---------------------------
    struct_vol_bull = (
        (df[hlc_col] == 1) &
        (df[st_col] == 1)
    )

    struct_vol_bear = (
        (df[hlc_col] == -1) &
        (df[st_col] == -1)
    )

    # ---------------------------
    # Final Composite Logic
    # ---------------------------
    conditions = [
        smart_bull & struct_vol_bull,
        smart_bear & struct_vol_bear
    ]

    choices = [1, -1]

    df["COMPOSITE_TREND_DIR"] = np.select(
        conditions,
        choices,
        default=0
    )

    return df



def add_composite_trend_strength(
    df,
    smart_score_col="smart_score",
    hlc_col="HLC_TREND_DIR",
    st_col="supertrend_dir"
):
    """
    Returns COMPOSITE_TREND_STRENGTH (0–3)
    """

    df = df.copy()

    score = np.zeros(len(df))

    score += (df[hlc_col] == 1).astype(int)
    score += (df[st_col] == 1).astype(int)
    score += (df[smart_score_col] >= 60).astype(int)

    score -= (df[hlc_col] == -1).astype(int)
    score -= (df[st_col] == -1).astype(int)
    score -= (df[smart_score_col] <= -60).astype(int)

    df["COMPOSITE_TREND_STRENGTH"] = score

    return df


def add_money_st_dir(df):
    df = df.copy()

    df["money_st_dir"] = np.select(
        [
            (df["money_change_cal"] > 200) & (df["supertrend_dir"] == 1),
            (df["money_change_cal"] > 200) & (df["supertrend_dir"] == -1)
        ],
        [1, -1],
        default=0
    )
    return df

def add_smart_money_flags(df):
    df = df.copy()

    # SMB Signal
    df["SMB_Signal"] = (
        (df["is_breakout"]) &
        (df["IMA_Score"] > 1.0) &
        (df["money_flow"] > df["money_flow"].rolling(10).mean())
    ).astype(int)

    # Accumulation → Breakout
    acc_phase = (
        (df["IMA_Score"] > 0) &
        (df["money_flow"] > df["money_flow"].rolling(5).mean())
    ).rolling(8).sum() >= 6

    df["A2B_Breakout"] = (acc_phase & df["is_breakout"]).astype(int)

    # IMA Surge
    ima_z = (df["IMA_Score"] - df["IMA_Score"].rolling(10).mean()) / \
            (df["IMA_Score"].rolling(10).std() + 1e-9)

    df["IMA_Surge_Flag"] = (ima_z > 2).astype(int)

    return df

def add_rolling_alert_signals(df, window=5):
    """
    Rolling alert pressure logic (Script-1 compatible)
    """
    df = df.copy()

    # rolling signed pressure
    df["rolling_alert_score_5"] = (
        df
        .groupby("symbol")["alert_score"]
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # convert score → regime
    df["rolling_alert_signal"] = np.where(
        df["rolling_alert_score_5"] > 2,  1,
        np.where(df["rolling_alert_score_5"] < -2, -1, 0)
    )

    # previous regime (state tracking)
    df["rolling_alert_signal_old"] = df["rolling_alert_signal"].shift(1)

    return df


def add_dark_pool_activity(df):
    df = df.copy()

    df["DPA_Index"] = (
        0.4 * df["anomaly_score"].fillna(0) +
        0.3 * df["SM_Div"].fillna(0) +
        0.2 * (1 / (df["high"] - df["low"]).rolling(5).mean().replace(0, np.nan)) +
        0.1 * (df["money_flow"] / df["money_flow"].rolling(20).mean())
    )

    df["DarkPool_Flag"] = (
        (df["DPA_Index"] > 2) &
        (df["DPA_Index"].shift(1) <= 2)
    ).astype(int)

    return df



def compute_inflow_rank_snapshot(df):
    df = df.copy()

    cols = [
        "daily_inflow",
        "money_change_%",
        "daily_inflow_change_%"
    ]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["inflow_norm"] = df["daily_inflow"] / (df["daily_inflow"].abs().max() + 1e-9)

    df["inflow_rank"] = (
        0.5 * df["money_change_%"].rank(pct=True) +
        0.3 * df["daily_inflow_change_%"].rank(pct=True) +
        0.2 * df["inflow_norm"].rank(pct=True)
    )

    df["inflow_rank_top5"] = df["inflow_rank"].rank(ascending=False) <= 5
    return df


def add_inflow_dynamics(df):
    """
    Replicates Script-1 intraday inflow logic
    """
    df = df.copy()

    # % spike candle-to-candle
    df["inflow_spike_%"] = (
        df["money_flow"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0) * 100
    )

    # velocity of cumulative inflow
    df["inflow_velocity"] = df["daily_inflow"].diff()

    # 3-candle burst inflow
    df["inflow_3c_sum"] = df["money_flow"].rolling(3).sum()
    df["inflow_3c_spike_%"] = (
        df["inflow_3c_sum"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0) * 100
    )

    # normalized inflow (for ranking)
    df["inflow_norm"] = (
        df["daily_inflow"] /
        (df["daily_inflow"].abs().max() + 1e-9)
    )

    return df



# ===================== FETCH DATA =====================
def get_futures(kite):
    df = pd.DataFrame(kite.instruments("NFO"))
    df = df[df.instrument_type == "FUT"]
    expiry = df.expiry.dropna().sort_values().iloc[0]
    return df[df.expiry == expiry][["tradingsymbol", "instrument_token"]]

def fetch_intraday(kite, token):
    to_dt = datetime.now() - timedelta(minutes=1)
    from_dt = to_dt - timedelta(days=DAYS)
    data = kite.historical_data(token, from_dt, to_dt, INTERVAL, oi=True)
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


FINAL_SCHEMA = [
    # ---- Time / Identity ----
    "timestamp",
    "symbol",

    # ---- Price / Volume / OI ----
    "close",
    "volume",
    "OI",
    # ---- FO Structure / OI ----
    "max_OI",
    "oi_drop",
    "oi_drop_%",
    "oi_change_%",
    "price_change_%",

    # ---- Symbol mapping ----
    "symbol1",

    # ---- Money Flow ----
    "money_flow",
    "money_change_%",
    "money_change1_%",
    "money_change_cal",
    "money_change_spike",

    # ---- Inflow Engine ----
    "daily_inflow",
    "daily_inflow_change_%",
    "inflow_spike_%",
    "inflow_velocity",
    "inflow_3c_spike_%",
    "inflow_norm",
    "inflow_rank",
    "inflow_rank_top5",

    # ---- FO Structure ----
    "fo_buildup",
    "fo_buildup_old",

    # ---- Smart Money ----
    "UMM_score",
    "MFT_20",
    "SM_Div",
    "IMA_Score",
    "WMF",

    # ---- VWAP / Flow ----
    "vwap_dev_%",
    "flow_signal",
    "vol_spike_%",

    # ---- Market Structure ----
    "is_breakout",
    "is_ath",
    "ath_count_10d",
    "ath_recent_flag",
    "is_breakdown",
    "is_atl",
    "atl_count_10d",
    "atl_recent_flag",

    # ---- Alerts ----
    "trigger_alert",
    "alert_score",
    "rolling_alert_score_5",
    "rolling_alert_signal",
    "rolling_alert_signal_old",
    "cnt_breakout_strength",

    # ---- Trade Signals ----
    "HLC_TREND_DIR",
    "HLC_TREND_STRENGTH",
    "SUPER_TRADE_FLAG",
    "SUPER_TRADE_SIGNAL",
    "superbuy_count",
    "supersell_count",
    "SUPER_TRADE_FIRST",
    "HOURLY_SUPER_TRADE_SIGNAL",
    "hourly_superbuy_count",
    "hourly_supersell_count",

    "BUY_SIGNAL",
    "SELL_SIGNAL",
    "SIGNAL_REASON",

    # ---- Technicals ----
    "anomaly_score",
    "ema_fast",
    "ema_slow",
    "supertrend_dir",
    "money_st_dir",
    #"COMPOSITE_TREND_DIR",
    "COMPOSITE_TREND_STRENGTH",

    "rsi14",
    "rsi_volume",
    "R_factor",
    "BIG_CANDLE_DIR",
    "SHORT_COVERING_OI",
    "LONG_UNWINDING_OI",
    "INSTITUTIONAL_EXIT",

    # ---- Advanced Smart Money ----
    "SMB_Signal",
    "A2B_Breakout",
    "IMA_Surge_Flag",
    "DPA_Index",
    "DarkPool_Flag",
]

def enforce_schema(
    df: pd.DataFrame,
    schema: list,
    drop_extra: bool = False,
    fill_value=np.nan
) -> pd.DataFrame:
    """
    Enforces:
    - exact column order
    - adds missing columns
    - optionally drops unknown columns
    """

    df = df.copy()

    # 1️⃣ Add missing columns
    for col in schema:
        if col not in df.columns:
            df[col] = fill_value

    # 2️⃣ Drop extra columns (optional, SAFE default = False)
    if drop_extra:
        df = df[[c for c in schema if c in df.columns]]

    # 3️⃣ Reorder columns
    df = df[schema]

    return df

def add_oi_drop(df, window=10):
    df = df.copy()

    # safety
    if "max_OI" not in df.columns or "OI" not in df.columns:
        df["oi_drop"] = 0.0
        df["oi_drop_%"] = 0.0
        return df

    df["oi_drop"] = df["max_OI"] - df["OI"]
    df["oi_drop_%"] = (
        df["oi_drop"] / (df["max_OI"] + 1e-9)
    ) * 100

    return df

def send_consolidated_super_trade_alert():
    if not SUPER_BUY_ALERTS and not SUPER_SELL_ALERTS:
        return

    msg = "🚨 <b>FnO TRADE ALERTS</b> (5-min)\n\n"

    if SUPER_BUY_ALERTS:
        msg += f"🟢 <b>Buy ({len(SUPER_BUY_ALERTS)})</b>\n"
        for s in SUPER_BUY_ALERTS:
            msg += f"• {s}\n"
        msg += "\n"

    if SUPER_SELL_ALERTS:
        msg += f"🔴 <b>Sell ({len(SUPER_SELL_ALERTS)})</b>\n"
        for s in SUPER_SELL_ALERTS:
            msg += f"• {s}\n"
        msg += "\n"

    msg += f"⏱ Time: {datetime.now().strftime('%H:%M')}"

    #send_telegram_alert(msg)

def send_daily_first_supertrade_alerts(final_df):
    """
    Sends Telegram alerts ONLY for:
    - SuperBuy with superbuy_count == 1
    - SuperSell with supersell_count == 1
    """

    if final_df.empty:
        return

    # take only latest candle per symbol
    latest = (
        final_df
        .sort_values("timestamp")
        .groupby("symbol")
        .tail(1)
    )

    buys = latest[
       ( (latest["SUPER_TRADE_SIGNAL"] == "SuperBuy") | (latest["cnt_breakout_strength"] > 3)) & 
        # (latest["HOURLY_SUPER_TRADE_SIGNAL"] == "SuperBuy") &
        (latest["superbuy_count"] > 1) &
        (latest["HLC_TREND_STRENGTH"] > 1)
    ]

    sells = latest[
        ((latest["SUPER_TRADE_SIGNAL"] == "SuperSell") | (latest["cnt_breakout_strength"] < -3))  &
        # (latest["HOURLY_SUPER_TRADE_SIGNAL"] == "SuperSell") &
        (latest["supersell_count"] > 1) &
        (latest["HLC_TREND_STRENGTH"] < -1)
    ]

    if buys.empty and sells.empty:
        return

    msg = "🚨 <b>FnO FIRST SUPER TRADE ALERTS</b>\n\n"

    if not buys.empty:
        msg += f"🟢 <b>SuperBuy ({len(buys)})</b>\n"
        for _, r in buys.iterrows():
            sym = r["symbol"][:-8]
            msg += f"• {sym} @ {round(r['close'], 2)}\n"
        msg += "\n"

    if not sells.empty:
        msg += f"🔴 <b>SuperSell ({len(sells)})</b>\n"
        for _, r in sells.iterrows():
            sym = r["symbol"][:-8]
            msg += f"• {sym} @ {round(r['close'], 2)}\n"
        msg += "\n"

    msg += f"⏱ Time: {datetime.now().strftime('%H:%M')}"

    send_telegram_alert(msg)





def add_super_trade_signal(df):
    """
    SuperBuy / SuperSell based on:
    - rolling_alert_score_5
    - oi_drop
    """
    df = df.copy()

    # numeric flag
    df["SUPER_TRADE_FLAG"] = np.select(
        [
            (df["rolling_alert_score_5"] >= 2) & (df["oi_drop"] > 1),
            (df["rolling_alert_score_5"] <= -2) & (df["oi_drop"] > 1),
        ],
        [1, -1],
        default=0
    )

    # readable label
    df["SUPER_TRADE_SIGNAL"] = np.select(
        [
            df["SUPER_TRADE_FLAG"] == 1,
            df["SUPER_TRADE_FLAG"] == -1,
        ],
        ["SuperBuy", "SuperSell"],
        default=""
    )

    # ---- SuperTrade FIRST trigger (0 → ±1) ----
    df["SUPER_TRADE_FIRST"] = (
        (df["SUPER_TRADE_FLAG"] != 0) &
        (df["SUPER_TRADE_FLAG"].shift(1).fillna(0) == 0)
    ).astype(int)


    return df

# def add_supertrade_count(df, inactivity_sessions=10):
#     df = df.copy()

#     # --- ensure ordering ---
#     df = df.sort_values("timestamp")

#     # --- trade date (daily reset) ---
#     df["trade_date"] = pd.to_datetime(df["timestamp"]).dt.date

#     # ===============================
#     # SuperBuy logic
#     # ===============================
#     is_buy = (df["SUPER_TRADE_SIGNAL"] == "SuperBuy").astype(int)

#     # index of last SuperBuy per symbol+day
#     last_buy_idx = (
#         is_buy
#         .groupby([df["symbol"], df["trade_date"]])
#         .apply(lambda x: x.where(x == 1).index)
#     )

#     # distance from last SuperBuy (in candles)
#     df["buy_gap"] = (
#         is_buy
#         .groupby([df["symbol"], df["trade_date"]])
#         .apply(lambda x: x.groupby((x == 1).cumsum()).cumcount())
#         .reset_index(level=[0,1], drop=True)
#     )

#     # running count
#     df["superbuy_count"] = (
#         is_buy
#         .groupby([df["symbol"], df["trade_date"]])
#         .cumsum()
#     )

#     # reset if inactivity exceeded
#     df.loc[df["buy_gap"] > inactivity_sessions, "superbuy_count"] = 0

#     # ===============================
#     # SuperSell logic
#     # ===============================
#     is_sell = (df["SUPER_TRADE_SIGNAL"] == "SuperSell").astype(int)

#     df["sell_gap"] = (
#         is_sell
#         .groupby([df["symbol"], df["trade_date"]])
#         .apply(lambda x: x.groupby((x == 1).cumsum()).cumcount())
#         .reset_index(level=[0,1], drop=True)
#     )

#     df["supersell_count"] = (
#         is_sell
#         .groupby([df["symbol"], df["trade_date"]])
#         .cumsum()
#     )

#     df.loc[df["sell_gap"] > inactivity_sessions, "supersell_count"] = 0

#     # --- cleanup ---
#     df["superbuy_count"] = df["superbuy_count"].fillna(0).astype(int)
#     df["supersell_count"] = df["supersell_count"].fillna(0).astype(int)

#     df.drop(columns=["buy_gap", "sell_gap"], inplace=True, errors="ignore")

#     return df


# def add_supertrade_count(df, inactivity_sessions=5):
#     df = df.copy()
#     df = df.sort_values(["symbol", "timestamp"])

#     df["superbuy_count"] = 0   
#     df["supersell_count"] = 0

#     for symbol, g in df.groupby("symbol"):
#         buy_count = 0
#         sell_count = 0
#         buy_inactive = 0
#         sell_inactive = 0

#         for idx in g.index:
#             signal = df.at[idx, "SUPER_TRADE_SIGNAL"]

#             # ===============================
#             # SuperBuy logic
#             # ===============================
#             if signal == "SuperBuy":
#                 if buy_inactive >= inactivity_sessions:
#                     buy_count = 1
#                 else:
#                     buy_count += 1
#                 buy_inactive = 0
#             else:
#                 buy_inactive += 1
#                 if buy_inactive >= inactivity_sessions:
#                     buy_count = 0

#             # ===============================
#             # SuperSell logic
#             # ===============================
#             if signal == "SuperSell":
#                 if sell_inactive >= inactivity_sessions:
#                     sell_count = 1
#                 else:
#                     sell_count += 1
#                 sell_inactive = 0
#             else:
#                 sell_inactive += 1
#                 if sell_inactive >= inactivity_sessions:
#                     sell_count = 0

#             df.at[idx, "superbuy_count"] = buy_count
#             df.at[idx, "supersell_count"] = sell_count

#     return df

def add_supertrade_count(df, inactivity_sessions=5):
    df = df.copy()

    # ensure datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # day key (daily reset)
    df["trade_date"] = df["timestamp"].dt.date

    df = df.sort_values(["symbol", "timestamp"])

    df["superbuy_count"] = 0
    df["supersell_count"] = 0

    for (symbol, trade_date), g in df.groupby(["symbol", "trade_date"]):
        buy_count = 0
        sell_count = 0
        buy_inactive = 0
        sell_inactive = 0

        for idx in g.index:
            signal = df.at[idx, "SUPER_TRADE_SIGNAL"]

            # ===============================
            # SuperBuy
            # ===============================
            if signal == "SuperBuy":
                buy_count = 1 if buy_inactive >= inactivity_sessions else buy_count + 1
                buy_inactive = 0
            else:
                buy_inactive += 1
                if buy_inactive >= inactivity_sessions:
                    buy_count = 0

            # ===============================
            # SuperSell
            # ===============================
            if signal == "SuperSell":
                sell_count = 1 if sell_inactive >= inactivity_sessions else sell_count + 1
                sell_inactive = 0
            else:
                sell_inactive += 1
                if sell_inactive >= inactivity_sessions:
                    sell_count = 0

            df.at[idx, "superbuy_count"] = buy_count
            df.at[idx, "supersell_count"] = sell_count

    return df



def add_hourly_supertrade_from_5min(df):
    """
    Creates Hourly SuperBuy / SuperSell using 5-min SuperTrade signals
    and maps it back to each 5-min row
    """

    df = df.copy()

    # ensure datetime
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

    # hourly bucket
    df["hour_bucket"] = df["timestamp_dt"].dt.floor("1h")

    # convert signals to numeric
    df["sb_flag"] = (df["SUPER_TRADE_SIGNAL"] == "SuperBuy").astype(int)
    df["ss_flag"] = (df["SUPER_TRADE_SIGNAL"] == "SuperSell").astype(int)

    # aggregate at hourly level
    hourly = (
        df
        .groupby(["symbol", "hour_bucket"])
        .agg(
            hour_sb=("sb_flag", "sum"),
            hour_ss=("ss_flag", "sum")
        )
        .reset_index()
    )

    # derive hourly signal
    hourly["HOURLY_SUPER_TRADE_SIGNAL"] = np.select(
        [
            hourly["hour_sb"] > 0,
            hourly["hour_ss"] > 0
        ],
        ["SuperBuy", "SuperSell"],
        default=""
    )

    # merge back to 5-min df
    df = df.merge(
        hourly[["symbol", "hour_bucket", "HOURLY_SUPER_TRADE_SIGNAL"]],
        on=["symbol", "hour_bucket"],
        how="left"
    )

    # cleanup
    df.drop(
        columns=["sb_flag", "ss_flag", "timestamp_dt"],
        inplace=True,
        errors="ignore"
    )

    return df


def add_hourly_supertrade_counts_safe(
    df,
    inactivity_hours=2
):
    """
    Hourly counts using existing HOURLY_SUPER_TRADE_SIGNAL
    Session reset = daily
    No merge → no blanks
    """

    df = df.copy()

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df["hour_bucket"] = df["timestamp_dt"].dt.floor("1h")
    df["trade_date"] = df["timestamp_dt"].dt.date

    df["hourly_superbuy_count"] = 0
    df["hourly_supersell_count"] = 0

    for (symbol, trade_date), g in (
        df.sort_values("timestamp_dt")
          .groupby(["symbol", "trade_date"])
    ):
        buy_count = sell_count = 0
        buy_idle = sell_idle = 0
        last_hour = None

        for idx in g.index:
            cur_hour = df.at[idx, "hour_bucket"]

            # new hour
            if last_hour != cur_hour:
                sig = df.at[idx, "HOURLY_SUPER_TRADE_SIGNAL"]

                # BUY
                if sig == "SuperBuy":
                    buy_count += 1
                    buy_idle = 0
                else:
                    buy_idle += 1
                    if buy_idle >= inactivity_hours:
                        buy_count = 0

                # SELL
                if sig == "SuperSell":
                    sell_count += 1
                    sell_idle = 0
                else:
                    sell_idle += 1
                    if sell_idle >= inactivity_hours:
                        sell_count = 0

                last_hour = cur_hour

            df.at[idx, "hourly_superbuy_count"] = buy_count
            df.at[idx, "hourly_supersell_count"] = sell_count

    return df


# ================= ALERT DEFINITIONS =================

POSITIVE_ALERTS = {
    "FRESH_BREAKOUT",
    "BREAKOUT_MONEY_SURGE",
    "STACKED_BREAKOUT_MONEY_ATH",
    "BREAKOUT+UMM",
    "BREAKOUT+ATH",
    "SMART_MONEY_BREAKOUT",
    "ATH"
}

NEGATIVE_ALERTS = {
    "FRESH_BREAKDOWN",
    "BREAKDOWN_MONEY_SURGE",
    "STACKED_BREAKDOWN_MONEY_ATL",
    "BREAKDOWN+UMM",
    "BREAKDOWN+ATL",
    "BREAKDOWN+ATL+UMM",
    "ATL"
}

ALERT_SCORE_MAP = {
    # Bullish
    "FRESH_BREAKOUT": 1,
    "BREAKOUT_MONEY_SURGE": 2,
    "BREAKOUT+UMM": 2,
    "BREAKOUT+ATH": 2,
    "SMART_MONEY_BREAKOUT": 2,
    "ATH": 1,
    "STACKED_BREAKOUT_MONEY_ATH": 3,

    # Bearish
    "FRESH_BREAKDOWN": -1,
    "BREAKDOWN_MONEY_SURGE": -2,
    "BREAKDOWN+UMM": -2,
    "BREAKDOWN+ATL": -2,
    "BREAKDOWN+ATL+UMM": -3,
    "ATL": -1,
    "STACKED_BREAKDOWN_MONEY_ATL": -3,
}


STACK_WINDOW_HOURS = 6
alert_stack_memory = {} 


def update_and_check_alert_stack(symbol, alert_reason, ts):
    if not alert_reason:
        return None

    if symbol not in alert_stack_memory:
        alert_stack_memory[symbol] = deque()

    dq = alert_stack_memory[symbol]

    dq.append((ts, alert_reason))

    cutoff = ts - timedelta(hours=STACK_WINDOW_HOURS)
    while dq and dq[0][0] < cutoff:
        dq.popleft()

    reasons = {r for _, r in dq}

    # 🔼 Bullish stack
    if (
        "BREAKOUT_MONEY_SURGE" in reasons and ("is_breakdown" == "True") and
        ("ATH" in reasons or "BREAKOUT+ATH" in reasons)
    ):
        return "STACKED_BREAKOUT_MONEY_ATH"

    # 🔽 Bearish stack
    if (
        "BREAKDOWN_MONEY_SURGE" in reasons and ("is_breakout" == "True") and
        ("ATL" in reasons or "BREAKDOWN+ATL" in reasons)
    ):
        return "STACKED_BREAKDOWN_MONEY_ATL"

    return None



def add_hlc_trend_dir(df):
    """
    Adds:
    HLC_TREND_DIR
        +1 → higher high, higher low, higher close
        -1 → lower high, lower low, lower close
         0 → mixed / neutral
    """

    df = df.copy()

    up = (
        (df["high"]  > df["high"].shift(1)) &
        (df["low"]   > df["low"].shift(1)) #&
        # (df["close"] > df["close"].shift(1))
    )

    down = (
        (df["high"]  < df["high"].shift(1)) &
        (df["low"]   < df["low"].shift(1)) 
        #&
        # (df["close"] < df["close"].shift(1))
    )

    df["HLC_TREND_DIR"] = np.select(
        [up, down],
        [1, -1],
        default=0
    )

    return df


# def add_hlc_trend_dir(df, lookback=2):
#     df = df.copy()

#     hh = df["high"] > df["high"].shift(1)
#     hl = df["low"]  > df["low"].shift(1)
#     #hc = df["close"] > df["close"].shift(1)

#     ll = df["high"] < df["high"].shift(1)
#     lh = df["low"]  < df["low"].shift(1)
#     #lc = df["close"] < df["close"].shift(1)

#     up = (hh & hl ).rolling(lookback).sum() == lookback
#     down = (ll & lh).rolling(lookback).sum() == lookback

#     df["HLC_TREND_DIR"] = np.select(
#         [up, down],
#         [1, -1],
#         default=0
#     )

#     return df


def compute_index_market_regime_log(
    df_latest: pd.DataFrame,
    index_name: str,
    weight_map: dict,
    default_weight: float,
    trend_col: str = "HLC_TREND_STRENGTH",
    symbol_col: str = "symbol1"
):
    """
    Computes weighted index trend distribution and returns
    a single consolidated log string.

    Output example:
    NIFTY50 | STRONG BULL | +67.4% / -18.1% / 14.5% | Breadth +49.3
    """

    pos = neg = neu = 0.0

    # -----------------------------
    # Aggregate weighted signals
    # -----------------------------
    for stock, wt in weight_map.items():
        row = df_latest[df_latest[symbol_col] == stock]

        if row.empty:
            st = 0
            wt = default_weight
        else:
            st = int(row.iloc[-1][trend_col])

        if st >= 1:
            pos += wt
        elif st <= -1:
            neg += wt
        else:
            neu += wt

    total = pos + neg + neu

    if total == 0:
        return f"{index_name} | MIXED | 0.0 / 0.0 / 100.0"

    pos_pct = round((pos / total) * 100, 1)
    neg_pct = round((neg / total) * 100, 1)
    neu_pct = round((neu / total) * 100, 1)

    # -----------------------------
    # Market Bias Logic
    # -----------------------------
    if pos_pct >= 60 and neg_pct <= 20:
        bias = "STRONG BULLISH"
    elif neg_pct >= 60 and pos_pct <= 20:
        bias = "STRONG BEARISH"
    # elif (pos_pct - neg_pct) >= 25:
    #     bias = "BULLISH"
    # elif (neg_pct - pos_pct) >= 25:
    #     bias = "BEARISH"
    else:
        bias = "MIXED"

    breadth = round(pos_pct - neg_pct, 1)

    # -----------------------------
    # Final Unified Log
    # -----------------------------
    log = (
        f"{index_name} | {bias} | "
        f"+{pos_pct}% / -{neg_pct}% / {neu_pct}% | "
        f"Breadth {breadth:+}"
    )

    return log

NIFTY50_WEIGHTS = { "HDFCBANK": 0.11, "RELIANCE": 0.10, "ICICIBANK": 0.07, "INFY": 0.06, "TCS": 0.04, "ITC": 0.04, "BHARTIARTL": 0.04, "KOTAKBANK": 0.03, "SBIN": 0.03, "HINDUNILVR": 0.03, }
DEFAULT_NIFTY_WT = 0.005
SENSEX_WEIGHTS = { "RELIANCE": 0.12, "HDFCBANK": 0.10, "ICICIBANK": 0.07, "INFY": 0.06, "TCS": 0.05, "HINDUNILVR": 0.04 }
DEFAULT_SENSEX_WT = 0.008
BANKNIFTY_WEIGHTS = { "HDFCBANK": 0.28, "ICICIBANK": 0.23, "KOTAKBANK": 0.13, "AXISBANK": 0.12, "SBIN": 0.12, "INDUSINDBK": 0.06, "BANKBARODA": 0.03, "FEDERALBNK": 0.02, "PNB": 0.02, "IDFCFIRSTB": 0.015, "BANDHANBNK": 0.01, }
DEFAULT_BANKNIFTY_WT = 0.005


def send_index_regime_telegram_alert(
    df_latest: pd.DataFrame,
    index_name: str,
    weight_map: dict,
    default_weight: float,
    threshold: float = 60.0,
    trend_col: str = "HLC_TREND_STRENGTH",
    symbol_col: str = "symbol1"
):
    """
    Sends Telegram alert if weighted POS or NEG >= threshold.
    """

    pos = neg = neu = 0.0

    # -----------------------------
    # Aggregate weighted signals
    # -----------------------------
    for stock, wt in weight_map.items():
        row = df_latest[df_latest[symbol_col] == stock]

        if row.empty:
            st = 0
            wt = default_weight
        else:
            st = int(row.iloc[-1][trend_col])

        if st >= 1:
            pos += wt
        elif st <= -1:
            neg += wt
        else:
            neu += wt

    total = pos + neg + neu
    if total == 0:
        return

    pos_pct = round((pos / total) * 100, 1)
    neg_pct = round((neg / total) * 100, 1)
    neu_pct = round((neu / total) * 100, 1)

    # -----------------------------
    # Alert Condition
    # -----------------------------
    if pos_pct < threshold and neg_pct < threshold:
        return  # no alert → mixed regime

    # -----------------------------
    # Bias Label
    # -----------------------------
    if pos_pct >= threshold:
        bias = "🟢 STRONG BULL"
        strength = f"+{pos_pct}%"
    else:
        bias = "🔴 STRONG BEARISH"
        strength = f"-{neg_pct}%"

    breadth = round(pos_pct - neg_pct, 1)

    # -----------------------------
    # Telegram Message
    # -----------------------------
    msg = (
        f"📊 <b>{index_name} INDEX REGIME ALERT</b>\n\n"
        f"{bias}\n"
        f"POS / NEG / NEU → "
        f"{pos_pct}% / {neg_pct}% / {neu_pct}%\n"
        f"Market Breadth → {breadth:+}\n"
        f"⏱ {datetime.now().strftime('%H:%M')}"
    )

    send_telegram_alert(msg)


def send_nifty50_regime_alert(df_latest):
    send_index_regime_telegram_alert(
        df_latest=df_latest,
        index_name="NIFTY50",
        weight_map=NIFTY50_WEIGHTS,
        default_weight=DEFAULT_NIFTY_WT
    )


def send_banknifty_regime_alert(df_latest):
    send_index_regime_telegram_alert(
        df_latest=df_latest,
        index_name="BANKNIFTY",
        weight_map=BANKNIFTY_WEIGHTS,
        default_weight=DEFAULT_BANKNIFTY_WT
    )


def send_sensex_regime_alert(df_latest):
    send_index_regime_telegram_alert(
        df_latest=df_latest,
        index_name="SENSEX",
        weight_map=SENSEX_WEIGHTS,
        default_weight=DEFAULT_SENSEX_WT
    )


def compute_r_factor(entry_price, exit_price, stop_price, direction):
    """
    direction: 1 for long, -1 for short
    """
    risk = abs(entry_price - stop_price)

    if risk == 0:
        return 0.0

    reward = (exit_price - entry_price) * direction
    return reward / risk

def add_volume_inflow_signal(
    df,
    mf_col="money_flow",
    price_col="close",
    lookback=20,
    spike_mult=1.5
):
    df = df.copy()

    mf_avg = df[mf_col].rolling(lookback).mean()

    df["INFLOW_TRADE_FLAG"] = np.select(
        [
            (df[mf_col] > mf_avg * spike_mult) &
            (df[price_col] > df[price_col].shift(1)),  # BUY

            (df[mf_col] > mf_avg * spike_mult) &
            (df[price_col] < df[price_col].shift(1))   # SELL
        ],
        [1, -1],
        default=0
    )

    # first entry only
    df["INFLOW_TRADE_FIRST"] = (
        (df["INFLOW_TRADE_FLAG"] != 0) &
        (df["INFLOW_TRADE_FLAG"].shift(1).fillna(0) == 0)
    ).astype(int)

    return df
def add_r_factor(
    df,
    entry_flag_col="SUPER_TRADE_FIRST",
    signal_col="SUPER_TRADE_FLAG",
    price_col="close",
    atr_col="atr",
    atr_mult=1.5
):
    """
    Adds:
    - entry_price
    - stop_price
    - R_factor
    - max_R (per trade)
    """

    df = df.copy()

    df["entry_price"] = np.nan
    df["stop_price"] = np.nan
    df["R_factor"] = 0.0
    df["max_R"] = 0.0

    entry_price = stop_price = None
    direction = 0
    max_r = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # 🔑 New Trade Entry
        if row[entry_flag_col] == 1 and row[signal_col] != 0:
            entry_price = row[price_col]
            direction = row[signal_col]

            # ATR based stop
            if direction == 1:
                stop_price = entry_price - atr_mult * row[atr_col]
            else:
                stop_price = entry_price + atr_mult * row[atr_col]

            max_r = 0

        # 📈 Trade running
        if entry_price is not None and stop_price is not None:
            r = compute_r_factor(
                entry_price,
                row[price_col],
                stop_price,
                direction
            )

            max_r = max(max_r, r)

            df.at[df.index[i], "entry_price"] = entry_price
            df.at[df.index[i], "stop_price"] = stop_price
            df.at[df.index[i], "R_factor"] = round(r, 2)
            df.at[df.index[i], "max_R"] = round(max_r, 2)

        # 🛑 Stop hit → reset
        if entry_price is not None:
            if (direction == 1 and row["low"] <= stop_price) or \
               (direction == -1 and row["high"] >= stop_price):
                entry_price = stop_price = None
                direction = 0
                max_r = 0

    return df

def add_volume_inflow_signal(
    df,
    mf_col="money_flow",
    price_col="close",
    lookback=20,
    spike_mult=1.5
):
    df = df.copy()

    mf_avg = df[mf_col].rolling(lookback).mean()

    df["INFLOW_TRADE_FLAG"] = np.select(
        [
            (df[mf_col] > mf_avg * spike_mult) &
            (df[price_col] > df[price_col].shift(1)),  # BUY

            (df[mf_col] > mf_avg * spike_mult) &
            (df[price_col] < df[price_col].shift(1))   # SELL
        ],
        [1, -1],
        default=0
    )

    # first entry only
    df["INFLOW_TRADE_FIRST"] = (
        (df["INFLOW_TRADE_FLAG"] != 0) &
        (df["INFLOW_TRADE_FLAG"].shift(1).fillna(0) == 0)
    ).astype(int)

    return df

def add_simple_big_candle(df, lookback=20, mult=1.5):
    df = df.copy()

    df["candle_range"] = df["high"] - df["low"]
    avg_range = df["candle_range"].rolling(lookback).mean()

    df["candle_bias"] = (
        df["close"] - df["low"] -
        (df["high"] - df["close"])
    )

    df["BIG_CANDLE_DIR"] = np.select(
        [
            (df["candle_range"] > avg_range * mult) & (df["candle_bias"] > 0),
            (df["candle_range"] > avg_range * mult) & (df["candle_bias"] < 0)
        ],
        [1, -1],
        default=0
    )

    return df



# ===================== CORE ENGINE =====================
def analyze(df, symbol):
    df = df.copy()
    df["symbol"] = symbol
    df["timestamp"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- PRICE / OI ---
    # --- PRICE / OI ---
    df["price_change_%"] = df["close"].pct_change() * 100
    df["oi_change_%"] = df["oi"].pct_change() * 100
    df["OI"] = df["oi"]          # MUST exist first

    df = add_max_oi(df)          # creates max_OI
    df = add_oi_drop(df)         # uses max_OI + OI  ✅
  
    # --- VWAP ---
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_pv"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_pv"] / df["cum_vol"]
    df["vwap_dev_%"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    df["flow_signal"] = np.where(df["vwap_dev_%"] > 0, "INFLOW", "OUTFLOW")

    # --- MONEY FLOW ---
    df["symbol1"] = df["symbol"].str[:-8]
    df["money_flow"] = df["close"] * df["volume"]
    df["money_change_%"] = df["money_flow"].pct_change() * 100
    df["money_change1_%"] = (df["money_flow"] - df["money_flow"].shift(1)) / df["money_flow"].shift(1) * 100
    df["money_change_cal"] = df["money_change1_%"].rolling(10).sum()
    df["money_change_spike"] = (df["money_change_cal"] > 200).astype(int)

    # --- DAILY INFLOW ---
    df["daily_inflow"] = df["money_flow"].cumsum()
    df["daily_inflow_change_%"] = df["daily_inflow"].pct_change() * 100

    # --- RSI / EMA ---
    df["rsi14"] = rsi(df["close"])
    df["ema_fast"] = ema(df["close"], 8)
    df["ema_slow"] = ema(df["close"], 21)

    # --- UMM / SMART MONEY ---
    mf_avg = df["money_flow"].rolling(20).mean()
    mf_std = df["money_flow"].rolling(20).std()
    df["UMM_score"] = (df["money_flow"] - mf_avg) / (mf_std + 1e-9)

    df["MFT_20"] = df["UMM_score"]
    df["SM_Div"] = rsi(df["money_flow"]) - df["rsi14"]

    df["IMA_Score"] = (
        0.4 * df["UMM_score"] +
        0.3 * df["MFT_20"] +
        0.2 * df["SM_Div"] +
        0.1 * df["vwap_dev_%"]
    )

    df["WMF"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
                (df["high"] - df["low"] + 1e-9) * df["volume"]

    # --- BREAKOUT / ATH / ATL ---
    df["rolling_high"] = df["close"].shift(1).rolling(20).max()
    df["rolling_low"] = df["close"].shift(1).rolling(20).min()

    df["is_breakout"] = df["close"] > df["rolling_high"]
    df["is_breakdown"] = df["close"] < df["rolling_low"]

    df["contract_ath"] = df.groupby("symbol")["close"].cummax()
    df["is_ath"] = df["close"] >= df["contract_ath"].shift(1)

    df["contract_atl"] = df.groupby("symbol")["close"].cummin()
    df["is_atl"] = df["close"] <= df["contract_atl"].shift(1)

    df["ath_count_10d"] = df.groupby("symbol")["is_ath"].rolling(10, min_periods=1).sum().reset_index(0, drop=True)
    df["ath_recent_flag"] = (df["ath_count_10d"] >= 1).astype(int)

    df["atl_count_10d"] = df.groupby("symbol")["is_atl"].rolling(10, min_periods=1).sum().reset_index(0, drop=True)
    df["atl_recent_flag"] = (df["atl_count_10d"] >= 1).astype(int)

    # --- FO BUILDUP ---
    def buildup(r):
        if r["oi_change_%"] > 0 and r["price_change_%"] > 0: return "LONG_BUILDUP"
        if r["oi_change_%"] > 0 and r["price_change_%"] < 0: return "SHORT_BUILDUP"
        if r["oi_change_%"] < 0 and r["price_change_%"] > 0: return "SHORT_COVERING"
        if r["oi_change_%"] < 0 and r["price_change_%"] < 0: return "LONG_UNWINDING"
        return "NONE"

    df["fo_buildup"] = df.apply(buildup, axis=1)
    df["fo_buildup_old"] = df["fo_buildup"].shift(1)

    # --- SIGNALS ---
    df["BUY_SIGNAL"] = (
        (df["IMA_Score"] > 1.2) &
        (df["ema_fast"] > df["ema_slow"]) &
        (df["vwap_dev_%"] > 0)
    )

    df["SELL_SIGNAL"] = (
        (df["IMA_Score"] < -1.2) &
        (df["ema_fast"] < df["ema_slow"]) &
        (df["vwap_dev_%"] < 0)
    )

    df["SIGNAL_REASON"] = np.where(df["BUY_SIGNAL"], "SMART_MONEY_BUY",
                            np.where(df["SELL_SIGNAL"], "SMART_MONEY_SELL", ""))

    # --- ALERTS ---
    # df["trigger_alert"] = np.where(df["is_breakout"], "FRESH_BREAKOUT",
    #                         np.where(df["is_breakdown"], "FRESH_BREAKDOWN", "NO"))
    def decide_alert(row):
        if row["is_breakout"] and row["money_change_%"] > 30:
            return "BREAKOUT_MONEY_SURGE"

        if row["is_breakdown"] and row["money_change_%"] < -30:
            return "BREAKDOWN_MONEY_SURGE"

        if row["is_breakout"] and row["is_ath"] and row["UMM_score"] > 2:
            return "BREAKOUT+ATH+UMM"

        if row["is_breakout"] and row["UMM_score"] > 2:
            return "BREAKOUT+UMM"

        if row["is_breakout"] and row["is_ath"]:
            return "BREAKOUT+ATH"

        if row["is_ath"]:
            return "ATH"

        if row["is_atl"]:
            return "ATL"

        if row["is_breakout"]:
            return "FRESH_BREAKOUT"

        if row["is_breakdown"]:
            return "FRESH_BREAKDOWN"

        return None


    df["trigger_alert"] = df.apply(decide_alert, axis=1)

    df["stacked_alert"] = None

    for i in range(len(df)):
        ts = pd.to_datetime(df.at[i, "timestamp"])
        base_alert = df.at[i, "trigger_alert"]

        stacked = update_and_check_alert_stack(
            df.at[i, "symbol"],
            base_alert,
            ts
        )

        if stacked:
            df.at[i, "trigger_alert"] = stacked


     
    # ALERT_SCORE = {
    #     "FRESH_BREAKOUT": 1,
    #     "FRESH_BREAKDOWN": -1
    # }
    # df["alert_score"] = df["trigger_alert"].map(ALERT_SCORE).fillna(0)

    # df["cnt_breakout_strength"] = df.groupby("symbol")["alert_score"].rolling(10, min_periods=1).sum().reset_index(0, drop=True)

    df["alert_score"] = df["trigger_alert"].map(ALERT_SCORE_MAP).fillna(0)

    df["cnt_breakout_strength"] = (
        df.groupby("symbol")["alert_score"]
        .rolling(10, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


    df["rsi14"] = compute_rsi_generic(df["close"])

    df["rsi_volume"] = compute_rsi_generic(df["volume"])


    df["SHORT_COVERING_OI"] = (
    (df["oi_drop_%"] > 5) &
    (df["price_change_%"] > 0)
    ).astype(int)

    df["LONG_UNWINDING_OI"] = (
    (df["oi_drop_%"] > 5) &
    (df["price_change_%"] < 0)
    ).astype(int)

    df["INSTITUTIONAL_EXIT"] = (
    (df["oi_drop_%"] > 7) &
    (df["money_change_cal"] < 0) &
    (df["vwap_dev_%"] < 0)
    ).astype(int)




    df = add_max_oi(df)
    df = add_oi_drop(df, window=10)
    df = add_volume_spike(df)
    df = add_anomaly_score(df)
    df = add_supertrend_dir(df)
    

    df = add_money_st_dir(df)
    df = add_smart_money_flags(df)
    df = add_dark_pool_activity(df)
    df = compute_inflow_rank_snapshot(df)
    # ---- Inflow dynamics ----
    df = add_inflow_dynamics(df)

    # ---- Inflow rank snapshot (already added earlier) ----
    df = compute_inflow_rank_snapshot(df)
    df = add_composite_trend_dir(df)
    
    


    # ---- Rolling alert pressure ----
    df = add_rolling_alert_signals(df, window=5)
    df = add_hlc_trend_dir(df)
    df["HLC_TREND_STRENGTH"] = (
    df["HLC_TREND_DIR"]
    .rolling(3, min_periods=1)
    .sum())


    # ---- Super Trade Signal (OI + Alert Pressure) ----

    df = add_super_trade_signal(df)
    df = add_supertrade_count(df, inactivity_sessions=5)
    df = add_hourly_supertrade_from_5min(df)
    df = add_volume_inflow_signal(df)  # creates INFLOW_TRADE_FLAG / FIRST

    df = add_r_factor(
        df,
        entry_flag_col="INFLOW_TRADE_FIRST",
        signal_col="INFLOW_TRADE_FLAG"
    )
    df = add_simple_big_candle(df) 

    #df = add_hourly_supertrade_counts_only(df, inactivity_hours=2)
    df = add_hourly_supertrade_counts_safe(
    df,
    inactivity_hours=2
)


    # ---- Collect SuperTrade FIRST triggers (NO sending here) ----
    last_row = df.iloc[-1]

    if (
        last_row["SUPER_TRADE_FLAG"] != 0 and
        last_row["SUPER_TRADE_FLAG"] != df["SUPER_TRADE_FLAG"].shift(1).iloc[-1]
    ):
        entry = f"{symbol} @ {round(last_row['close'], 2)}"

        if last_row["SUPER_TRADE_FLAG"] == 1:
            SUPER_BUY_ALERTS.append(entry)
        elif last_row["SUPER_TRADE_FLAG"] == -1:
            SUPER_SELL_ALERTS.append(entry)


    return df

# ===================== MAIN =====================
if __name__ == "__main__":
    kite = init_kite()
    instruments = get_futures(kite)

    all_df = []

    for _, row in instruments.iterrows():
        try:
            raw = fetch_intraday(kite, row["instrument_token"])
            if raw.empty:
                continue
            analyzed = analyze(raw, row["tradingsymbol"])
            all_df.append(analyzed)
        except Exception as e:
            print(row["tradingsymbol"], e)

    final_df = pd.concat(all_df, ignore_index=True)

    date_tag = datetime.now().strftime("%Y%m%d")
    outfile = f"fo_futures_full_scan_{date_tag}.csv"
    final_df = enforce_schema(final_df, FINAL_SCHEMA)

    final_df.to_csv(outfile, index=False)

    #send_consolidated_super_trade_alert()
    send_daily_first_supertrade_alerts(final_df)
    
    df_latest = (final_df.sort_values("timestamp").groupby("symbol1").tail(1))

    send_nifty50_regime_alert(df_latest)
    send_banknifty_regime_alert(df_latest)
    send_sensex_regime_alert(df_latest)




    print(f"✅ FULL SCAN GENERATED → {outfile}")
