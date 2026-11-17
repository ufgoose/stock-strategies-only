import pandas as pd
import numpy as np


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # 1. 布林通道 (20日, 2倍標準差) 與帶寬真開口
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # 條件 2：帶寬比昨天大，且上軌往上、下軌往下（兩邊真張開，非平移）
    df["cond_bb_open"] = (
        (df["bb_width"] > df["bb_width"].shift(1)) &
        (df["bb_upper"] > df["bb_upper"].shift(1)) &
        (df["bb_lower"] < df["bb_lower"].shift(1))
    )

    # KD 指標（保留原本欄位以免其他模組讀取報錯）
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    df["k"] = rsv.ewm(com=2).mean()
    df["d"] = df["k"].ewm(com=2).mean()

    # 2. MACD(12, 26, 9) — 採用加權收盤價，與 XQ 內建算法完全對齊
    w_close = (df["high"] + df["low"] + 2 * df["close"]) / 4.0
    ema12 = w_close.ewm(span=12, adjust=False).mean()
    ema26 = w_close.ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["dif"] - df["dea"]

    # 條件 1：柱狀體今天 > 0、昨天 <= 0（剛好今天翻正）
    df["cond_macd_flip"] = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)

    # 3. 月線條件：收盤 > 20日線，且 20日線比 ma_up_bars 天前高
    # 想更嚴格的話，可把 ma_up_bars 改成 3 或 5
    ma_up_bars = 1
    df["cond_ma20_up"] = (
        (df["close"] > df["ma20"]) &
        (df["ma20"] > df["ma20"].shift(ma_up_bars))
    )

    # 4. 成交量條件：最低成交量 500 張（自動相容「股」或「張」單位；設為 0 就關閉）
    min_lots = 500
    vol_col = "volume" if "volume" in df.columns else "Trading_Volume"
    if min_lots <= 0:
        df["cond_vol"] = True
    elif vol_col in df.columns:
        # 若數值大於 10000 視為「股」(500張 = 500,000股)，否則視為「張」
        threshold = min_lots * 1000 if df[vol_col].median() > 10000 else min_lots
        df["cond_vol"] = df[vol_col] >= threshold
    else:
        df["cond_vol"] = True

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    return df


def tech_score_at(row: pd.Series, params: dict | None = None) -> dict:
    """對一天計算技術分 (0-100)。
    整合自訂四條件：MACD柱狀體翻正 + 布林通道真開口 + 站上上揚月線 + 成交量 >= 500張。
    """
    if params is None:
        params = {}

    c1 = bool(row.get("cond_macd_flip", False))
    c2 = bool(row.get("cond_bb_open", False))
    c3 = bool(row.get("cond_ma20_up", False))
    c4 = bool(row.get("cond_vol", False))

    score = 0.0
    signals: list[str] = []

    if c1:
        score += 30
        signals.append("MACD柱狀體翻正")
    if c2:
        score += 30
        signals.append("布林通道真開口")
    if c3:
        score += 25
        signals.append("站上上揚月線")
    if c4:
        score += 15
        signals.append("量能>=500張")

    # 嚴格把關：只有四項條件「同時成立」才給 100 分（觸發回測取樣 >= 60 與 BUY 門檻）
    # 若有任何一項沒過，最高壓在 45 分，避免誤判成買進訊號
    if not (c1 and c2 and c3 and c4):
        score = min(score, 45.0)

    return {"score": int(round(score)), "signals": signals}
