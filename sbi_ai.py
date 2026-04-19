import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from xgboost import XGBClassifier

# =========================
# Discord通知
# =========================
WEBHOOK_URL = "https://discord.com/api/webhooks/1495436109038227556/EE_CBMLrOZltLm-Tqv5o7LTbFifxhADNTY3Tff40XfKsiXLNoLPtpBRktsoCVsXA4-0S"

def send_discord(msg):
    try:
        requests.post(WEBHOOK_URL, json={"content": msg})
    except:
        pass

# =========================
# 時間枠（GitHub用）
# =========================
now = datetime.utcnow()  # GitHubはUTC
hour = (now.hour + 9) % 24  # 日本時間に変換

slots = [
    {"label": "09:00約定（朝）", "hour": 9, "threshold": 0.65},
    {"label": "12:00約定（昼）", "hour": 12, "threshold": 0.63},
    {"label": "15:00約定（引け）", "hour": 15, "threshold": 0.60},
]

slot = min(slots, key=lambda s: abs(s["hour"] - hour))
label = slot["label"]
threshold = slot["threshold"]

# =========================
# 銘柄
# =========================
stocks = {
    "トヨタ": "7203.T",
    "三菱UFJ": "8306.T",
}

# =========================
# 特徴量
# =========================
def make_features(df):
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return_5d"] = df["Close"].pct_change(5)

    return df

# =========================
# 実行
# =========================
for name, ticker in stocks.items():

    data = yf.download(ticker, period="5y")

    df = pd.DataFrame(index=data.index)
    df["Close"] = data["Close"].squeeze()

    df = make_features(df)

    df["Future"] = df["Close"].shift(-5)
    df["Target"] = ((df["Future"] - df["Close"]) / df["Close"] > 0.015).astype(int)

    df = df.dropna().reset_index(drop=True)

    features = ["Close", "MA25", "MA75", "RSI", "Return_5d"]

    X = df[features]
    y = df["Target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss"
    )

    model.fit(X, y)

    latest = X.tail(1)
    prob = model.predict_proba(latest)[0][1]

    price = float(df["Close"].iloc[-1])

    ma25 = df["MA25"].iloc[-1]
    ma75 = df["MA75"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    trend = (ma25 - ma75) / ma75

    trend_ok = ma25 > ma75 and trend > 0.002
    rsi_ok = 45 < rsi < 70
    prob_ok = prob >= threshold

    if trend_ok and rsi_ok and prob_ok:
        signal = "BUY"
        take_profit = 0.08
        stop_loss = 0.03
    else:
        signal = "WAIT"
        take_profit = None
        stop_loss = None

    msg = f"""
【S株AI GitHub版】

時間枠: {label}
銘柄: {name}

価格: {price:.2f}
上昇確率: {prob*100:.2f}%

判定: {signal}

トレンド: {trend:.4f}
RSI: {rsi:.2f}

利確: {take_profit}
損切り: {stop_loss}
"""

    print(msg)
    send_discord(msg)
