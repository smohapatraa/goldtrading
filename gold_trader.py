import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import numpy as np
from streamlit_gsheets import GSheetsConnection

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Gold Trading Portfolio Tracker",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# IST TIME HELPER
# ------------------------------------------------------------
def _ist_now():
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%H:%M:%S")

def _ist_today():
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).date()

# ------------------------------------------------------------
# GOOGLE SHEETS CONNECTION
# ------------------------------------------------------------
conn = st.connection("gsheets", type=GSheetsConnection)

# Worksheet names in your Google Sheet
BUYS_SHEET = "Buys"
SELLS_SHEET = "Sells"

# Column definitions
BUY_COLUMNS = ["id", "buy_date", "shares", "price", "invested_amount", "notes", "created_at"]
SELL_COLUMNS = ["id", "sell_date", "shares", "price", "sold_amount", "profit", "notes", "created_at"]

# ------------------------------------------------------------
# LOAD DATA FROM GOOGLE SHEETS
# ------------------------------------------------------------
def load_buys():
    try:
        df = conn.read(worksheet=BUYS_SHEET, ttl=5)
        if df.empty:
            return pd.DataFrame(columns=BUY_COLUMNS)
        df = df.dropna(how='all')
        return df
    except Exception:
        return pd.DataFrame(columns=BUY_COLUMNS)


def load_sells():
    try:
        df = conn.read(worksheet=SELLS_SHEET, ttl=5)
        if df.empty:
            return pd.DataFrame(columns=SELL_COLUMNS)
        df = df.dropna(how='all')
        return df
    except Exception:
        return pd.DataFrame(columns=SELL_COLUMNS)


# ------------------------------------------------------------
# WRITE TO GOOGLE SHEETS
# ------------------------------------------------------------
def save_buys(df):
    conn.update(worksheet=BUYS_SHEET, data=df)


def save_sells(df):
    conn.update(worksheet=SELLS_SHEET, data=df)


def add_buy(buy_date, shares, price, notes=""):
    buys = load_buys()
    invested = shares * price

    new_id = 1
    if not buys.empty:
        new_id = int(buys['id'].max()) + 1

    new_row = pd.DataFrame([{
        "id": new_id,
        "buy_date": str(buy_date),
        "shares": int(shares),
        "price": float(price),
        "invested_amount": float(invested),
        "notes": notes,
        "created_at": str(datetime.now())
    }])

    buys = pd.concat([buys, new_row], ignore_index=True)
    save_buys(buys)


def add_sell(sell_date, shares, price, cost_basis, notes=""):
    sells = load_sells()
    sold = shares * price
    profit = sold - (shares * cost_basis)

    new_id = 1
    if not sells.empty:
        new_id = int(sells['id'].max()) + 1

    new_row = pd.DataFrame([{
        "id": new_id,
        "sell_date": str(sell_date),
        "shares": int(shares),
        "price": float(price),
        "sold_amount": float(sold),
        "profit": float(profit),
        "notes": notes,
        "created_at": str(datetime.now())
    }])

    sells = pd.concat([sells, new_row], ignore_index=True)
    save_sells(sells)


def delete_buy(buy_id):
    buys = load_buys()
    buys = buys[buys['id'] != buy_id]
    save_buys(buys)


def delete_sell(sell_id):
    sells = load_sells()
    sells = sells[sells['id'] != sell_id]
    save_sells(sells)


# ------------------------------------------------------------
# PORTFOLIO CALCULATION
# ------------------------------------------------------------
def compute_portfolio(buys_df, sells_df):
    total_shares_bought = int(buys_df['shares'].sum()) if not buys_df.empty else 0
    total_shares_sold = int(sells_df['shares'].sum()) if not sells_df.empty else 0
    net_shares = total_shares_bought - total_shares_sold

    total_invested = float(buys_df['invested_amount'].sum()) if not buys_df.empty else 0.0
    total_sold_amount = float(sells_df['sold_amount'].sum()) if not sells_df.empty else 0.0
    total_profit = float(sells_df['profit'].sum()) if not sells_df.empty else 0.0

    avg_cost = (total_invested / total_shares_bought) if total_shares_bought > 0 else 0.0
    unsold_cost = net_shares * avg_cost

    return {
        'total_shares_bought': total_shares_bought,
        'total_shares_sold': total_shares_sold,
        'net_shares': net_shares,
        'total_invested': total_invested,
        'total_sold_amount': total_sold_amount,
        'total_profit': total_profit,
        'avg_cost': avg_cost,
        'unsold_cost': unsold_cost,
    }


# ------------------------------------------------------------
# LIVE PRICE HELPERS
# ------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_gold_data():
    result = {}
    try:
        gold = yf.download("GC=F", period="90d", progress=False, auto_adjust=False)
        if not gold.empty:
            result['gold'] = gold
    except Exception:
        pass
    try:
        usdinr = yf.download("INR=X", period="90d", progress=False, auto_adjust=False)
        if not usdinr.empty:
            result['usdinr'] = usdinr
    except Exception:
        pass
    try:
        setfgold = yf.download("SETFGOLD.NS", period="90d", progress=False, auto_adjust=False)
        if not setfgold.empty:
            result['setfgold'] = setfgold
    except Exception:
        pass
    return result


def _squeeze_close(df):
    if df is None or df.empty:
        return None
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce').dropna()
    return close if len(close) > 0 else None


def predict_next_day_open(gold_df, inr_df, etf_df):
    predictions = {}

    gold_close = _squeeze_close(gold_df)
    if gold_close is not None and len(gold_close) >= 10:
        last = float(gold_close.iloc[-1])
        recent_5 = gold_close.tail(5).values.astype(float)
        slope = float(np.polyfit(np.arange(len(recent_5)), recent_5, 1)[0])
        returns_3 = float(gold_close.pct_change().tail(3).mean())
        gold_pred = float((last + slope) * 0.6 + last * (1 + returns_3) * 0.4)
        predictions['gold'] = {
            'last_close': round(last, 2),
            'predicted_open': round(gold_pred, 2),
            'change_pct': round(((gold_pred - last) / last) * 100, 2),
        }

    inr_close = _squeeze_close(inr_df)
    if inr_close is not None and len(inr_close) >= 10:
        last = float(inr_close.iloc[-1])
        recent_5 = inr_close.tail(5).values.astype(float)
        slope = float(np.polyfit(np.arange(len(recent_5)), recent_5, 1)[0])
        returns_3 = float(inr_close.pct_change().tail(3).mean())
        inr_pred = float((last + slope) * 0.6 + last * (1 + returns_3) * 0.4)
        predictions['usdinr'] = {
            'last_close': round(last, 4),
            'predicted_open': round(inr_pred, 4),
            'change_pct': round(((inr_pred - last) / last) * 100, 3),
        }

    etf_close = _squeeze_close(etf_df)
    if etf_close is not None and len(etf_close) >= 10:
        last = float(etf_close.iloc[-1])
        recent_5 = etf_close.tail(5).values.astype(float)
        slope = float(np.polyfit(np.arange(len(recent_5)), recent_5, 1)[0])
        returns_3 = float(etf_close.pct_change().tail(3).mean())
        etf_pred_tech = float((last + slope) * 0.6 + last * (1 + returns_3) * 0.4)

        etf_pred_corr = None
        if 'gold' in predictions and 'usdinr' in predictions:
            combined = (
                predictions['gold']['change_pct'] / 100 +
                predictions['usdinr']['change_pct'] / 100
            )
            etf_pred_corr = float(last * (1 + combined))

        etf_pred_final = (
            (etf_pred_tech * 0.5 + etf_pred_corr * 0.5)
            if etf_pred_corr is not None else etf_pred_tech
        )

        predictions['setfgold'] = {
            'last_close': round(last, 2),
            'predicted_open': round(etf_pred_final, 2),
            'change_pct': round(((etf_pred_final - last) / last) * 100, 2),
        }

    return predictions


# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------
st.title("🥇 Gold Trading Portfolio & Prediction Dashboard")
st.caption(f"🕐 Live prices updated {_ist_now()} IST · Data stored in Google Sheets")

col_r1, col_r2 = st.columns([1, 4])
with col_r1:
    if st.button("🔄 Refresh Prices", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.toast("♻️ Refreshed — fetching latest prices...", icon="🔄")
        st.rerun()

st.divider()

# Load from Google Sheets
buys_df = load_buys()
sells_df = load_sells()

portfolio = compute_portfolio(buys_df, sells_df)

with st.spinner("Fetching live gold prices..."):
    bundle = fetch_gold_data()

gold_df = bundle.get('gold')
inr_df = bundle.get('usdinr')
etf_df = bundle.get('setfgold')

predictions = predict_next_day_open(gold_df, inr_df, etf_df)

current_price = None
if 'setfgold' in predictions:
    current_price = predictions['setfgold']['last_close']
elif etf_df is not None:
    s = _squeeze_close(etf_df)
    if s is not None:
        current_price = float(s.iloc[-1])

# ------------------------------------------------------------
# PORTFOLIO SUMMARY
# ------------------------------------------------------------
st.subheader("📊 Portfolio Summary")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    st.metric("Net Holdings", f"{portfolio['net_shares']} units")

with col_p2:
    st.metric("Avg Cost", f"₹{portfolio['avg_cost']:,.2f}")

with col_p3:
    if current_price is not None and portfolio['net_shares'] > 0:
        unrealized = (current_price - portfolio['avg_cost']) * portfolio['net_shares']
        pct = ((current_price - portfolio['avg_cost']) / portfolio['avg_cost'] * 100) if portfolio['avg_cost'] > 0 else 0
        st.metric("Unrealized P&L", f"₹{unrealized:,.0f}", delta=f"{pct:+.2f}%")
    else:
        st.metric("Unrealized P&L", "—")

with col_p4:
    st.metric("Realized Profit", f"₹{portfolio['total_profit']:,.0f}")

# ------------------------------------------------------------
# 25% TARGET CALCULATOR
# ------------------------------------------------------------
st.divider()
st.subheader("🎯 Next Sell Target (25% Profit Target)")

if portfolio['net_shares'] > 0 and portfolio['avg_cost'] > 0:
    target_price = portfolio['avg_cost'] * 1.25
    target_value = target_price * portfolio['net_shares']

    if current_price is not None:
        gap_pct = ((target_price - current_price) / current_price) * 100
    else:
        gap_pct = 0

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)

    with col_t1:
        st.metric("Unsold Units", f"{portfolio['net_shares']}")

    with col_t2:
        st.metric("Avg Cost", f"₹{portfolio['avg_cost']:,.2f}")

    with col_t3:
        st.metric("🎯 Sell Target", f"₹{target_price:,.2f}")

    with col_t4:
        if current_price is not None:
            st.metric("Distance", f"{gap_pct:+.2f}%")
        else:
            st.metric("Distance", "N/A")

    st.info(f"""
    💡 **Action Plan:**
    - Sell **{portfolio['net_shares']} unsold units** when SETFGOLD reaches **₹{target_price:,.2f}**
    - Estimated proceeds: **₹{target_value:,.0f}**
    - Estimated profit: **₹{target_value - portfolio['unsold_cost']:,.0f}** (25% of ₹{portfolio['unsold_cost']:,.0f})
    """)
else:
    st.info("ℹ️ No unsold holdings yet. Add a BUY transaction to see your 25% profit target.")

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------
st.divider()
st.subheader("🔮 Next-Day Gold Price Prediction")

if predictions:
    col_pred1, col_pred2, col_pred3 = st.columns(3)

    with col_pred1:
        if 'gold' in predictions:
            p = predictions['gold']
            arrow = "🔼" if p['change_pct'] > 0 else ("🔽" if p['change_pct'] < 0 else "➡️")
            st.metric("🌍 Gold (COMEX)", f"${p['predicted_open']:,.2f}",
                      delta=f"{p['change_pct']:+.2f}% {arrow}")
            st.caption(f"Last: ${p['last_close']:,.2f}")

    with col_pred2:
        if 'usdinr' in predictions:
            p = predictions['usdinr']
            arrow = "🔼" if p['change_pct'] > 0 else ("🔽" if p['change_pct'] < 0 else "➡️")
            st.metric("💱 USD/INR", f"₹{p['predicted_open']:,.4f}",
                      delta=f"{p['change_pct']:+.3f}% {arrow}")
            st.caption(f"Last: ₹{p['last_close']:,.4f}")

    with col_pred3:
        if 'setfgold' in predictions:
            p = predictions['setfgold']
            arrow = "🔼" if p['change_pct'] > 0 else ("🔽" if p['change_pct'] < 0 else "➡️")
            st.metric("🥇 SETFGOLD (NSE)", f"₹{p['predicted_open']:,.2f}",
                      delta=f"{p['change_pct']:+.2f}% {arrow}")
            st.caption(f"Last: ₹{p['last_close']:,.2f}")

# ------------------------------------------------------------
# TRADING SUGGESTION
# ------------------------------------------------------------
if 'gold' in predictions and 'usdinr' in predictions and 'setfgold' in predictions:
    gold_chg = predictions['gold']['change_pct']
    inr_chg = predictions['usdinr']['change_pct']
    etf_chg = predictions['setfgold']['change_pct']

    if gold_chg > 0.3 and inr_chg > 0:
        st.success(f"🟢 **STRONG BUY** — Gold {gold_chg:+.2f}% + INR weak {inr_chg:+.3f}% → Expected: {etf_chg:+.2f}%")
    elif gold_chg > 0.3 and inr_chg <= 0:
        st.success(f"🟢 **BUY (Mild)** — Gold {gold_chg:+.2f}%, INR strong. Expected: {etf_chg:+.2f}%")
    elif gold_chg < -0.3 and inr_chg > 0:
        st.warning(f"🟡 **HOLD / WAIT** — Gold {gold_chg:+.2f}%, INR weak. Expected: {etf_chg:+.2f}%")
    elif gold_chg < -0.3 and inr_chg <= 0:
        st.error(f"🔴 **SELL / AVOID** — Gold {gold_chg:+.2f}% + INR strong {inr_chg:+.3f}% → Expected: {etf_chg:+.2f}%")
    else:
        st.info(f"⚪ **NEUTRAL** — Gold {gold_chg:+.2f}%, INR {inr_chg:+.3f}%. Expected: {etf_chg:+.2f}%")

# ------------------------------------------------------------
# TRANSACTION ENTRY
# ------------------------------------------------------------
st.divider()
st.subheader("➕ Add Transaction")

tab_buy, tab_sell = st.tabs(["🟢 Buy", "🔴 Sell"])

with tab_buy:
    with st.form("buy_form", clear_on_submit=True):
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            buy_date = st.date_input("Buy Date", value=_ist_today())

        with col_b2:
            buy_shares = st.number_input("Number of Units", min_value=1, value=1, step=1)

        with col_b3:
            default_price = current_price if current_price else 0.0
            buy_price = st.number_input("Price per Unit (₹)", min_value=0.01,
                                        value=float(default_price), step=0.01)

        buy_notes = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("✅ Record Buy", use_container_width=True, type="primary")

        if submitted:
            if buy_shares > 0 and buy_price > 0:
                add_buy(buy_date, int(buy_shares), float(buy_price), buy_notes)
                st.success(f"✅ Recorded: {buy_shares} units @ ₹{buy_price:,.2f}")
                st.rerun()
            else:
                st.error("Please enter valid shares and price.")

with tab_sell:
    if portfolio['net_shares'] <= 0:
        st.warning("⚠️ No unsold holdings to sell.")
    else:
        with st.form("sell_form", clear_on_submit=True):
            col_s1, col_s2, col_s3 = st.columns(3)

            with col_s1:
                sell_date = st.date_input("Sell Date", value=_ist_today())

            with col_s2:
                sell_shares = st.number_input("Number of Units", min_value=1,
                                              max_value=portfolio['net_shares'],
                                              value=min(portfolio['net_shares'], 1), step=1)

            with col_s3:
                default_sell_price = current_price if current_price else portfolio['avg_cost'] * 1.25
                sell_price = st.number_input("Price per Unit (₹)", min_value=0.01,
                                             value=float(default_sell_price), step=0.01)

            sell_notes = st.text_input("Notes (optional)")

            submitted = st.form_submit_button("✅ Record Sell", use_container_width=True, type="primary")

            if submitted:
                if sell_shares > 0 and sell_price > 0:
                    add_sell(sell_date, int(sell_shares), float(sell_price),
                             portfolio['avg_cost'], sell_notes)
                    st.success(f"✅ Sold: {sell_shares} units @ ₹{sell_price:,.2f}")
                    st.rerun()
                else:
                    st.error("Please enter valid shares and price.")

# ------------------------------------------------------------
# TRANSACTION HISTORY
# ------------------------------------------------------------
st.divider()
st.subheader("📜 Transaction History")

col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown("### 🟢 Buy Transactions")
    if buys_df.empty:
        st.info("No buy transactions yet.")
    else:
        display_buys = buys_df[['id', 'buy_date', 'shares', 'price', 'invested_amount', 'notes']].copy()
        display_buys.columns = ['ID', 'Date', 'Units', 'Price', 'Invested (₹)', 'Notes']
        st.dataframe(display_buys, hide_index=True, use_container_width=True)
        st.caption(f"Total invested: **₹{portfolio['total_invested']:,.2f}**")

with col_h2:
    st.markdown("### 🔴 Sell Transactions")
    if sells_df.empty:
        st.info("No sell transactions yet.")
    else:
        display_sells = sells_df[['id', 'sell_date', 'shares', 'price', 'sold_amount', 'profit', 'notes']].copy()
        display_sells.columns = ['ID', 'Date', 'Units', 'Price', 'Sold (₹)', 'Profit (₹)', 'Notes']
        st.dataframe(display_sells, hide_index=True, use_container_width=True)
        st.caption(f"Total sold: **₹{portfolio['total_sold_amount']:,.2f}** · Profit: **₹{portfolio['total_profit']:,.2f}**")

# ------------------------------------------------------------
# DELETE TRANSACTION
# ------------------------------------------------------------
with st.expander("🗑️ Delete a Transaction"):
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if not buys_df.empty:
            buy_to_delete = st.selectbox("Select Buy to delete", buys_df['id'].tolist())
            if st.button("🗑️ Delete Buy", key="del_buy"):
                delete_buy(int(buy_to_delete))
                st.success(f"Deleted buy #{buy_to_delete}")
                st.rerun()

    with col_d2:
        if not sells_df.empty:
            sell_to_delete = st.selectbox("Select Sell to delete", sells_df['id'].tolist())
            if st.button("🗑️ Delete Sell", key="del_sell"):
                delete_sell(int(sell_to_delete))
                st.success(f"Deleted sell #{sell_to_delete}")
                st.rerun()

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("""
🥇 **Gold Trading Portfolio Tracker** · Built by S. Mohapatra

⚠️ **Disclaimer:** For educational and personal tracking purposes only. 
Not investment advice. Prices from Yahoo Finance (may be delayed).
""")
