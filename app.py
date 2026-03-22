import streamlit as st
import pandas as pd
import json
import os
from datetime import date, timedelta

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            records = json.load(f)
        if records:
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
    return pd.DataFrame(columns=["date", "weight", "protein", "calories", "sleep"])


def save_data(df: pd.DataFrame):
    records = df.copy()
    records["date"] = records["date"].astype(str)
    with open(DATA_FILE, "w") as f:
        json.dump(records.to_dict(orient="records"), f, indent=2)


def compute_weekly_averages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["week_start"] = df["date"].dt.to_period("W-SAT").apply(lambda p: p.start_time.date())
    df["week_end"] = df["date"].dt.to_period("W-SAT").apply(lambda p: p.end_time.date())

    weekly = (
        df.groupby(["week_start", "week_end"])
        .agg(
            weight=("weight", "mean"),
            protein=("protein", "mean"),
            calories=("calories", "mean"),
            sleep=("sleep", "mean"),
            days_logged=("date", "count"),
        )
        .reset_index()
    )
    weekly["week"] = weekly.apply(
        lambda r: f"{r['week_start'].strftime('%b %d')} – {r['week_end'].strftime('%b %d')}", axis=1
    )
    weekly = weekly.sort_values("week_start", ascending=False)
    return weekly[["week", "days_logged", "weight", "protein", "calories", "sleep"]]


st.set_page_config(page_title="Health Tracker", page_icon="💪", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 960px; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Health Tracker")
st.caption("Track weight, protein, calories & sleep — see your weekly averages.")

df = load_data()

# ── Add / Edit entry ──────────────────────────────────────────────────────────
with st.expander("**Add new entry**", expanded=True):
    cols = st.columns([1.5, 1, 1, 1, 1, 0.8])
    with cols[0]:
        entry_date = st.date_input("Date", value=date(2026, 3, 21))
    with cols[1]:
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
    with cols[2]:
        protein = st.number_input("Protein (g)", min_value=0, max_value=1000, value=120, step=1)
    with cols[3]:
        calories = st.number_input("Calories", min_value=0, max_value=10000, value=2000, step=50)
    with cols[4]:
        sleep = st.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=7.5, step=0.25)
    with cols[5]:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("Add", type="primary", use_container_width=True)

    if add_btn:
        existing = df["date"] == entry_date if not df.empty else pd.Series(dtype=bool)
        if existing.any():
            df.loc[existing, ["weight", "protein", "calories", "sleep"]] = [weight, protein, calories, sleep]
            st.success(f"Updated entry for {entry_date}.")
        else:
            new_row = pd.DataFrame(
                [{"date": entry_date, "weight": weight, "protein": protein, "calories": calories, "sleep": sleep}]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            st.success(f"Added entry for {entry_date}.")
        save_data(df)
        st.rerun()

# ── Daily log table ───────────────────────────────────────────────────────────
st.subheader("Daily Log")

if df.empty:
    st.info("No entries yet. Add your first entry above!")
else:
    display_df = df.sort_values("date", ascending=False).reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.columns = ["Date", "Weight (kg)", "Protein (g)", "Calories", "Sleep (hrs)"]
    st.dataframe(display_df, use_container_width=True, height=min(len(display_df) * 40 + 60, 500))

    # delete an entry
    with st.expander("Delete an entry"):
        date_options = sorted(df["date"].tolist(), reverse=True)
        del_date = st.selectbox("Select date to delete", date_options, format_func=lambda d: d.strftime("%Y-%m-%d"))
        if st.button("Delete", type="secondary"):
            df = df[df["date"] != del_date]
            save_data(df)
            st.rerun()

# ── Weekly averages ───────────────────────────────────────────────────────────
st.subheader("Weekly Averages")

if df.empty:
    st.info("Add entries to see weekly averages.")
else:
    weekly = compute_weekly_averages(df)
    fmt = {
        "weight": "{:.1f}",
        "protein": "{:.0f}",
        "calories": "{:.0f}",
        "sleep": "{:.1f}",
    }
    display_weekly = weekly.copy()
    display_weekly.columns = ["Week", "Days Logged", "Avg Weight (kg)", "Avg Protein (g)", "Avg Calories", "Avg Sleep (hrs)"]
    display_weekly.index = range(1, len(display_weekly) + 1)
    st.dataframe(display_weekly, use_container_width=True)

    # quick summary of current week
    today = date(2026, 3, 21)
    current_period = pd.Timestamp(today).to_period("W-SAT")
    current_week_start = current_period.start_time.date()
    current_week_end = current_period.end_time.date()
    current_week = df[
        (pd.to_datetime(df["date"]).dt.date >= current_week_start)
        & (pd.to_datetime(df["date"]).dt.date <= current_week_end)
    ]

    if not current_week.empty:
        st.markdown("#### This Week at a Glance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Weight", f"{current_week['weight'].mean():.1f} kg")
        m2.metric("Avg Protein", f"{current_week['protein'].mean():.0f} g")
        m3.metric("Avg Calories", f"{current_week['calories'].mean():.0f}")
        m4.metric("Avg Sleep", f"{current_week['sleep'].mean():.1f} hrs")
