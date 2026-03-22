import streamlit as st
import pandas as pd
import psycopg2
from datetime import date


def get_conn():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
    )


def load_health_data() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("SELECT date, weight, protein, calories, sleep FROM health ORDER BY date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def upsert_health_row(entry: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO health (date, weight, protein, calories, sleep)
                VALUES (%(date)s, %(weight)s, %(protein)s, %(calories)s, %(sleep)s)
                ON CONFLICT (date) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    protein = EXCLUDED.protein,
                    calories = EXCLUDED.calories,
                    sleep = EXCLUDED.sleep
                """,
                entry,
            )
        conn.commit()


def delete_health_row(target_date):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM health WHERE date = %s", (target_date,))
        conn.commit()


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


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tracker", page_icon="📊", layout="wide")

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

st.title("Tracker")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_health, tab_analytics = st.tabs(["🏋️ Health", "📈 Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_health:
    df = load_health_data()

    with st.expander("**Add new entry**", expanded=True):
        cols = st.columns([1.5, 1, 1, 1, 1, 0.8])
        with cols[0]:
            entry_date = st.date_input("Date", value=date.today(), key="health_date")
        with cols[1]:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1, key="health_weight")
        with cols[2]:
            protein = st.number_input("Protein (g)", min_value=0, max_value=1000, value=120, step=1, key="health_protein")
        with cols[3]:
            calories = st.number_input("Calories", min_value=0, max_value=10000, value=2000, step=50, key="health_calories")
        with cols[4]:
            sleep = st.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=7.5, step=0.25, key="health_sleep")
        with cols[5]:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add", type="primary", use_container_width=True, key="health_add")

        if add_btn:
            entry = {
                "date": str(entry_date),
                "weight": weight,
                "protein": protein,
                "calories": calories,
                "sleep": sleep,
            }
            upsert_health_row(entry)
            st.success(f"Saved entry for {entry_date}.")
            st.rerun()

    # ── Daily log ─────────────────────────────────────────────────────────────
    st.subheader("Daily Log")

    if df.empty:
        st.info("No entries yet. Add your first entry above!")
    else:
        display_df = df.sort_values("date", ascending=False).reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.columns = ["Date", "Weight (kg)", "Protein (g)", "Calories", "Sleep (hrs)"]
        st.dataframe(display_df, use_container_width=True, height=min(len(display_df) * 40 + 60, 500))

        with st.expander("Delete an entry"):
            date_options = sorted(df["date"].tolist(), reverse=True)
            del_date = st.selectbox(
                "Select date to delete",
                date_options,
                format_func=lambda d: d.strftime("%Y-%m-%d"),
                key="health_del_date",
            )
            if st.button("Delete", type="secondary", key="health_del_btn"):
                delete_health_row(del_date)
                st.rerun()

    # ── Weekly averages ───────────────────────────────────────────────────────
    st.subheader("Weekly Averages")

    if df.empty:
        st.info("Add entries to see weekly averages.")
    else:
        weekly = compute_weekly_averages(df)
        display_weekly = weekly.copy()
        display_weekly.columns = ["Week", "Days Logged", "Avg Weight (kg)", "Avg Protein (g)", "Avg Calories", "Avg Sleep (hrs)"]
        display_weekly.index = range(1, len(display_weekly) + 1)
        st.dataframe(display_weekly, use_container_width=True)

        today = date.today()
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

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    df = load_health_data()

    if df.empty:
        st.info("Add health entries to see charts.")
    else:
        chart_df = df.sort_values("date").copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        chart_df = chart_df.set_index("date")

        st.subheader("Weight Trend")
        st.line_chart(chart_df["weight"], color="#FF6B6B")

        st.subheader("Protein Intake")
        st.bar_chart(chart_df["protein"], color="#4ECDC4")

        st.subheader("Calorie Intake")
        st.bar_chart(chart_df["calories"], color="#FFE66D")

        st.subheader("Sleep")
        st.line_chart(chart_df["sleep"], color="#6C5CE7")
