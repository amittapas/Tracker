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
        sslmode="require",
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
    .app-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,.06);
        border: 1px solid rgba(255,255,255,.8);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f2f6;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }

    /* Dataframes */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .stButton > button[kind="secondary"] {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 10px;
    }

    /* Section headers */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #ddd, transparent);
        margin: 1.5rem 0;
    }

    /* Stat card row */
    .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; margin-top: 2px; }

    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,.04);
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>Tracker</h1>
        <p>Daily health metrics — weight, protein, calories & sleep</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_health, tab_analytics = st.tabs(["🏋️  Health", "📈  Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_health:
    df = load_health_data()

    # ── Entry form ────────────────────────────────────────────────────────────
    with st.expander("**+ New Entry**", expanded=True):
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
            add_btn = st.button("Save", type="primary", use_container_width=True, key="health_add")

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

    # ── This Week at a Glance ─────────────────────────────────────────────────
    if not df.empty:
        today = date.today()
        current_period = pd.Timestamp(today).to_period("W-SAT")
        current_week_start = current_period.start_time.date()
        current_week_end = current_period.end_time.date()
        current_week = df[
            (pd.to_datetime(df["date"]).dt.date >= current_week_start)
            & (pd.to_datetime(df["date"]).dt.date <= current_week_end)
        ]

        if not current_week.empty:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### This Week")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Days Logged", f"{len(current_week)}/7")
            m2.metric("Avg Weight", f"{current_week['weight'].mean():.1f} kg")
            m3.metric("Avg Protein", f"{current_week['protein'].mean():.0f} g")
            m4.metric("Avg Calories", f"{current_week['calories'].mean():.0f}")
            m5.metric("Avg Sleep", f"{current_week['sleep'].mean():.1f} hrs")

    # ── Daily log ─────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Daily Log")

    if df.empty:
        st.info("No entries yet — add your first one above!")
    else:
        display_df = df.sort_values("date", ascending=False).reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.columns = ["Date", "Weight (kg)", "Protein (g)", "Calories", "Sleep (hrs)"]
        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(len(display_df) * 40 + 60, 500),
            column_config={
                "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                "Weight (kg)": st.column_config.NumberColumn("Weight (kg)", format="%.1f"),
                "Protein (g)": st.column_config.NumberColumn("Protein (g)", format="%d"),
                "Calories": st.column_config.NumberColumn("Calories", format="%d"),
                "Sleep (hrs)": st.column_config.NumberColumn("Sleep (hrs)", format="%.1f"),
            },
        )

        with st.expander("Delete an entry"):
            date_options = sorted(df["date"].tolist(), reverse=True)
            del_date = st.selectbox(
                "Select date to delete",
                date_options,
                format_func=lambda d: d.strftime("%b %d, %Y"),
                key="health_del_date",
            )
            if st.button("Delete", type="secondary", key="health_del_btn"):
                delete_health_row(del_date)
                st.rerun()

    # ── Weekly averages ───────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Weekly Averages")

    if df.empty:
        st.info("Add entries to see weekly averages.")
    else:
        weekly = compute_weekly_averages(df)
        display_weekly = weekly.copy()
        display_weekly.columns = ["Week", "Days", "Avg Weight (kg)", "Avg Protein (g)", "Avg Calories", "Avg Sleep (hrs)"]
        display_weekly.index = range(1, len(display_weekly) + 1)
        st.dataframe(
            display_weekly,
            use_container_width=True,
            column_config={
                "Avg Weight (kg)": st.column_config.NumberColumn(format="%.1f"),
                "Avg Protein (g)": st.column_config.NumberColumn(format="%.0f"),
                "Avg Calories": st.column_config.NumberColumn(format="%.0f"),
                "Avg Sleep (hrs)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

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

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<div class="chart-container"><span style="font-weight:600;font-size:0.95rem;">Weight Trend</span></div>',
                unsafe_allow_html=True,
            )
            st.line_chart(chart_df["weight"], color="#667eea", height=280)

        with c2:
            st.markdown(
                '<div class="chart-container"><span style="font-weight:600;font-size:0.95rem;">Sleep Pattern</span></div>',
                unsafe_allow_html=True,
            )
            st.line_chart(chart_df["sleep"], color="#6C5CE7", height=280)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown(
                '<div class="chart-container"><span style="font-weight:600;font-size:0.95rem;">Protein Intake</span></div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(chart_df["protein"], color="#4ECDC4", height=280)

        with c4:
            st.markdown(
                '<div class="chart-container"><span style="font-weight:600;font-size:0.95rem;">Calorie Intake</span></div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(chart_df["calories"], color="#FF6B6B", height=280)
