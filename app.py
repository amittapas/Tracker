import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, datetime

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "Health"


@st.cache_resource
def get_gspread_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(spreadsheet, title, headers):
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=len(headers))
        ws.append_row(headers)
    return ws


def load_health_data(ws) -> pd.DataFrame:
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame(columns=["date", "weight", "protein", "calories", "sleep"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["protein"] = pd.to_numeric(df["protein"], errors="coerce")
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
    df["sleep"] = pd.to_numeric(df["sleep"], errors="coerce")
    return df


def save_health_row(ws, entry: dict):
    ws.append_row([entry["date"], entry["weight"], entry["protein"], entry["calories"], entry["sleep"]])


def update_health_row(ws, row_idx: int, entry: dict):
    ws.update(f"A{row_idx}:E{row_idx}", [[entry["date"], entry["weight"], entry["protein"], entry["calories"], entry["sleep"]]])


def delete_health_row(ws, row_idx: int):
    ws.delete_rows(row_idx)


def find_row_by_date(ws, target_date: str) -> int | None:
    """Returns 1-based row index (header is row 1)."""
    cells = ws.col_values(1)
    for i, val in enumerate(cells):
        if val == target_date:
            return i + 1
    return None


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

# ── Connect to Google Sheets ──────────────────────────────────────────────────
try:
    client = get_gspread_client()
    spreadsheet = client.open(st.secrets["spreadsheet_name"])
except Exception as e:
    st.error(f"Could not connect to Google Sheets. Check your secrets.\n\n`{e}`")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_health, tab_analytics = st.tabs(["🏋️ Health", "📈 Analytics"])

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_health:
    HEALTH_HEADERS = ["date", "weight", "protein", "calories", "sleep"]
    ws_health = get_or_create_worksheet(spreadsheet, SHEET_NAME, HEALTH_HEADERS)
    df = load_health_data(ws_health)

    # ── Add / Edit entry ──────────────────────────────────────────────────────
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
            existing_row = find_row_by_date(ws_health, str(entry_date))
            if existing_row:
                update_health_row(ws_health, existing_row, entry)
                st.success(f"Updated entry for {entry_date}.")
            else:
                save_health_row(ws_health, entry)
                st.success(f"Added entry for {entry_date}.")
            st.cache_resource.clear()
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
                row_idx = find_row_by_date(ws_health, str(del_date))
                if row_idx:
                    delete_health_row(ws_health, row_idx)
                st.cache_resource.clear()
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
    ws_health = get_or_create_worksheet(spreadsheet, SHEET_NAME, ["date", "weight", "protein", "calories", "sleep"])
    df = load_health_data(ws_health)

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
