import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import pytz

TIMEZONE = pytz.timezone("America/Los_Angeles")


def get_conn():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require",
    )


# ── Health DB helpers ─────────────────────────────────────────────────────────

def load_health_data() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT date, weight, protein, calories, sleep, steps FROM health ORDER BY date",
            conn,
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def upsert_health_row(entry: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO health (date, weight, protein, calories, sleep, steps)
                VALUES (%(date)s, %(weight)s, %(protein)s, %(calories)s, %(sleep)s, %(steps)s)
                ON CONFLICT (date) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    protein = EXCLUDED.protein,
                    calories = EXCLUDED.calories,
                    sleep = EXCLUDED.sleep,
                    steps = EXCLUDED.steps
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
            steps=("steps", "mean"),
            days_logged=("date", "count"),
        )
        .reset_index()
    )
    weekly["week"] = weekly.apply(
        lambda r: f"{r['week_start'].strftime('%b %d')} – {r['week_end'].strftime('%b %d')}", axis=1
    )
    weekly = weekly.sort_values("week_start", ascending=False)
    return weekly[["week", "days_logged", "weight", "protein", "calories", "sleep", "steps"]]


CHART_BG = "#fafbfc"

def _style_ax(ax, fig):
    ax.set_facecolor(CHART_BG)
    ax.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="x", labelsize=8, colors="#6b7280")
    ax.tick_params(axis="y", labelsize=8, colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("white")
    fig.autofmt_xdate()
    plt.tight_layout(pad=1.2)


def _static_line_fig(dates, y, *, color="#2563eb", title=""):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1f2937", pad=10, loc="left")
    if len(dates) <= 25:
        ax.plot(dates, y, color=color, linewidth=2.2, marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1.2)
    else:
        ax.plot(dates, y, color=color, linewidth=2.2)
    ax.fill_between(dates, y, alpha=0.07, color=color)
    _style_ax(ax, fig)
    return fig


def _static_sleep_lines_fig(dates, daily, avg_7d):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.set_title("Sleep (hrs)", fontsize=11, fontweight="bold", color="#1f2937", pad=10, loc="left")
    if len(dates) <= 25:
        ax.plot(dates, daily, color="#2563eb", linewidth=2.2, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=1.2, label="Daily")
    else:
        ax.plot(dates, daily, color="#2563eb", linewidth=2.2, label="Daily")
    ax.plot(dates, avg_7d, color="#f59e0b", linewidth=2, linestyle="--", label="7-day avg")
    ax.fill_between(dates, daily, alpha=0.06, color="#2563eb")
    ax.legend(loc="best", fontsize=8, framealpha=0.9, edgecolor="#e5e7eb")
    _style_ax(ax, fig)
    return fig


# ── Reading DB helpers ────────────────────────────────────────────────────────

def load_reading_data() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT date, book, end_page FROM reading ORDER BY date, book",
            conn,
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def insert_reading_row(entry: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reading (date, book, end_page)
                VALUES (%(date)s, %(book)s, %(end_page)s)
                """,
                entry,
            )
        conn.commit()


def delete_reading_row(row_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reading WHERE id = %s", (row_id,))
        conn.commit()


def load_reading_data_with_id() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT id, date, book, end_page FROM reading ORDER BY date DESC, book",
            conn,
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_pages_per_day() -> pd.DataFrame:
    """For each book, compute pages read per day using the difference in end_page between consecutive entries."""
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT date, book, end_page FROM reading ORDER BY book, date",
            conn,
        )
    if df.empty:
        return pd.DataFrame(columns=["date", "pages"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["end_page"] = pd.to_numeric(df["end_page"], errors="coerce").astype("float64")

    rows = []
    for _, group in df.groupby("book"):
        group = group.sort_values("date")
        pages_read = group["end_page"].diff()
        for d, p in zip(group["date"], pages_read):
            if pd.notna(p) and p >= 0:
                rows.append({"date": d, "pages": p})

    if not rows:
        return pd.DataFrame(columns=["date", "pages"])
    result = pd.DataFrame(rows)
    result = result.groupby("date", as_index=False)["pages"].sum()
    result = result.sort_values("date")
    return result


# ── Sleep DB helpers ──────────────────────────────────────────────────────────

def get_open_sleep_session():
    """Get a sleep session where user said 'sleeping' but hasn't said 'waking up' yet."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, sleep_at FROM sleep_log WHERE wake_at IS NULL ORDER BY sleep_at DESC LIMIT 1")
            row = cur.fetchone()
    return row


def start_sleep():
    now = datetime.now(TIMEZONE)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sleep_log (sleep_at) VALUES (%s)", (now,))
        conn.commit()
    return now


def end_sleep(session_id: int):
    now = datetime.now(TIMEZONE)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT sleep_at FROM sleep_log WHERE id = %s", (session_id,))
            sleep_at = cur.fetchone()[0]
            duration = (now - sleep_at).total_seconds() / 3600
            cur.execute(
                "UPDATE sleep_log SET wake_at = %s, duration_hrs = %s WHERE id = %s",
                (now, round(duration, 2), session_id),
            )
        conn.commit()
    return now, duration


def load_sleep_data() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT sl.id, sl.sleep_at, sl.wake_at, sl.duration_hrs,
                   COALESCE(d.disturbances, 0) AS disturbances
            FROM sleep_log sl
            LEFT JOIN (
                SELECT sleep_log_id, COUNT(*) AS disturbances
                FROM sleep_disturbance GROUP BY sleep_log_id
            ) d ON d.sleep_log_id = sl.id
            WHERE sl.wake_at IS NOT NULL
            ORDER BY sl.sleep_at DESC
            """,
            conn,
        )
    if not df.empty:
        df["sleep_at"] = pd.to_datetime(df["sleep_at"]).dt.tz_convert(TIMEZONE)
        df["wake_at"] = pd.to_datetime(df["wake_at"]).dt.tz_convert(TIMEZONE)
    return df


def log_disturbance(session_id: int, note: str = ""):
    now = datetime.now(TIMEZONE)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sleep_disturbance (sleep_log_id, disturbed_at, note) VALUES (%s, %s, %s)",
                (session_id, now, note or None),
            )
        conn.commit()
    return now


def get_disturbance_count(session_id: int) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sleep_disturbance WHERE sleep_log_id = %s", (session_id,))
            return cur.fetchone()[0]


def delete_sleep_row(sleep_at_val):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sleep_log WHERE sleep_at = %s", (sleep_at_val,))
        conn.commit()


def fmt_time(ts):
    return ts.strftime("%-I:%M %p")


def fmt_duration(hrs):
    h = int(hrs)
    m = int((hrs - h) * 60)
    return f"{h}h {m}m"


def render_health_section():
    df = load_health_data()

    with st.expander("**Add new entry**", expanded=True):
        row1 = st.columns(3)
        with row1[0]:
            entry_date = st.date_input("Date", value=date.today(), key="health_date")
        with row1[1]:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1, key="health_weight")
        with row1[2]:
            protein = st.number_input("Protein (g)", min_value=0, max_value=1000, value=120, step=1, key="health_protein")

        row2 = st.columns([1, 1, 1, 0.6])
        with row2[0]:
            calories = st.number_input("Calories", min_value=0, max_value=10000, value=2000, step=50, key="health_calories")
        with row2[1]:
            sleep = st.number_input("Sleep (hrs)", min_value=0.0, max_value=24.0, value=7.5, step=0.25, key="health_sleep")
        with row2[2]:
            steps = st.number_input("Steps (day)", min_value=0, max_value=100000, value=0, step=100, key="health_steps")
        with row2[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add", type="primary", use_container_width=True, key="health_add")

        if add_btn:
            entry = {
                "date": str(entry_date),
                "weight": weight,
                "protein": protein,
                "calories": calories,
                "sleep": sleep,
                "steps": int(steps),
            }
            upsert_health_row(entry)
            st.success(f"Saved entry for {entry_date}.")
            st.rerun()

    st.subheader("Daily Log")

    if df.empty:
        st.info("No entries yet. Add your first entry above!")
    else:
        display_df = df.sort_values("date", ascending=False).reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df.columns = ["Date", "Weight (kg)", "Protein (g)", "Calories", "Sleep (hrs)", "Steps (day)"]
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

    st.subheader("Weekly Averages")

    if df.empty:
        st.info("Add entries to see weekly averages.")
    else:
        weekly = compute_weekly_averages(df)
        display_weekly = weekly.copy()
        display_weekly["steps"] = display_weekly["steps"].round(0)
        display_weekly.columns = [
            "Week",
            "Days Logged",
            "Avg Weight (kg)",
            "Avg Protein (g)",
            "Avg Calories",
            "Avg Sleep (hrs)",
            "Avg Steps",
        ]
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
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Avg Weight", f"{current_week['weight'].mean():.1f} kg")
            m2.metric("Avg Protein", f"{current_week['protein'].mean():.0f} g")
            m3.metric("Avg Calories", f"{current_week['calories'].mean():.0f}")
            m4.metric("Avg Sleep", f"{current_week['sleep'].mean():.1f} hrs")
            m5.metric("Avg Steps", f"{current_week['steps'].mean():.0f}")


def render_reading_section():
    st.subheader("Reading Log")

    with st.expander("**Log a reading session**", expanded=True):
        row1 = st.columns([2, 1.5, 0.6])
        with row1[0]:
            book = st.text_input("Book title", key="reading_book")
        with row1[1]:
            end_page = st.number_input("End page", min_value=0, max_value=99999, value=0, step=1, key="reading_end_page")
        with row1[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add", type="primary", use_container_width=True, key="reading_add")

        reading_date = st.date_input("Date", value=date.today(), key="reading_date")

        if add_btn:
            if not book.strip():
                st.warning("Please enter a book title.")
            else:
                insert_reading_row({
                    "date": str(reading_date),
                    "book": book.strip(),
                    "end_page": int(end_page),
                })
                st.success(f"Logged **{book.strip()}** — page {end_page} on {reading_date}.")
                st.rerun()

    rdf = load_reading_data_with_id()

    if rdf.empty:
        st.info("No reading entries yet. Log your first session above!")
    else:
        display_df = rdf[["date", "book", "end_page"]].copy()
        display_df.columns = ["Date", "Book", "End Page"]
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True, height=min(len(display_df) * 40 + 60, 500))

        with st.expander("Delete an entry"):
            del_options = rdf.apply(
                lambda r: f"{r['date'].strftime('%Y-%m-%d')} — {r['book']} (p.{r['end_page']})", axis=1
            ).tolist()
            del_idx = st.selectbox("Select entry to delete", range(len(del_options)), format_func=lambda i: del_options[i], key="reading_del")
            if st.button("Delete", type="secondary", key="reading_del_btn"):
                delete_reading_row(int(rdf.iloc[del_idx]["id"]))
                st.rerun()


def render_graphs_section():
    gdf = load_health_data()
    st.header("Trends")

    if gdf.empty:
        st.info("No entries yet. Add health data in the **Health** section to see graphs.")
    else:
        chart_df = gdf.copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])
        chart_df = chart_df.sort_values("date")
        for col in ("weight", "protein", "calories", "sleep", "steps"):
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce").astype("float64")

        dates = chart_df["date"]
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(_static_line_fig(dates, chart_df["weight"], title="Weight (kg)"), clear_figure=True)
        with g2:
            st.pyplot(_static_line_fig(dates, chart_df["protein"], color="#7c3aed", title="Protein (g)"), clear_figure=True)

        g3, g4 = st.columns(2)
        with g3:
            st.pyplot(_static_line_fig(dates, chart_df["calories"], color="#059669", title="Calories"), clear_figure=True)
        with g4:
            avg7 = chart_df["sleep"].rolling(7, min_periods=1).mean()
            st.pyplot(_static_sleep_lines_fig(dates, chart_df["sleep"], avg7), clear_figure=True)

        g5, g6 = st.columns(2)
        with g5:
            st.pyplot(_static_line_fig(dates, chart_df["steps"], color="#ca8a04", title="Steps"), clear_figure=True)

        ppd = compute_pages_per_day()
        with g6:
            if ppd.empty:
                st.caption("Log reading sessions (at least two entries per book) to see pages/day.")
            else:
                ppd_dates = pd.to_datetime(ppd["date"])
                st.pyplot(_static_line_fig(ppd_dates, ppd["pages"], color="#be185d", title="Pages Read / Day"), clear_figure=True)


def render_sleep_section():
    now = datetime.now(TIMEZONE)
    open_session = get_open_sleep_session()

    # ── Trigger buttons ───────────────────────────────────────────────────────
    st.subheader("Sleep Tracker")
    st.caption(f"Current time: **{now.strftime('%b %d, %Y  %-I:%M %p')}**")

    if open_session:
        sleep_at = open_session[1]
        if sleep_at.tzinfo is None:
            sleep_at = pytz.utc.localize(sleep_at)
        sleep_at = sleep_at.astimezone(TIMEZONE)
        elapsed = (now - sleep_at).total_seconds() / 3600
        dist_count = get_disturbance_count(open_session[0])
        dist_label = f" — {dist_count} disturbance{'s' if dist_count != 1 else ''}" if dist_count else ""
        st.info(f"Sleeping since **{fmt_time(sleep_at)}** ({fmt_duration(elapsed)} ago){dist_label}")

    col_sleep, col_disturb, col_wake = st.columns(3)

    with col_sleep:
        if open_session:
            st.button("Going to sleep", disabled=True, use_container_width=True, key="sleep_btn")
        else:
            if st.button("🌙 Going to sleep", type="primary", use_container_width=True, key="sleep_btn"):
                ts = start_sleep()
                st.success(f"Sleep started at {fmt_time(ts)}. Good night!")
                st.rerun()

    with col_disturb:
        if open_session:
            if st.button("⚡ Disturbance", use_container_width=True, key="disturb_btn"):
                ts = log_disturbance(open_session[0])
                st.warning(f"Disturbance logged at {fmt_time(ts)}.")
                st.rerun()
        else:
            st.button("Disturbance", disabled=True, use_container_width=True, key="disturb_btn")

    with col_wake:
        if open_session:
            if st.button("☀️ Waking up", type="primary", use_container_width=True, key="wake_btn"):
                ts, dur = end_sleep(open_session[0])
                st.success(f"Woke up at {fmt_time(ts)}. You slept **{fmt_duration(dur)}**.")
                st.rerun()
        else:
            st.button("Waking up", disabled=True, use_container_width=True, key="wake_btn")

    # ── Sleep stats ───────────────────────────────────────────────────────────
    sleep_df = load_sleep_data()

    if sleep_df.empty:
        st.info("No completed sleep sessions yet. Use the buttons above to start tracking!")
    else:
        st.markdown("---")
        st.subheader("Sleep Stats")

        avg_duration = sleep_df["duration_hrs"].mean()
        avg_bedtime_minutes = sleep_df["sleep_at"].apply(
            lambda t: t.hour * 60 + t.minute if t.hour >= 12 else (t.hour + 24) * 60 + t.minute
        ).mean()
        avg_bed_h = int(avg_bedtime_minutes // 60) % 24
        avg_bed_m = int(avg_bedtime_minutes % 60)
        avg_bedtime_str = datetime(2000, 1, 1, avg_bed_h, avg_bed_m).strftime("%-I:%M %p")

        avg_wake_minutes = sleep_df["wake_at"].apply(lambda t: t.hour * 60 + t.minute).mean()
        avg_wake_h = int(avg_wake_minutes // 60)
        avg_wake_m = int(avg_wake_minutes % 60)
        avg_waketime_str = datetime(2000, 1, 1, avg_wake_h, avg_wake_m).strftime("%-I:%M %p")

        avg_disturbances = sleep_df["disturbances"].mean()
        undisturbed_nights = (sleep_df["disturbances"] == 0).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Duration", fmt_duration(avg_duration))
        m2.metric("Avg Bedtime", avg_bedtime_str)
        m3.metric("Avg Wake Time", avg_waketime_str)

        m4, m5, m6 = st.columns(3)
        m4.metric("Avg Disturbances", f"{avg_disturbances:.1f}/night")
        m5.metric("Undisturbed Nights", f"{undisturbed_nights}/{len(sleep_df)}")
        m6.metric("Total Nights", str(len(sleep_df)))

        # ── Last 7 days ──────────────────────────────────────────────────────
        week_ago = now - timedelta(days=7)
        recent = sleep_df[sleep_df["sleep_at"] >= week_ago]

        if len(recent) >= 2:
            st.markdown("#### Last 7 Days")
            r1, r2, r3, r4 = st.columns(4)

            recent_avg = recent["duration_hrs"].mean()
            best = recent["duration_hrs"].max()
            worst = recent["duration_hrs"].min()
            recent_dist = recent["disturbances"].sum()

            r1.metric("Avg Duration", fmt_duration(recent_avg))
            r2.metric("Best Night", fmt_duration(best))
            r3.metric("Worst Night", fmt_duration(worst))
            r4.metric("Total Disturbances", str(int(recent_dist)))

            if len(recent) >= 3:
                std = recent["duration_hrs"].std()
                if std < 0.5:
                    consistency = "Excellent"
                elif std < 1.0:
                    consistency = "Good"
                elif std < 1.5:
                    consistency = "Fair"
                else:
                    consistency = "Poor"
                st.metric("Sleep Consistency", consistency)

        # ── Sleep log table ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Sleep Log")

        log_display = sleep_df.copy()
        log_display["Night of"] = log_display["sleep_at"].dt.strftime("%b %d, %Y")
        log_display["Bedtime"] = log_display["sleep_at"].apply(fmt_time)
        log_display["Wake Time"] = log_display["wake_at"].apply(fmt_time)
        log_display["Duration"] = log_display["duration_hrs"].apply(fmt_duration)
        log_display["Disturbances"] = log_display["disturbances"].astype(int)
        log_display = log_display[["Night of", "Bedtime", "Wake Time", "Duration", "Disturbances"]].reset_index(drop=True)
        log_display.index = log_display.index + 1
        st.dataframe(log_display, use_container_width=True)

        with st.expander("Delete a sleep entry"):
            del_options = sleep_df["sleep_at"].tolist()
            del_sleep = st.selectbox(
                "Select night to delete",
                del_options,
                format_func=lambda t: t.strftime("%b %d, %Y  %-I:%M %p"),
                key="sleep_del",
            )
            if st.button("Delete", type="secondary", key="sleep_del_btn"):
                delete_sleep_row(del_sleep)
                st.rerun()


# ── Navigation & page config ──────────────────────────────────────────────────
pg = st.navigation(
    {
        "Tracker": [
            st.Page(render_health_section, title="Health", icon=":material/favorite:"),
            st.Page(render_sleep_section, title="Sleep", icon=":material/bedtime:"),
            st.Page(render_reading_section, title="Reading", icon=":material/menu_book:"),
            st.Page(render_graphs_section, title="Trends", icon=":material/bar_chart:"),
        ],
    }
)

st.set_page_config(page_title="Tracker", page_icon=":material/monitoring:", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1000px; padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-size: 13px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0f172a !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.logo(":material/monitoring:", size="large")

pg.run()
