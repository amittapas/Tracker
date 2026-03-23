from typing import Optional

import html
import hmac
import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import pytz

TIMEZONE = pytz.timezone("America/Los_Angeles")


def _get_app_password() -> Optional[str]:
    try:
        return str(st.secrets["app"]["password"])
    except (KeyError, TypeError):
        try:
            return str(st.secrets["app_password"])
        except (KeyError, TypeError):
            return None


def _password_matches(entered: str, expected: str) -> bool:
    if not expected:
        return False
    try:
        return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))
    except Exception:
        return False


def trigger_celebration():
    """Call before st.rerun() on successes; confetti plays on the next run."""
    st.session_state["celebrate"] = True


def show_confetti():
    """Celebration effect — native Streamlit balloons (works reliably; iframe confetti often blocked)."""
    st.balloons()


def render_password_gate():
    st.title("Tracker")
    st.caption("Enter the app password to continue.")
    expected = _get_app_password()
    if not expected:
        st.error(
            "Password not configured. Add `[app]` with `password = \"...\"` to `.streamlit/secrets.toml` "
            "(or `app_password = \"...\"`)."
        )
        return
    with st.form("app_login", clear_on_submit=False):
        pwd = st.text_input("Password", type="password", key="gate_password", autocomplete="current-password")
        if st.form_submit_button("Continue"):
            if _password_matches(pwd, expected):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")


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


# ── NFP (streak timer: anchor + resets) ───────────────────────────────────────

def _ensure_nfp_schema():
    """Create NFP tables if missing (same as migrations/004_nfp_streak.sql) so Cloud works without manual SQL."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS nfp_streak (
                  id INTEGER PRIMARY KEY CHECK (id = 1),
                  epoch_started_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS nfp_relapse (
                  id SERIAL PRIMARY KEY,
                  relapsed_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nfp_relapse_at ON nfp_relapse (relapsed_at)")
        conn.commit()


def _nfp_default_epoch() -> datetime:
    """Anchor streak at 4:00 PM local (Pacific). If before 4pm today, anchor is today 4pm (streak 0 until then)."""
    now = datetime.now(TIMEZONE)
    today_4pm = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return today_4pm


def _ensure_nfp_streak_row():
    _ensure_nfp_schema()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM nfp_streak WHERE id = 1")
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO nfp_streak (id, epoch_started_at) VALUES (1, %s)",
                    (_nfp_default_epoch(),),
                )
        conn.commit()


def load_nfp_epoch() -> datetime:
    _ensure_nfp_streak_row()
    with get_conn() as conn:
        df = pd.read_sql("SELECT epoch_started_at FROM nfp_streak WHERE id = 1", conn)
    t = df.iloc[0]["epoch_started_at"]
    if hasattr(t, "tzinfo") and t.tzinfo is None:
        t = pytz.utc.localize(t).astimezone(TIMEZONE)
    elif hasattr(t, "tz_convert"):
        t = t.tz_convert(TIMEZONE)
    return t


def load_nfp_relapses() -> list:
    _ensure_nfp_schema()
    with get_conn() as conn:
        df = pd.read_sql("SELECT relapsed_at FROM nfp_relapse ORDER BY relapsed_at ASC", conn)
    if df.empty:
        return []
    out = []
    for x in df["relapsed_at"]:
        if hasattr(x, "tzinfo") and x.tzinfo is None:
            x = pytz.utc.localize(x).astimezone(TIMEZONE)
        elif hasattr(x, "tz_convert"):
            x = x.tz_convert(TIMEZONE)
        out.append(x)
    return out


def record_nfp_relapse():
    _ensure_nfp_schema()
    now = datetime.now(TIMEZONE)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO nfp_relapse (relapsed_at) VALUES (%s)", (now,))
        conn.commit()


def nfp_current_streak_seconds(epoch: datetime, relapses: list, now: datetime) -> float:
    """Time since last segment start (epoch or most recent reset)."""
    starts = [epoch] + sorted(relapses)
    last_start = starts[-1]
    return max(0.0, (now - last_start).total_seconds())


def _format_streak_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    s = int(seconds)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _ts_local(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        return t.tz_localize(TIMEZONE)
    return t.tz_convert(TIMEZONE)


def _nfp_segments_data(epoch, relapses: list, now) -> list:
    """Ordered segments (start, end, duration_sec, ongoing)."""
    rel_sorted = sorted(relapses)
    starts = [_ts_local(epoch)] + [_ts_local(r) for r in rel_sorted]
    now_ts = _ts_local(now)
    out = []
    for i, seg_start in enumerate(starts):
        seg_end = starts[i + 1] if i + 1 < len(starts) else now_ts
        if seg_end <= seg_start:
            continue
        dur = (seg_end - seg_start).total_seconds()
        out.append(
            {
                "start": seg_start,
                "end": seg_end,
                "duration_sec": dur,
                "ongoing": i == len(starts) - 1,
            }
        )
    return out


def _nfp_timeline_fig(epoch, relapses: list, now: datetime):
    """Single-row calendar view: green spans = clean periods, red ticks = resets."""
    rel_sorted = sorted(relapses)
    starts = [_ts_local(epoch)] + [_ts_local(r) for r in rel_sorted]
    now_ts = _ts_local(now)
    fig, ax = plt.subplots(figsize=(9, 2.0))
    colors = ("#a7f3d0", "#86efac")
    idx = 0
    drew = False
    for i, seg_start in enumerate(starts):
        seg_end = starts[i + 1] if i + 1 < len(starts) else now_ts
        if seg_end <= seg_start:
            continue
        ax.axvspan(seg_start, seg_end, ymin=0.15, ymax=0.85, color=colors[idx % 2], alpha=0.92, linewidth=0)
        idx += 1
        drew = True
    for r in rel_sorted:
        ax.axvline(r, color="#dc2626", linewidth=2.0, alpha=0.85, linestyle="-")
    if not drew:
        ax.text(0.5, 0.5, "Nothing to show yet.", ha="center", va="center", transform=ax.transAxes, color="#64748b")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Timeline — green = on track, red line = reset", fontsize=11, fontweight="bold", color="#1f2937", pad=10, loc="left")
    pad = timedelta(hours=1)
    ax.set_xlim(_ts_local(epoch) - pad, now_ts + pad)
    _style_ax(ax, fig)
    return fig


def _nfp_segment_bars_fig(segments: list):
    """Horizontal bars: duration of each segment (hours); ongoing segment updates live."""
    if not segments:
        fig, ax = plt.subplots(figsize=(9, 2.0))
        ax.text(0.5, 0.5, "No segments yet.", ha="center", va="center", transform=ax.transAxes, color="#64748b")
        ax.set_xticks([])
        ax.set_yticks([])
        _style_ax(ax, fig)
        return fig

    hours = [s["duration_sec"] / 3600.0 for s in segments]
    labels = []
    for i, s in enumerate(segments):
        tag = f"#{i + 1}"
        if s["ongoing"]:
            tag += " · current"
        labels.append(tag)

    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.45 * len(segments))))
    y_pos = list(range(len(segments)))
    colors_b = ["#7c3aed" if s["ongoing"] else "#94a3b8" for s in segments]
    ax.barh(y_pos, hours, color=colors_b, height=0.65, alpha=0.9, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color="#475569")
    ax.invert_yaxis()
    ax.set_xlabel("Duration (hours)", fontsize=9, color="#6b7280")
    ax.set_title("Length of each segment (until next reset)", fontsize=11, fontweight="bold", color="#1f2937", pad=10, loc="left")
    span = max(hours) if hours else 1.0
    ax.set_xlim(0, span * 1.18)
    for i, h in enumerate(hours):
        ax.text(min(h + span * 0.02, span * 1.14), i, f"{h:.1f} h", va="center", fontsize=8, color="#64748b")
    _style_ax(ax, fig)
    return fig


def _nfp_sawtooth_fig(epoch: datetime, relapses: list, now: datetime):
    """Streak length (seconds) vs time: ramps up, vertical drop on each reset."""
    rel_sorted = sorted(relapses)
    starts = [_ts_local(epoch)] + [_ts_local(r) for r in rel_sorted]
    now_ts = _ts_local(now)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    color = "#7c3aed"
    fill_a = 0.08
    drew = False

    for i, seg_start in enumerate(starts):
        seg_end = starts[i + 1] if i + 1 < len(starts) else now_ts
        if seg_end <= seg_start:
            continue
        n = int((seg_end - seg_start).total_seconds() // 40)
        n = min(500, max(30, n))
        ts = pd.date_range(start=seg_start, end=seg_end, periods=n)
        t0 = seg_start
        ys = (ts - t0).total_seconds()
        ax.plot(ts, ys, color=color, linewidth=2.1, solid_capstyle="round")
        ax.fill_between(ts, ys, color=color, alpha=fill_a)
        drew = True
        if i + 1 < len(starts):
            rnext = starts[i + 1]
            peak = (rnext - t0).total_seconds()
            ax.plot([rnext, rnext], [peak, 0.0], color=color, linewidth=2.1, solid_capstyle="round")

    if not drew:
        ax.text(
            0.5,
            0.5,
            "No streak interval to plot yet (anchor may be in the future).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#64748b",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Streak over time (seconds)", fontsize=11, fontweight="bold", color="#1f2937", pad=10, loc="left")
    ax.set_xlabel("Time", fontsize=9, color="#6b7280")
    ax.set_ylabel("Current streak (seconds)", fontsize=9, color="#6b7280")
    _style_ax(ax, fig)
    return fig


@st.fragment(run_every=timedelta(seconds=2))
def _render_nfp_live_block():
    """Metrics + sawtooth chart; auto-refreshes so the current segment keeps climbing in real time."""
    epoch = load_nfp_epoch()
    relapses = load_nfp_relapses()
    now = datetime.now(TIMEZONE)

    streak_sec = nfp_current_streak_seconds(epoch, relapses, now)
    segments = _nfp_segments_data(epoch, relapses, now)
    longest_sec = max((s["duration_sec"] for s in segments), default=0.0)
    total_clean_sec = sum(s["duration_sec"] for s in segments)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current streak", _format_streak_duration(streak_sec))
    m2.metric("Resets logged", len(relapses))
    m3.metric("Anchor started", epoch.strftime("%b %d, %Y  %-I:%M %p"))

    s1, s2, s3 = st.columns(3)
    s1.metric("Longest segment", _format_streak_duration(longest_sec))
    s2.metric("Total on-track time", _format_streak_duration(total_clean_sec))
    s3.metric("Segments (runs)", len(segments))

    if longest_sec > 0:
        ratio = min(1.0, streak_sec / longest_sec)
        st.progress(ratio, text=f"Current streak vs longest: {ratio * 100:.0f}%")

    if epoch > now:
        st.info(
            f"Streak anchor is **{epoch.strftime('%b %d, %Y  %-I:%M %p')}** — your streak stays at 0 until that time."
        )

    st.markdown("---")
    st.subheader("Streak history (live)")
    st.caption(
        "Sawtooth: elapsed seconds in the current segment; the right edge rises until the next reset. "
        f"Refreshes every 2s · **{now.strftime('%H:%M:%S')}**"
    )

    fig = _nfp_sawtooth_fig(epoch, relapses, now)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Other views")
    st.caption("Same data — timeline shows *when* you were on track; bars compare *how long* each run lasted (hours).")

    c_timeline, c_bars = st.columns(2)
    with c_timeline:
        st.pyplot(_nfp_timeline_fig(epoch, relapses, now), clear_figure=True)
    with c_bars:
        st.pyplot(_nfp_segment_bars_fig(segments), clear_figure=True)


def render_nfp_section():
    st.header("NFP")
    st.caption(
        "Your streak is how long you’ve stayed on track since the anchor time or your last reset. "
        f"Times use **{TIMEZONE.zone}**."
    )

    with st.form("nfp_relapse_form"):
        st.markdown("**Reset streak** — records this moment and restarts your streak from now (the chart drops to zero).")
        confirm = st.checkbox("I understand this resets my current streak")
        submitted = st.form_submit_button("Record reset", type="secondary")
        if submitted:
            if not confirm:
                st.error("Please confirm that you understand this resets your streak.")
            else:
                record_nfp_relapse()
                st.toast("Reset recorded — streak restarted.", icon="🔄")
                st.rerun()

    st.markdown("---")
    _render_nfp_live_block()

    relapses = load_nfp_relapses()
    if relapses:
        with st.expander("Reset log (newest first)"):
            rdf = pd.DataFrame({"Reset at (local)": [r.strftime("%b %d, %Y  %-I:%M:%S %p") for r in reversed(relapses)]})
            st.dataframe(rdf, use_container_width=True, hide_index=True)


# ── Goals (pushups, running, pages, steps, writing, fasting, passive income) ─

GOAL_METRICS = {
    "pushups": {"label": "Pushups", "unit": "reps", "start": 15, "delta": 3, "format": "{:.0f}"},
    "running": {"label": "Running", "unit": "min", "start": 10, "delta": 1, "format": "{:.0f}"},
    "pages": {"label": "Pages read (session goal)", "unit": "pages", "start": 10, "delta": 1, "format": "{:.0f}"},
    "steps": {"label": "Daily steps", "unit": "steps", "start": 5000, "delta": 100, "format": "{:.0f}"},
    "writing": {"label": "Writing", "unit": "words", "start": 100, "delta": 50, "format": "{:.0f}"},
    "fasting": {"label": "Fasting", "unit": "hrs", "start": 3, "delta": 1, "format": "{:.0f}"},
    "passive_income": {"label": "Passive income (monthly)", "unit": "$", "start": 100, "delta": 10, "format": "{:.0f}"},
}

GOAL_ICONS = {
    "pushups": "💪",
    "running": "🏃",
    "pages": "📖",
    "steps": "👟",
    "writing": "✍️",
    "fasting": "⏱️",
    "passive_income": "💰",
}

GOAL_SHORT_TITLES = {
    "pushups": "Pushups",
    "running": "Running",
    "pages": "Pages / session",
    "steps": "Daily steps",
    "writing": "Writing",
    "fasting": "Fasting",
    "passive_income": "Passive (mo.)",
}

# Per-goal accent: main hue, soft chip behind icon, hairline border
GOAL_STYLE = {
    "pushups": {"accent": "#e11d48", "chip": "#fff1f2", "ring": "rgba(225, 29, 72, 0.12)"},
    "running": {"accent": "#2563eb", "chip": "#eff6ff", "ring": "rgba(37, 99, 235, 0.12)"},
    "pages": {"accent": "#c2410c", "chip": "#fffbeb", "ring": "rgba(194, 65, 12, 0.1)"},
    "steps": {"accent": "#059669", "chip": "#ecfdf5", "ring": "rgba(5, 150, 105, 0.12)"},
    "writing": {"accent": "#7c3aed", "chip": "#f5f3ff", "ring": "rgba(124, 58, 237, 0.12)"},
    "fasting": {"accent": "#0e7490", "chip": "#ecfeff", "ring": "rgba(14, 116, 144, 0.12)"},
    "passive_income": {"accent": "#a16207", "chip": "#fefce8", "ring": "rgba(161, 98, 7, 0.12)"},
}


def _goal_compact_card_html(metric_key: str, cfg: dict, goals: dict) -> str:
    g = goals.get(metric_key, {"current_target": cfg["start"], "max_achieved": 0.0})
    cur_t = g["current_target"]
    max_t = g["max_achieved"]
    vt = html.escape(_fmt_goal_value(metric_key, cur_t))
    vm = html.escape(_fmt_goal_value(metric_key, max_t))
    title = html.escape(GOAL_SHORT_TITLES.get(metric_key, cfg["label"]))
    cap = html.escape(_goals_delta_caption(metric_key, cfg))
    icon = GOAL_ICONS.get(metric_key, "🎯")
    stl = GOAL_STYLE.get(metric_key, {"accent": "#475569", "chip": "#f1f5f9", "ring": "rgba(71, 85, 105, 0.12)"})
    ac, chip, ring = stl["accent"], stl["chip"], stl["ring"]
    return f"""
<div style="font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;
            background:linear-gradient(180deg,#ffffff 0%,#fafbfc 100%);
            border:1px solid #e8ecf1;border-radius:16px;padding:0.75rem 0.85rem 0.8rem;margin-bottom:0.35rem;
            box-shadow:0 1px 2px rgba(15,23,42,0.04),0 8px 24px -8px rgba(15,23,42,0.08);">
  <div style="display:flex;align-items:flex-start;gap:0.65rem;">
    <div style="flex-shrink:0;width:2.65rem;height:2.65rem;border-radius:14px;
                background:{chip};box-shadow:inset 0 0 0 1px {ring};
                display:flex;align-items:center;justify-content:center;font-size:1.45rem;line-height:1;">{icon}</div>
    <div style="flex:1;min-width:0;padding-top:0.05rem;">
      <div style="color:#0f172a;font-size:0.92rem;font-weight:600;letter-spacing:-0.02em;line-height:1.25;">{title}</div>
      <div style="color:#64748b;font-size:0.72rem;margin-top:0.2rem;line-height:1.35;">{cap}</div>
      <div style="display:flex;margin-top:0.65rem;border-radius:12px;overflow:hidden;
                  background:#f4f6f9;box-shadow:inset 0 0 0 1px #e8ecf1;">
        <div style="flex:1;padding:0.5rem 0.55rem;min-width:0;border-right:1px solid #e2e8f0;">
          <div style="color:#64748b;font-size:0.65rem;font-weight:500;letter-spacing:0.02em;">Personal best</div>
          <div style="color:{ac};font-size:1.2rem;font-weight:700;letter-spacing:-0.03em;
                      font-variant-numeric:tabular-nums;line-height:1.2;margin-top:0.15rem;">{vm}</div>
        </div>
        <div style="flex:1;padding:0.5rem 0.55rem;min-width:0;background:rgba(255,255,255,0.55);">
          <div style="color:#64748b;font-size:0.65rem;font-weight:500;letter-spacing:0.02em;">Next target</div>
          <div style="color:#1e293b;font-size:1.05rem;font-weight:600;letter-spacing:-0.02em;
                      font-variant-numeric:tabular-nums;line-height:1.2;margin-top:0.15rem;">{vt}</div>
        </div>
      </div>
    </div>
  </div>
</div>
"""


def _ensure_goal_rows():
    with get_conn() as conn:
        with conn.cursor() as cur:
            for key, cfg in GOAL_METRICS.items():
                cur.execute(
                    """
                    INSERT INTO user_goals (metric_key, current_target, max_achieved)
                    VALUES (%s, %s, 0)
                    ON CONFLICT (metric_key) DO NOTHING
                    """,
                    (key, cfg["start"]),
                )
        conn.commit()


def load_goals() -> dict:
    _ensure_goal_rows()
    with get_conn() as conn:
        df = pd.read_sql("SELECT metric_key, current_target, max_achieved FROM user_goals", conn)
    out = {}
    for _, r in df.iterrows():
        out[r["metric_key"]] = {
            "current_target": float(r["current_target"]),
            "max_achieved": float(r["max_achieved"]),
        }
    return out


def apply_goal_success(metric_key: str):
    if metric_key not in GOAL_METRICS:
        return
    cfg = GOAL_METRICS[metric_key]
    delta = cfg["delta"]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_target, max_achieved FROM user_goals WHERE metric_key = %s",
                (metric_key,),
            )
            row = cur.fetchone()
            if not row:
                return
            current_target, max_achieved = float(row[0]), float(row[1])
            new_max = max(max_achieved, current_target)
            new_target = current_target + delta
            cur.execute(
                """
                UPDATE user_goals
                SET current_target = %s, max_achieved = %s
                WHERE metric_key = %s
                """,
                (new_target, new_max, metric_key),
            )
        conn.commit()


def apply_goal_failure(metric_key: str):
    """No change to targets — placeholder for future logging."""
    pass


def _fmt_goal_value(metric_key: str, val: float) -> str:
    if metric_key == "passive_income":
        return f"${val:.0f}"
    cfg = GOAL_METRICS[metric_key]
    return f"{cfg['format'].format(val)} {cfg['unit']}"


def _goals_delta_caption(metric_key: str, cfg: dict) -> str:
    if metric_key == "passive_income":
        return f"On success: +${cfg['delta']:.0f}"
    if metric_key == "fasting":
        return f"On success: +{cfg['delta']:.0f} hr"
    return f"On success: +{cfg['delta']:.0f} {cfg['unit']}"


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
            st.toast(f"Saved entry for {entry_date}.", icon="✅")
            trigger_celebration()
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
                st.toast(f"Logged {book.strip()} — page {end_page}", icon="📖")
                trigger_celebration()
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


def render_goals_section():
    st.markdown(
        """
        <div style="font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;
                    margin-bottom:1rem;padding-bottom:0.85rem;border-bottom:1px solid #e8ecf1;">
            <h2 style="margin:0;color:#0f172a;font-size:1.35rem;font-weight:700;letter-spacing:-0.03em;">Goals</h2>
            <p style="margin:0.35rem 0 0 0;color:#64748b;font-size:0.82rem;line-height:1.45;max-width:36rem;">
                Log how it went: <span style="color:#2563eb;font-weight:600;">Achieved</span> nudges your next target up;
                <span style="color:#475569;font-weight:600;">Failed</span> keeps it steady.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    goals = load_goals()
    items = list(GOAL_METRICS.items())

    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(items):
                break
            mk, cfg = items[idx]
            with col:
                st.markdown(_goal_compact_card_html(mk, cfg, goals), unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Achieved", type="primary", key=f"goal_ok_{mk}", use_container_width=True):
                        apply_goal_success(mk)
                        st.toast("Nice — target bumped!", icon="🎯")
                        trigger_celebration()
                        st.rerun()
                with b2:
                    if st.button("Failed", key=f"goal_fail_{mk}", use_container_width=True):
                        apply_goal_failure(mk)
                        st.toast("Target unchanged.")


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
                st.toast(f"You slept {fmt_duration(dur)}", icon="☀️")
                trigger_celebration()
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
st.set_page_config(page_title="Tracker", page_icon=":material/monitoring:", layout="wide")

if not st.session_state.get("authenticated"):
    render_password_gate()
    st.stop()

if st.session_state.pop("celebrate", False):
    show_confetti()

with st.sidebar:
    if st.button("Log out", key="app_logout"):
        st.session_state["authenticated"] = False
        st.rerun()

pg = st.navigation(
    {
        "Tracker": [
            st.Page(render_health_section, title="Health", icon=":material/favorite:"),
            st.Page(render_sleep_section, title="Sleep", icon=":material/bedtime:"),
            st.Page(render_reading_section, title="Reading", icon=":material/menu_book:"),
            st.Page(render_goals_section, title="Goals", icon=":material/flag:"),
            st.Page(render_nfp_section, title="NFP", icon=":material/timer:"),
            st.Page(render_graphs_section, title="Trends", icon=":material/bar_chart:"),
        ],
    }
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1000px; padding-top: 2rem; }
    .stApp { background-color: #f8fafc; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
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
