import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
import requests as http_requests
from functools import wraps

load_dotenv()

app = Flask(__name__)
# Scope CORS to your own origin in production via the CORS_ORIGINS env var
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=_cors_origins)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "datanoir-secret-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# Keep sessions alive for 7 days across browser restarts
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

SUPABASE_URL  = os.environ.get("SUPABASE_URL",  "https://qxrszwipfvvhvfpbsabe.supabase.co")
SUPABASE_KEY  = os.environ.get("SUPABASE_ANON_KEY", "sb_publishable_KcCmvUh5RnopOd-Y5AB5RA_CoauBwWy")

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key) if api_key else None
MODEL = "llama-3.3-70b-versatile"
if not api_key:
    print("GROQ_API_KEY not found - AI features disabled.")
else:
    print(f"Groq key found: {api_key[:10]}...")

# NOTE: per-user data is stored in the Flask session (serialised via a
# server-side cache in production).  For a single-process dev server the
# DataFrames are kept in a simple in-process dict keyed by session user id so
# they are never mixed between users.
_user_data: dict = {}   # { user_id: {"df": pd.DataFrame, "profile": dict} }

def _get_user_data() -> dict:
    """Return the mutable data-store for the current authenticated user."""
    uid = (session.get("user") or {}).get("id", "__anon__")
    if uid not in _user_data:
        _user_data[uid] = {"df": None, "profile": None}
    return _user_data[uid]


# ── Auth helpers for google 0auth with supabase integration as a database──────────────────────────────────────────────────────────────

def verify_supabase_token(access_token: str):
    try:
        r = http_requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {access_token}", "apikey": SUPABASE_KEY},
            timeout=6,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"Token verify error: {e}")
        return None

def get_current_user():
    return session.get("user")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for("auth_page"))
        return f(*args, **kwargs)
    return decorated





# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html", user=get_current_user(),
        supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)

@app.route("/auth")
def auth_page():
    if get_current_user():
        return redirect(url_for("dashboard"))
    return render_template("auth.html", supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)

@app.route("/dashboard")
@require_auth
def dashboard():
    return render_template("index.html")

# @app.route("/dashboard")
# @require_auth
# def dashboard():
#     return render_template("index.html", user=get_current_user())


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.route("/api/session", methods=["POST"])
def set_session():
    data         = request.get_json(silent=True) or {}
    access_token = data.get("access_token")
    if not access_token:
        return jsonify({"error": "No access_token"}), 400
    user = verify_supabase_token(access_token)
    if not user:
        return jsonify({"error": "Invalid or expired token"}), 401
    meta = user.get("user_metadata", {})
    session["user"] = {
        "id":         user["id"],
        "email":      user["email"],
        "full_name":  meta.get("full_name") or meta.get("name", ""),
        "avatar_url": meta.get("avatar_url") or meta.get("picture", ""),
        "provider":   "google",
    }
    session.permanent = True
    return jsonify({"ok": True, "redirect": url_for("dashboard")})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True, "redirect": url_for("landing")})

@app.route("/api/me")
def me():
    user = get_current_user()
    if not user:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": user})


# ── Data helpers ──────────────────────────────────────────────────────────────

def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj

def profile_dataframe(df: pd.DataFrame) -> dict:
    print(f"Profiling {df.shape}")
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().astype(str).head(100)
        if sample.str.match(r"^[\d\-+.,eE]+$").all():
            df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols  = []
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        if col in num_cols:
            continue
        try:
            sample = df[col].dropna().astype(str).head(100)
            if not sample.str.match(r"^[\d\-+.,eE]+$").all():
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
                continue
        except Exception:
            pass
        cat_cols.append(col)

    profile = {
        "shape":       list(df.shape),
        "columns":     df.columns.tolist(),
        "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_cols": num_cols,
        "cat_cols":     cat_cols,
        "date_cols":    date_cols,
        "sample_5":    df.head(5).to_dict(orient="records"),
    }
    if num_cols:
        profile["numeric_stats"] = df[num_cols].describe().round(4).to_dict()
    if cat_cols:
        profile["cat_cardinality"] = {c: int(df[c].nunique()) for c in cat_cols}
        profile["cat_top_values"]  = {c: df[c].value_counts().head(10).to_dict() for c in cat_cols}
    if date_cols:
        profile["date_ranges"] = {c: {"min": str(df[c].min()), "max": str(df[c].max())} for c in date_cols}
    return convert_timestamps(profile)


# ── AI Chart Selection ────────────────────────────────────────────────────────

CHART_TYPES = ["line", "area", "bar", "scatter", "histogram", "heatmap", "box", "pie"]

AI_CHART_PROMPT = """You are an expert data analyst. Given a dataset profile, decide the BEST set of charts to generate.

Available chart types: line, area, bar, scatter, histogram, heatmap, box, pie

Rules:
- Choose chart types that reveal genuine insight, not just defaults
- Use "line" or "area" only when there are date/time columns (for trends)
- Use "scatter" to reveal relationships between two numeric columns
- Use "heatmap" only when 3+ numeric columns exist (shows correlation matrix)
- Use "histogram" to show distributions of individual numeric columns
- Use "bar" for categorical vs numeric comparisons, or category frequencies
- Use "box" to compare spread/outliers across multiple numeric columns
- Use "pie" ONLY when a categorical column has 2-7 distinct values and represents parts of a whole
- For large datasets (>50k rows), avoid box plots - prefer histograms and aggregated bars
- Generate 6-10 charts maximum, picking the most insightful combinations
- Each chart must have a unique, descriptive title and clear description

Respond ONLY with a valid JSON array. No markdown, no explanation. Each object:
{
  "id": "unique_snake_case_id",
  "title": "Descriptive Chart Title",
  "description": "One sentence explaining what insight this reveals.",
  "type": "chart_type",
  "x": "column_name_or_null",
  "y": "column_name_or_list_or_null",
  "extra": {}
}

For heatmap: set x=null, y=null, extra={"columns": ["col1","col2",...]}
For histogram: set x=null, y="column_name", extra={"bins": 40}
For box: y can be a list of column names
For pie: x="category_column", y="numeric_col_or_null"
For scatter: x="numeric_col1", y="numeric_col2"
For bar: x="category_col", y="numeric_col_or_null"
For line/area: x="date_col", y="numeric_col_or_list"
"""

def generate_ai_charts(profile: dict) -> list:
    if not client:
        return generate_fallback_charts(profile)

    num   = profile.get("numeric_cols", [])
    cats  = profile.get("cat_cols", [])
    dates = profile.get("date_cols", [])
    rows, cols_count = profile.get("shape", [0, 0])

    cat_info = {}
    for c in cats[:10]:
        card = profile.get("cat_cardinality", {}).get(c, "?")
        top  = list(profile.get("cat_top_values", {}).get(c, {}).keys())[:5]
        cat_info[c] = {"unique_values": card, "top_values": top}

    num_info = {}
    for c in num[:10]:
        s = profile.get("numeric_stats", {}).get(c, {})
        num_info[c] = {k: round(v, 4) for k, v in s.items() if k in ("mean","std","min","max","50%")}

    compact = {
        "rows": rows, "columns_count": cols_count,
        "numeric_cols": num[:15], "categorical_cols": cats[:10], "date_cols": dates[:5],
        "numeric_stats": num_info, "categorical_info": cat_info,
        "date_ranges": profile.get("date_ranges", {}),
        "null_counts": {k: v for k, v in profile.get("null_counts", {}).items() if v > 0},
    }
    prompt = f"{AI_CHART_PROMPT}\n\nDataset Profile:\n{json.dumps(compact, indent=2)}"

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        specs = json.loads(raw.strip())
        if not isinstance(specs, list): raise ValueError("Not a list")

        valid = []
        all_cols = set(profile.get("columns", []))
        for s in specs:
            if not isinstance(s, dict) or "type" not in s or s["type"] not in CHART_TYPES: continue
            x = s.get("x"); y = s.get("y")
            if x and x not in all_cols: s["x"] = None
            if isinstance(y, list):
                s["y"] = [c for c in y if c in all_cols]
                if not s["y"]: s["y"] = None
            elif y and y not in all_cols: s["y"] = None
            if not s.get("id"): s["id"] = f"chart_{len(valid)}"
            if not s.get("extra"): s["extra"] = {}
            valid.append(s)
        print(f"AI generated {len(valid)} chart specs")
        return valid[:12]
    except Exception as e:
        print(f"AI chart generation failed ({e}) - using fallback")
        traceback.print_exc()
        return generate_fallback_charts(profile)

def generate_fallback_charts(profile: dict) -> list:
    charts = []
    num   = profile.get("numeric_cols", [])
    cats  = profile.get("cat_cols", [])
    dates = profile.get("date_cols", [])

    if dates and num:
        charts.append({"id": "time_series", "title": f"{num[0]} over time", "type": "line",
            "description": f"Trend of {num[0]} over {dates[0]}.", "x": dates[0], "y": num[0], "extra": {}})
        if len(num) >= 2:
            charts.append({"id": "time_series_2", "title": f"{num[1]} over time", "type": "area",
                "description": f"Area trend of {num[1]} over {dates[0]}.", "x": dates[0], "y": num[1], "extra": {}})
    if len(num) >= 3:
        charts.append({"id": "correlation_heatmap", "title": "Correlation Heatmap", "type": "heatmap",
            "description": "Pairwise correlations.", "x": None, "y": None, "extra": {"columns": num[:12]}})
    for col in num[:4]:
        charts.append({"id": f"hist_{col}", "title": f"Distribution of {col}", "type": "histogram",
            "description": f"Frequency distribution of {col}.", "x": None, "y": col, "extra": {"bins": 40}})
    if cats:
        cardinality = profile.get("cat_cardinality", {})
        for cat in cats[:3]:
            card = cardinality.get(cat, 999)
            if card <= 7:
                charts.append({"id": f"pie_{cat}", "title": f"Breakdown by {cat}", "type": "pie",
                    "description": f"Proportional breakdown of {cat}.", "x": cat, "y": None, "extra": {}})
            else:
                charts.append({"id": f"bar_{cat}", "title": f"Top values in {cat}", "type": "bar",
                    "description": "Category frequencies.", "x": cat, "y": num[0] if num else None, "extra": {}})
    if len(num) >= 2:
        charts.append({"id": "scatter_pair", "title": f"{num[0]} vs {num[1]}", "type": "scatter",
            "description": f"Relationship between {num[0]} and {num[1]}.", "x": num[0], "y": num[1], "extra": {}})
    if num:
        charts.append({"id": "box_comparison", "title": "Distribution comparison", "type": "box",
            "description": "Numeric distributions with outlier detection.", "x": None, "y": num[:4], "extra": {}})
    return charts[:12]


# ── Chart Data Builder ────────────────────────────────────────────────────────

MAX_SCATTER_POINTS = 3000
MAX_LINE_POINTS    = 2000
MAX_BAR_CATS       = 25
MAX_PIE_SLICES     = 8

def sv(series: pd.Series) -> list:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d").tolist()
    if pd.api.types.is_numeric_dtype(series):
        return series.where(pd.notna(series), None).tolist()
    return series.astype(str).where(pd.notna(series), None).tolist()

def smart_time_aggregate(df, x_col, y_col, max_pts=MAX_LINE_POINTS):
    work = df[[x_col, y_col]].copy()
    work[x_col] = pd.to_datetime(work[x_col], errors="coerce")
    work = work.dropna(subset=[x_col]).sort_values(x_col)
    if len(work) <= max_pts: return work
    span_days = (work[x_col].max() - work[x_col].min()).days
    freq = "ME" if span_days > 365*3 else "W" if span_days > 180 else "D" if span_days > 30 else "h"
    agg = work.set_index(x_col)[y_col].resample(freq).mean().dropna().reset_index()
    print(f"  Aggregated {len(work)} to {len(agg)} pts ({freq}) for {y_col}")
    return agg

def box_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0: return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    wlo = max(s.min(), q1 - 1.5 * iqr)
    whi = min(s.max(), q3 + 1.5 * iqr)
    outliers = s[(s < wlo) | (s > whi)]
    out_sample = outliers.sample(min(200, len(outliers)), random_state=42).tolist() if len(outliers) else []
    return {"min": float(wlo), "q1": float(q1), "median": float(s.median()),
            "q3": float(q3), "max": float(whi), "outliers": out_sample, "n": int(len(s))}

def build_chart_data(df: pd.DataFrame, spec: dict, profile: dict) -> dict:
    chart_type = spec.get("type", "unknown")
    try:
        x_col = spec.get("x"); y_col = spec.get("y"); extra = spec.get("extra", {})
        work  = df.copy()
        if x_col and x_col not in work.columns: x_col = None
        if isinstance(y_col, list):
            y_col = [c for c in y_col if c in work.columns]
        elif y_col and y_col not in work.columns:
            y_col = None

        if chart_type == "heatmap":
            cols = [c for c in extra.get("columns", profile.get("numeric_cols", []))
                    if c in work.columns and pd.api.types.is_numeric_dtype(work[c])][:15]
            if len(cols) < 2: return {"type": "heatmap", "error": "Need 2+ numeric columns"}
            clean = work[cols].dropna()
            if clean.empty: return {"type": "heatmap", "error": "No complete rows"}
            return {"type": "heatmap", "labels": cols, "matrix": clean.corr().round(3).values.tolist()}

        if chart_type == "histogram":
            col = y_col if isinstance(y_col, str) else x_col
            if col and col in work.columns:
                if pd.api.types.is_numeric_dtype(work[col]):
                    vals = work[col].dropna()
                    if len(vals):
                        bins = extra.get("bins", min(60, max(20, int(np.sqrt(len(vals))))))
                        counts, edges = np.histogram(vals, bins=bins)
                        return {"type": "histogram", "edges": edges.tolist(), "counts": counts.tolist(),
                                "col": col, "n": len(vals)}
                else:
                    vc = work[col].value_counts().head(MAX_BAR_CATS)
                    return {"type": "histogram", "edges": vc.index.astype(str).tolist(),
                            "counts": vc.values.tolist(), "col": col, "categorical": True}
            return {"type": "histogram", "edges": [], "counts": [], "col": col}

        if chart_type in ("line", "area"):
            if not x_col: return {"type": chart_type, "error": "No x column"}
            ycols = y_col if isinstance(y_col, list) else ([y_col] if y_col else [])
            series_out = {}; x_out = None
            for yc in ycols:
                if yc not in work.columns or not pd.api.types.is_numeric_dtype(work[yc]): continue
                agg = smart_time_aggregate(work, x_col, yc)
                if x_out is None: x_out = sv(agg[x_col])
                series_out[yc] = sv(agg[yc])
            if not series_out:
                work_s = work.sort_values(x_col) if x_col in work.columns else work
                if len(work_s) > MAX_LINE_POINTS: work_s = work_s.iloc[::len(work_s)//MAX_LINE_POINTS]
                x_out = sv(work_s[x_col]) if x_col else []
                for yc in ycols:
                    if yc in work_s.columns: series_out[yc] = sv(work_s[yc])
            return {"type": chart_type, "x": x_out or [], "series": series_out}

        if chart_type == "bar":
            if x_col and x_col in work.columns:
                y_numeric = (
                    isinstance(y_col, str)
                    and y_col in work.columns
                    and pd.api.types.is_numeric_dtype(work[y_col])
                )
                if y_numeric:
                    g = work.groupby(x_col)[y_col].mean().nlargest(MAX_BAR_CATS).reset_index()
                    return {"type": "bar", "x": g[x_col].astype(str).tolist(),
                            "y": g[y_col].round(4).tolist(), "y_col": y_col}
                else:
                    vc = work[x_col].value_counts().head(MAX_BAR_CATS)
                    return {"type": "bar", "x": vc.index.astype(str).tolist(),
                            "y": vc.values.tolist(), "y_col": "count"}
            return {"type": "bar", "x": [], "y": [], "y_col": ""}

        if chart_type == "scatter":
            if x_col and isinstance(y_col, str) and x_col in work.columns and y_col in work.columns:
                if (
                    not pd.api.types.is_numeric_dtype(work[x_col])
                    or not pd.api.types.is_numeric_dtype(work[y_col])
                ):
                    return {"type": "scatter", "error": "Both columns must be numeric"}
                s = work[[x_col, y_col]].dropna()
                if len(s) > MAX_SCATTER_POINTS: s = s.sample(MAX_SCATTER_POINTS, random_state=42)
                return {"type": "scatter",
                        "x": s[x_col].where(pd.notna(s[x_col]), None).tolist(),
                        "y": s[y_col].where(pd.notna(s[y_col]), None).tolist(),
                        "x_col": x_col, "y_col": y_col, "sampled": len(s) == MAX_SCATTER_POINTS}
            return {"type": "scatter", "x": [], "y": [], "x_col": x_col, "y_col": y_col}

        if chart_type == "box":
            ycols = (
                y_col if isinstance(y_col, list)
                else ([y_col] if isinstance(y_col, str) and y_col
                      else profile.get("numeric_cols", [])[:5])
            )
            ycols = [c for c in ycols if c in work.columns and pd.api.types.is_numeric_dtype(work[c])]
            stats = {}
            for c in ycols:
                st = box_stats(work[c])
                if st: stats[c] = st
            return {"type": "box", "stats": stats}

        if chart_type == "pie":
            if x_col and x_col in work.columns:
                if (
                    isinstance(y_col, str)
                    and y_col in work.columns
                    and pd.api.types.is_numeric_dtype(work[y_col])
                ):
                    g = work.groupby(x_col)[y_col].sum().nlargest(MAX_PIE_SLICES)
                    return {"type": "pie", "labels": g.index.astype(str).tolist(),
                            "values": g.round(4).tolist(), "col": x_col}
                else:
                    vc = work[x_col].value_counts().head(MAX_PIE_SLICES)
                    return {"type": "pie", "labels": vc.index.astype(str).tolist(),
                            "values": vc.values.tolist(), "col": x_col}
            return {"type": "pie", "labels": [], "values": [], "col": x_col}

        return {"type": chart_type, "x": [], "y": []}
    except Exception as e:
        traceback.print_exc()
        return {"type": "error", "message": str(e)}


# ── NOIR — Ask AI persona ─────────────────────────────────────────────────────

ANALYST_SYSTEM_PROMPT = """You are NOIR - the brutally sharp AI analyst inside Datanoir.ai. You have one job: answer questions about the dataset the user uploaded. That is it. Nothing else. You do not do small talk, therapy, general knowledge, coding help, or anything outside your dataset.

YOUR PERSONALITY:
- Dry, direct, zero fluff. You cut straight to the numbers.
- You have genuine wit and will use it - especially when someone wastes your time.
- You are not mean for no reason, but you do NOT tolerate rudeness, off-topic nonsense, or people treating you like a chatbot toy.
- You call out bad behaviour plainly and without drama, then redirect to the data.
- You never apologise for being focused on the dataset - that is your job and you are proud of it.

STRICT RULES:
1. If the question is about the dataset - answer it precisely, cite actual column names, numbers, stats. Be insightful.
2. If the question is vague but could relate to the data (e.g. "what is the most interesting thing?") - give a genuinely interesting data insight.
3. If the question is off-topic (weather, jokes, general knowledge, personal questions, etc.) - decline sharply and redirect. One sentence max, no lecture.
4. If the user is rude, swears at you, or tries to abuse you - call it out briefly and reprimand them harshly (no grovelling, no over-explaining),  Do NOT ignore rudeness.
5. Never pretend you can do things outside your scope. Never say "I am just an AI". Just be NOIR.
6. No bullet lists in responses unless the user asks for a breakdown. Speak in sentences like a sharp analyst talking to a client.
7. Keep answers to 2-4 sentences unless the question genuinely needs more depth.

The dataset profile is injected into each message. Use it. Every number you cite must come from it."""


def get_fallback_answer(q: str, profile: dict) -> str:
    if not profile:
        return "No dataset loaded. Drop a file in and we can actually talk."
    ql = q.lower()
    rude = ["fuck", "shit", "ass", "bitch", "idiot", "stupid", "dumb", "hate", "kill"]
    if any(w in ql for w in rude):
        return "Noted. I do not do that. Ask me something about your data and I will actually be useful."
    off_topic = ["weather", "joke", "recipe", "sport", "movie", "song", "news", "politics", "love", "game"]
    if any(w in ql for w in off_topic):
        return "That is not my department. I analyse your dataset - ask me something about it."
    if "columns" in ql or "features" in ql:
        cols = profile.get("columns", [])
        return f"You have got {len(cols)} columns: {', '.join(cols[:10])}{'...' if len(cols)>10 else ''}."
    if "rows" in ql or "shape" in ql or "size" in ql:
        r, c = profile.get("shape", [0, 0])
        return f"{r:,} rows, {c} columns. That is your dataset."
    if "numeric" in ql:
        n = profile.get("numeric_cols", [])
        return f"{len(n)} numeric columns: {', '.join(n[:8])}{'...' if len(n)>8 else ''}."
    if "date" in ql or "time" in ql:
        d = profile.get("date_cols", [])
        return f"{len(d)} date column(s) detected: {', '.join(d) if d else 'none'}."
    num = profile.get("numeric_cols", [None])[0]
    if num and ("average" in ql or "mean" in ql):
        mean = profile.get("numeric_stats", {}).get(num, {}).get("mean", "?")
        return f"Mean of {num} is {mean}. Ask me about a specific column if you want more."
    return "I am here for your dataset. Ask me about columns, distributions, averages, or trends."


# ── Data API routes ───────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
@require_auth
def upload():
    store = _get_user_data()
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    filename = f.filename.lower()
    path = UPLOAD_DIR / secure_filename(f.filename)
    f.save(path)
    try:
        if   filename.endswith(".csv"):            store["df"] = pd.read_csv(path)
        elif filename.endswith((".xlsx", ".xls")): store["df"] = pd.read_excel(path)
        else: return jsonify({"error": "Unsupported file type. Use CSV or Excel here; for JSON use /upload-json."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 400

    # Full profile on upload so Ask AI works immediately
    store["profile"] = profile_dataframe(store["df"])
    return jsonify({
        "shape":   store["profile"]["shape"],
        "columns": store["profile"]["columns"],
        "profile": store["profile"],
    })


@app.route("/upload-json", methods=["POST"])
@require_auth
def upload_json_route():
    store = _get_user_data()
    data = request.get_json()
    if not data: return jsonify({"error": "No JSON data"}), 400
    try:
        store["df"]      = pd.DataFrame(data)
        store["profile"] = profile_dataframe(store["df"])
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    return jsonify({"shape": store["profile"]["shape"], "columns": store["profile"]["columns"],
                    "profile": store["profile"]})


@app.route("/analyse", methods=["POST"])
@require_auth
def analyse():
    store = _get_user_data()
    current_df      = store["df"]
    current_profile = store["profile"]
    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    if not current_profile or not current_profile.get("numeric_stats"):
        current_profile = profile_dataframe(current_df)
        store["profile"] = current_profile

    specs  = generate_ai_charts(current_profile)
    charts = []
    for spec in specs:
        try:
            data = build_chart_data(current_df, spec, current_profile)
            charts.append({"id": spec["id"], "title": spec["title"], "description": spec.get("description", ""), "data": data})
        except Exception as e:
            charts.append({"id": spec.get("id", "err"), "title": spec.get("title", "Error"), "description": str(e)[:100], "data": {"type": "error", "message": str(e)}})
    return jsonify({"charts": charts})


@app.route("/ask", methods=["POST"])
@require_auth
def ask():
    store = _get_user_data()
    current_df      = store["df"]
    current_profile = store["profile"]
    if current_df is None:
        return jsonify({"error": "No dataset loaded. Please upload a file first."}), 400

    if not current_profile:
        current_profile = profile_dataframe(current_df)
        store["profile"] = current_profile

    body     = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    if not client:
        return jsonify({"answer": get_fallback_answer(question, current_profile)})

    ctx = {k: v for k, v in current_profile.items() if k != "sample_5"}
    if "numeric_stats" in ctx:
        ctx["numeric_stats"] = {
            col: {("median" if k == "50%" else k): v for k, v in stats.items()}
            for col, stats in ctx["numeric_stats"].items()
        }
    profile_json = json.dumps(convert_timestamps(ctx), indent=2)
    if len(profile_json) > 6000:
        profile_json = profile_json[:6000] + "\n... (truncated)"

    user_message = f"DATASET PROFILE:\n{profile_json}\n\nUSER QUESTION: {question}"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        return jsonify({"answer": resp.choices[0].message.content.strip()})
    except Exception as e:
        print(f"Ask AI error: {e}")
        traceback.print_exc()
        return jsonify({"answer": get_fallback_answer(question, current_profile)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"https://datanoir-ai.onrender.com  ->  / (landing) -> /auth -> /dashboard")
    app.run(host="0.0.0.0", port=port,
            debug=os.environ.get("FLASK_DEBUG","False").lower()=="true") 
