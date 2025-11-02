# app.py
import streamlit as st
import pandas as pd
import sqlite3
import io
import uuid
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import json

PRIMARY_COLOR = "#007AFF"
st.set_page_config(page_title="ClarityOS", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}
    .stButton>button {{
        background:{PRIMARY_COLOR};
        color:white;
        border-radius:8px;
        border:none;
    }}
    .metric-card {{
        background:white;
        border:1px solid rgba(0,0,0,0.03);
        border-radius:16px;
        padding:14px 16px 6px 16px;
        box-shadow:0 10px 25px rgba(0,0,0,0.02);
    }}
    .insight {{
        background:rgba(0,122,255,0.07);
        border-left:4px solid {PRIMARY_COLOR};
        padding:10px 12px;
        border-radius:6px;
        margin-bottom:8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = "clarityos.db"

# ---------------------- DB ----------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # core tables
    c.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id TEXT PRIMARY KEY,
        email TEXT,
        name TEXT,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS workspace (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        name TEXT,
        created_at TEXT
    )""")

    # data_source без category
    c.execute("""
    CREATE TABLE IF NOT EXISTS data_source (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        type TEXT,
        title TEXT,
        source_url TEXT,
        status TEXT,
        created_at TEXT,
        updated_at TEXT
    )""")

    # миграция: category
    c.execute("PRAGMA table_info(data_source)")
    cols = [r[1] for r in c.fetchall()]
    if "category" not in cols:
        c.execute("ALTER TABLE data_source ADD COLUMN category TEXT DEFAULT NULL")

    c.execute("""
    CREATE TABLE IF NOT EXISTS data_upload (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        original_filename TEXT,
        storage_path TEXT,
        detected_schema TEXT,
        rows_count INTEGER,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS mapping_profile (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        target_table TEXT,
        mapping_json TEXT,
        status TEXT,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS "order" (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        external_id TEXT,
        order_date TEXT,
        customer_name TEXT,
        product TEXT,
        revenue REAL,
        channel TEXT,
        customer_id TEXT,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS expense (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        expense_date TEXT,
        category TEXT,
        amount REAL,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS customer (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        name TEXT,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS metrics_snapshot (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        period_from TEXT,
        period_to TEXT,
        payload_json TEXT,
        created_at TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS insight (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        metrics_snapshot_id TEXT,
        text TEXT,
        rule_code TEXT,
        created_at TEXT
    )""")

    # demo user/workspace
    c.execute("SELECT COUNT(*) FROM user")
    if c.fetchone()[0] == 0:
        user_id = str(uuid.uuid4())
        ws_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        c.execute("INSERT INTO user (id, email, name, created_at) VALUES (?,?,?,?)",
                  (user_id, "demo@clarityos.app", "Demo User", now))
        c.execute("INSERT INTO workspace (id, owner_id, name, created_at) VALUES (?,?,?,?)",
                  (ws_id, user_id, "Demo workspace", now))
    conn.commit()
    conn.close()

init_db()

# ------------------ helpers ------------------
def list_workspaces():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name FROM workspace ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def create_workspace(name: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM user LIMIT 1")
    owner_id = c.fetchone()[0]
    ws_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    c.execute("INSERT INTO workspace (id, owner_id, name, created_at) VALUES (?,?,?,?)",
              (ws_id, owner_id, name, now))
    conn.commit()
    conn.close()
    return ws_id

def parse_google_sheet_to_csv_url(url: str):
    if "docs.google.com/spreadsheets" not in url:
        return None
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    sheet_id = parts[3]
    qs = parse_qs(parsed.fragment)
    gid = qs.get("gid", ["0"])[0]
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def suggest_mapping(detected_cols):
    orders_targets = [
        {"target":"order_id","label":"ID заказа","required":True,"synonyms":["id","order","orderno","номер","id заказа"]},
        {"target":"order_date","label":"Дата заказа","required":True,"synonyms":["date","order_date","дата"]},
        {"target":"customer_name","label":"Клиент","required":True,"synonyms":["client","customer","клиент","name"]},
        {"target":"product","label":"Продукт","required":True,"synonyms":["product","товар","course","позиция"]},
        {"target":"revenue","label":"Выручка","required":True,"synonyms":["revenue","amount","sum","доход","выручка","price"]},
        {"target":"channel","label":"Канал","required":False,"synonyms":["utm_source","channel","канал"]},
    ]
    expenses_targets = [
        {"target":"expense_date","label":"Дата расхода","required":True,"synonyms":["date","дата"]},
        {"target":"category","label":"Категория","required":True,"synonyms":["category","категория","type"]},
        {"target":"amount","label":"Сумма","required":True,"synonyms":["amount","sum","сумма","cost","расход"]},
    ]

    def find_suggest(syns):
        for col in detected_cols:
            cl = col.lower().strip()
            for s in syns:
                if s in cl:
                    return col
        return None

    for t in orders_targets:
        t["suggested_column"] = find_suggest(t["synonyms"])
    for t in expenses_targets:
        t["suggested_column"] = find_suggest(t["synonyms"])

    return {"orders": orders_targets, "expenses": expenses_targets}

def apply_mapping_to_df(df: pd.DataFrame, mapping: dict, target_table: str):
    out = {}
    for target, source in mapping.items():
        if source and source in df.columns:
            out[target] = df[source]
        else:
            out[target] = None
    out_df = pd.DataFrame(out)

    if target_table == "orders":
        req = ["order_id","order_date","customer_name","product","revenue"]
        for col in req:
            if col not in out_df.columns:
                raise ValueError(f"{col} is required")
        out_df["order_date"] = pd.to_datetime(out_df["order_date"], errors="coerce")
        out_df = out_df[out_df["order_date"].notna()]
        out_df["revenue"] = (
            out_df["revenue"].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        )
        out_df["revenue"] = pd.to_numeric(out_df["revenue"], errors="coerce").fillna(0.0)
    else:
        req = ["expense_date","category","amount"]
        for col in req:
            if col not in out_df.columns:
                raise ValueError(f"{col} is required")
        out_df["expense_date"] = pd.to_datetime(out_df["expense_date"], errors="coerce")
        out_df = out_df[out_df["expense_date"].notna()]
        out_df["amount"] = (
            out_df["amount"].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        )
        out_df["amount"] = pd.to_numeric(out_df["amount"], errors="coerce").fillna(0.0)
    return out_df

def insert_orders(df: pd.DataFrame, ds_id: str):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    for _, row in df.iterrows():
        c.execute(
            """INSERT INTO "order"
            (id, data_source_id, external_id, order_date, customer_name, product, revenue, channel, customer_id, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                ds_id,
                str(row["order_id"]),
                row["order_date"].date().isoformat(),
                str(row["customer_name"]),
                str(row["product"]),
                float(row["revenue"]),
                str(row.get("channel") if pd.notna(row.get("channel")) else None),
                None,
                now,
            ),
        )
    conn.commit()
    conn.close()

def insert_expenses(df: pd.DataFrame, ds_id: str):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    for _, row in df.iterrows():
        c.execute(
            """INSERT INTO expense
            (id, data_source_id, expense_date, category, amount, created_at)
            VALUES (?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                ds_id,
                row["expense_date"].date().isoformat(),
                str(row["category"]),
                float(row["amount"]),
                now,
            ),
        )
    conn.commit()
    conn.close()

def rebuild_customers(ws_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT o.customer_name
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (ws_id,))
    names = [r[0] for r in c.fetchall() if r[0]]
    c.execute("SELECT name FROM customer WHERE workspace_id = ?", (ws_id,))
    existing = {r[0] for r in c.fetchall()}
    now = datetime.utcnow().isoformat()
    for name in names:
        if name not in existing:
            c.execute("INSERT INTO customer (id, workspace_id, name, created_at) VALUES (?,?,?,?)",
                      (str(uuid.uuid4()), ws_id, name, now))
    conn.commit()
    conn.close()

def calc_metrics(ws_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT o.order_date, o.revenue
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (ws_id,))
    orders_df = pd.DataFrame(c.fetchall(), columns=["order_date","revenue"])
    if not orders_df.empty:
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])

    c.execute("""
        SELECT e.expense_date, e.amount
        FROM expense e
        JOIN data_source ds ON ds.id = e.data_source_id
        WHERE ds.workspace_id = ?
    """, (ws_id,))
    exp_df = pd.DataFrame(c.fetchall(), columns=["expense_date","amount"])
    if not exp_df.empty:
        exp_df["expense_date"] = pd.to_datetime(exp_df["expense_date"])

    revenue = float(orders_df["revenue"].sum()) if not orders_df.empty else 0.0
    expenses = float(exp_df["amount"].sum()) if not exp_df.empty else 0.0
    profit = revenue - expenses
    margin = profit / revenue if revenue > 0 else None

    # avg check
    c.execute("""
        SELECT COUNT(DISTINCT o.external_id)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (ws_id,))
    distinct_orders = c.fetchone()[0]
    avg_check = (revenue / distinct_orders) if distinct_orders else None

    revenue_series = []
    if not orders_df.empty:
        s = orders_df.groupby(orders_df["order_date"].dt.to_period("M"))["revenue"].sum().reset_index()
        for _, r in s.iterrows():
            revenue_series.append({"period": str(r["order_date"]), "revenue": float(r["revenue"])})

    expenses_series = []
    if not exp_df.empty:
        s = exp_df.groupby(exp_df["expense_date"].dt.to_period("M"))["amount"].sum().reset_index()
        for _, r in s.iterrows():
            expenses_series.append({"period": str(r["expense_date"]), "expenses": float(r["amount"])})

    # top customers
    c.execute("""
        SELECT o.customer_name, SUM(o.revenue) as rev, COUNT(*) as cnt, MAX(o.order_date)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        GROUP BY o.customer_name
        ORDER BY rev DESC
        LIMIT 20
    """, (ws_id,))
    tops = []
    for row in c.fetchall():
        tops.append({
            "customer_name": row[0],
            "revenue": float(row[1]),
            "orders": int(row[2]),
            "ltv": float(row[1]),
            "last_order_date": row[3]
        })

    top_share = 0.0
    if revenue > 0 and tops:
        top_5 = sum(t["revenue"] for t in tops[:5])
        top_share = top_5 / revenue

    conn.close()

    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "margin": margin,
        "avg_check": avg_check,
        "revenue_series": revenue_series,
        "expenses_series": expenses_series,
        "top_customers": tops,
        "top_customers_share": top_share,
        "period": {"from": None, "to": None},
    }

def growth(series_list, key):
    if not series_list or len(series_list) < 2:
        return 0.0
    first = series_list[-2].get(key, 0)
    last = series_list[-1].get(key, 0)
    if first == 0:
        return 0.0
    return (last - first) / first

def generate_insights(m):
    ins = []
    rev_g = growth(m["revenue_series"], "revenue")
    exp_g = growth(m["expenses_series"], "expenses")
    if rev_g < 0 and exp_g > 0:
        ins.append("Прибыль снизилась из-за роста расходов при падении выручки.")
    if m["top_customers_share"] > 0.6:
        ins.append("70% выручки дает узкая группа клиентов — держите их в фокусе.")
    if exp_g > 0.3:
        ins.append("Расходы растут быстрее выручки. Проверьте маркетинг и операционные издержки.")
    if not ins:
        ins.append("Метрики стабильны, значимых аномалий нет.")
    return ins[:3]

# ----------------- SIDEBAR: workspaces only -----------------
st.sidebar.title("Рабочие области")
workspaces = list_workspaces()
ws_names = {ws_id: name for ws_id, name in workspaces}

if "current_ws" not in st.session_state:
    st.session_state.current_ws = workspaces[0][0] if workspaces else None

selected_ws_name = st.sidebar.selectbox(
    "Выбери область",
    options=[ws_names[w[0]] for w in workspaces] if workspaces else [],
    index=0 if workspaces else None,
)
for ws_id, name in ws_names.items():
    if name == selected_ws_name:
        st.session_state.current_ws = ws_id
        break

with st.sidebar.expander("➕ Новая область"):
    new_ws = st.text_input("Название области")
    if st.button("Создать область"):
        if new_ws.strip():
            ws_id = create_workspace(new_ws.strip())
            st.session_state.current_ws = ws_id
            st.rerun()

current_ws = st.session_state.current_ws

# ----------------- session for uploads (split) -----------------
# отдельные состояния для оплат и для расходов
for key in [
    "orders_df", "orders_ds_id", "orders_schema",
    "expenses_df", "expenses_ds_id", "expenses_schema",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ----------------- MAIN TABS -----------------
st.title(f"ClarityOS — {ws_names.get(current_ws, '')}")
tab_dashboard, tab_upload, tab_mapping = st.tabs(["Дашборд", "Загрузка данных", "Маппинг"])

# ======== DASHBOARD ========
with tab_dashboard:
    metrics = calc_metrics(current_ws)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">Выручка<br><h3>{:,.0f} ₽</h3></div>'.format(metrics["revenue"]).replace(",", " "), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">Расходы<br><h3>{:,.0f} ₽</h3></div>'.format(metrics["expenses"]).replace(",", " "), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">Прибыль<br><h3>{:,.0f} ₽</h3></div>'.format(metrics["profit"]).replace(",", " "), unsafe_allow_html=True)
    with col4:
        margin_txt = f"{metrics['margin']*100:.1f}%" if metrics["margin"] is not None else "—"
        st.markdown(f'<div class="metric-card">Маржа<br><h3>{margin_txt}</h3></div>', unsafe_allow_html=True)

    st.subheader("Выручка и расходы по месяцам")
    periods = sorted({s["period"] for s in metrics["revenue_series"]} | {s["period"] for s in metrics["expenses_series"]})
    chart_data = []
    for p in periods:
        rev = next((x["revenue"] for x in metrics["revenue_series"] if x["period"] == p), 0)
        exp = next((x["expenses"] for x in metrics["expenses_series"] if x["period"] == p), 0)
        chart_data.append({"period": p, "Revenue": rev, "Expenses": exp})
    if chart_data:
        st.line_chart(pd.DataFrame(chart_data).set_index("period"))
    else:
        st.info("Нет данных, перейдите на вкладку «Загрузка данных».")

    st.subheader("Топ клиентов")
    if metrics["top_customers"]:
        st.dataframe(pd.DataFrame(metrics["top_customers"]))
    else:
        st.info("Появятся после загрузки и маппинга оплат.")

    st.subheader("AI-инсайты")
    for ins in generate_insights(metrics):
        st.markdown(f'<div class="insight">{ins}</div>', unsafe_allow_html=True)

# ======== UPLOAD ========
with tab_upload:
    st.subheader("Загрузка данных")
    left, right = st.columns(2)

    # ---- LEFT: orders ----
    with left:
        st.markdown("### Оплаты (orders)")
        orders_file = st.file_uploader("CSV с оплатами", type=["csv"], key="orders_upload")
        orders_gsheet = st.text_input("Google Sheets с оплатами (публичный)", key="orders_gsheet")

        if orders_file is not None:
            content = orders_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), sep=";")
                if df.shape[1] == 1:
                    df = pd.read_csv(io.BytesIO(content), sep=",")
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=",")
            detected = list(df.columns)

            conn = get_conn()
            c = conn.cursor()
            ds_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            c.execute("""INSERT INTO data_source
                (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (ds_id, current_ws, "csv", orders_file.name, None, "uploaded", "orders", now, now))
            up_id = str(uuid.uuid4())
            c.execute("""INSERT INTO data_upload
                (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (up_id, ds_id, orders_file.name, "", ",".join(detected), len(df), now))
            conn.commit()
            conn.close()

            st.session_state.orders_df = df
            st.session_state.orders_ds_id = ds_id
            st.session_state.orders_schema = detected

            st.success("Оплаты загружены. Перейдите на вкладку «Маппинг» → левый блок.")
            st.dataframe(df.head())

        elif orders_gsheet:
            csv_url = parse_google_sheet_to_csv_url(orders_gsheet)
            if not csv_url:
                st.error("Неверная ссылка на Google Sheets")
            else:
                df = pd.read_csv(csv_url)
                detected = list(df.columns)

                conn = get_conn()
                c = conn.cursor()
                ds_id = str(uuid.uuid4())
                now = datetime.utcnow().isoformat()
                c.execute("""INSERT INTO data_source
                    (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (ds_id, current_ws, "google_sheets", "Google Sheet (orders)", orders_gsheet, "uploaded", "orders", now, now))
                up_id = str(uuid.uuid4())
                c.execute("""INSERT INTO data_upload
                    (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                    VALUES (?,?,?,?,?,?,?)""",
                    (up_id, ds_id, "sheet_orders", "", ",".join(detected), len(df), now))
                conn.commit()
                conn.close()

                st.session_state.orders_df = df
                st.session_state.orders_ds_id = ds_id
                st.session_state.orders_schema = detected

                st.success("Оплаты из Google Sheets загружены.")
                st.dataframe(df.head())

        # список всех sources (orders)
        st.markdown("#### Источники оплат")
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT id, title, type, status, created_at
            FROM data_source
            WHERE workspace_id = ? AND category = 'orders'
            ORDER BY created_at DESC
        """, (current_ws,))
        rows = c.fetchall()
        conn.close()
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["id","title","type","status","created_at"]))
        else:
            st.caption("Пока нет источников оплат.")

    # ---- RIGHT: expenses ----
    with right:
        st.markdown("### Расходы (expenses)")
        exp_file = st.file_uploader("CSV с расходами", type=["csv"], key="exp_upload")
        exp_gsheet = st.text_input("Google Sheets с расходами (публичный)", key="exp_gsheet")

        if exp_file is not None:
            content = exp_file.read()
            try:
                df = pd.read_csv(io.BytesIO(content), sep=";")
                if df.shape[1] == 1:
                    df = pd.read_csv(io.BytesIO(content), sep=",")
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=",")
            detected = list(df.columns)

            conn = get_conn()
            c = conn.cursor()
            ds_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            c.execute("""INSERT INTO data_source
                (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (ds_id, current_ws, "csv", exp_file.name, None, "uploaded", "expenses", now, now))
            up_id = str(uuid.uuid4())
            c.execute("""INSERT INTO data_upload
                (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (up_id, ds_id, exp_file.name, "", ",".join(detected), len(df), now))
            conn.commit()
            conn.close()

            st.session_state.expenses_df = df
            st.session_state.expenses_ds_id = ds_id
            st.session_state.expenses_schema = detected

            st.success("Расходы загружены. Перейдите на вкладку «Маппинг» → правый блок.")
            st.dataframe(df.head())

        elif exp_gsheet:
            csv_url = parse_google_sheet_to_csv_url(exp_gsheet)
            if not csv_url:
                st.error("Неверная ссылка на Google Sheets")
            else:
                df = pd.read_csv(csv_url)
                detected = list(df.columns)

                conn = get_conn()
                c = conn.cursor()
                ds_id = str(uuid.uuid4())
                now = datetime.utcnow().isoformat()
                c.execute("""INSERT INTO data_source
                    (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (ds_id, current_ws, "google_sheets", "Google Sheet (expenses)", exp_gsheet, "uploaded", "expenses", now, now))
                up_id = str(uuid.uuid4())
                c.execute("""INSERT INTO data_upload
                    (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                    VALUES (?,?,?,?,?,?,?)""",
                    (up_id, ds_id, "sheet_expenses", "", ",".join(detected), len(df), now))
                conn.commit()
                conn.close()

                st.session_state.expenses_df = df
                st.session_state.expenses_ds_id = ds_id
                st.session_state.expenses_schema = detected

                st.success("Расходы из Google Sheets загружены.")
                st.dataframe(df.head())

        # список всех sources (expenses)
        st.markdown("#### Источники расходов")
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT id, title, type, status, created_at
            FROM data_source
            WHERE workspace_id = ? AND category = 'expenses'
            ORDER BY created_at DESC
        """, (current_ws,))
        rows = c.fetchall()
        conn.close()
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["id","title","type","status","created_at"]))
        else:
            st.caption("Пока нет источников расходов.")

# ======== MAPPING ========
with tab_mapping:
    st.subheader("Маппинг")
    col_left, col_right = st.columns(2)

    # ================== ОПЛАТЫ ==================
    with col_left:
        st.markdown("### Маппинг оплат")

        # источники оплат
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT id, title, status, created_at
            FROM data_source
            WHERE workspace_id = ? AND category = 'orders'
            ORDER BY created_at DESC
        """, (current_ws,))
        order_sources = c.fetchall()
        conn.close()

        if not order_sources:
            st.info("Нет источников оплат.")
        else:
            labels = [f"{r[1]} [{r[0][:6]}]" for r in order_sources]
            label = st.selectbox("Источник оплат", labels, key="orders_src_sel")
            order_source_id = next(r[0] for i, r in enumerate(order_sources) if labels[i] == label)

            # схема
            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                SELECT detected_schema
                FROM data_upload
                WHERE data_source_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (order_source_id,))
            row = c.fetchone()
            conn.close()
            detected = row[0].split(",") if row and row[0] else []
            suggest = suggest_mapping(detected)

            # превью
            orders_df_raw = None
            if st.session_state.orders_ds_id == order_source_id and st.session_state.orders_df is not None:
                orders_df_raw = st.session_state.orders_df.copy()
                st.caption("Первые строки файла:")
                st.dataframe(orders_df_raw.head())
            else:
                st.caption("Файл загружен ранее — показываем только схему.")

            # ---- ФОРМА ----
            with st.form(f"orders_form_{order_source_id}"):
                order_mapping = {}
                for f in suggest["orders"]:
                    tgt = f["target"]
                    options = ["— не выбрано —"] + detected
                    # первый показ — даём подсказку
                    suggested_col = f.get("suggested_column")
                    idx = options.index(suggested_col) if suggested_col in detected else 0
                    chosen = st.selectbox(
                        f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                        options=options,
                        index=idx,
                        key=f"ord_{order_source_id}_{tgt}",
                    )
                    order_mapping[tgt] = None if chosen == "— не выбрано —" else chosen

                submitted = st.form_submit_button("Сохранить маппинг и запустить ETL (оплаты)")

            if submitted:
                # почистили старые строки этого источника
                conn = get_conn()
                c = conn.cursor()
                c.execute('DELETE FROM "order" WHERE data_source_id = ?', (order_source_id,))
                conn.commit()
                conn.close()

                if orders_df_raw is None:
                    st.error("Перезагрузите файл на вкладке «Загрузка данных» и повторите.")
                else:
                    req = ["order_id", "order_date", "customer_name", "product", "revenue"]
                    miss = [x for x in req if not order_mapping.get(x)]
                    if miss:
                        st.error("Не заполнены обязательные поля: " + ", ".join(miss))
                    else:
                        df_orders = apply_mapping_to_df(orders_df_raw, order_mapping, "orders")
                        insert_orders(df_orders, order_source_id)
                        rebuild_customers(current_ws)
                        m = calc_metrics(current_ws)
                        ins = generate_insights(m)

                        conn = get_conn()
                        c = conn.cursor()
                        snap_id = str(uuid.uuid4())
                        now = datetime.utcnow().isoformat()
                        c.execute("""
                            INSERT INTO metrics_snapshot
                            (id, workspace_id, period_from, period_to, payload_json, created_at)
                            VALUES (?,?,?,?,?,?)
                        """, (snap_id, current_ws, None, None, json.dumps({"metrics": m, "insights": ins}), now))
                        for txt in ins:
                            c.execute("""
                                INSERT INTO insight
                                (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                                VALUES (?,?,?,?,?,?)
                            """, (str(uuid.uuid4()), current_ws, snap_id, txt, "rule", now))
                        c.execute("UPDATE data_source SET status=?, updated_at=? WHERE id=?",
                                  ("processed", now, order_source_id))
                        conn.commit()
                        conn.close()
                        st.success("Оплаты промапплены и загружены ✅")

    # ================== РАСХОДЫ ==================
    with col_right:
        st.markdown("### Маппинг расходов")

        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            SELECT id, title, status, created_at
            FROM data_source
            WHERE workspace_id = ? AND category = 'expenses'
            ORDER BY created_at DESC
        """, (current_ws,))
        exp_sources = c.fetchall()
        conn.close()

        if not exp_sources:
            st.info("Нет источников расходов.")
        else:
            labels = [f"{r[1]} [{r[0][:6]}]" for r in exp_sources]
            label = st.selectbox("Источник расходов", labels, key="exp_src_sel")
            exp_source_id = next(r[0] for i, r in enumerate(exp_sources) if labels[i] == label)

            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                SELECT detected_schema
                FROM data_upload
                WHERE data_source_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (exp_source_id,))
            row = c.fetchone()
            conn.close()
            detected = row[0].split(",") if row and row[0] else []
            suggest = suggest_mapping(detected)

            exp_df_raw = None
            if st.session_state.expenses_ds_id == exp_source_id and st.session_state.expenses_df is not None:
                exp_df_raw = st.session_state.expenses_df.copy()
                st.caption("Первые строки файла:")
                st.dataframe(exp_df_raw.head())
            else:
                st.caption("Файл загружен ранее — показываем только схему.")

            # ---- ФОРМА ----
            with st.form(f"expenses_form_{exp_source_id}"):
                expense_mapping = {}
                for f in suggest["expenses"]:
                    tgt = f["target"]
                    options = ["— не выбрано —"] + detected
                    suggested_col = f.get("suggested_column")
                    idx = options.index(suggested_col) if suggested_col in detected else 0
                    chosen = st.selectbox(
                        f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                        options=options,
                        index=idx,
                        key=f"exp_{exp_source_id}_{tgt}",
                    )
                    expense_mapping[tgt] = None if chosen == "— не выбрано —" else chosen

                submitted_exp = st.form_submit_button("Сохранить маппинг и запустить ETL (расходы)")

            if submitted_exp:
                conn = get_conn()
                c = conn.cursor()
                c.execute("DELETE FROM expense WHERE data_source_id = ?", (exp_source_id,))
                conn.commit()
                conn.close()

                if exp_df_raw is None:
                    st.error("Перезагрузите файл на вкладке «Загрузка данных» и повторите.")
                else:
                    req = ["expense_date", "category", "amount"]
                    miss = [x for x in req if not expense_mapping.get(x)]
                    if miss:
                        st.error("Не заполнены обязательные поля: " + ", ".join(miss))
                    else:
                        df_exp = apply_mapping_to_df(exp_df_raw, expense_mapping, "expenses")
                        insert_expenses(df_exp, exp_source_id)
                        rebuild_customers(current_ws)
                        m = calc_metrics(current_ws)
                        ins = generate_insights(m)

                        conn = get_conn()
                        c = conn.cursor()
                        snap_id = str(uuid.uuid4())
                        now = datetime.utcnow().isoformat()
                        c.execute("""
                            INSERT INTO metrics_snapshot
                            (id, workspace_id, period_from, period_to, payload_json, created_at)
                            VALUES (?,?,?,?,?,?)
                        """, (snap_id, current_ws, None, None, json.dumps({"metrics": m, "insights": ins}), now))
                        for txt in ins:
                            c.execute("""
                                INSERT INTO insight
                                (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                                VALUES (?,?,?,?,?,?)
                            """, (str(uuid.uuid4()), current_ws, snap_id, txt, "rule", now))
                        c.execute("UPDATE data_source SET status=?, updated_at=? WHERE id=?",
                                  ("processed", now, exp_source_id))
                        conn.commit()
                        conn.close()
                        st.success("Расходы промапплены и загружены ✅")
