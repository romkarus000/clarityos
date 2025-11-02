# app.py
import streamlit as st
import pandas as pd
import sqlite3
import io
import uuid
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import numpy as np
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

# --------------- DB ---------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # базовые таблицы
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
    )
    """)

    # ДОП: добавить колонку category, если её нет
    c.execute("PRAGMA table_info(data_source)")
    cols = [r[1] for r in c.fetchall()]
    if "category" not in cols:
        c.execute("ALTER TABLE data_source ADD COLUMN category TEXT DEFAULT NULL")

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

    # ensure one demo user/workspace
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

# --------- small helpers ----------
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
    # возьмём первого пользователя как владельца
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
        {"target":"amount","label":"Сумма","required":True,"synonyms":["amount","sum","сумма","cost","расход"]}
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
        required = ["order_id","order_date","customer_name","product","revenue"]
        out_df = out_df[[c for c in out_df.columns if c is not None]]
        for col in required:
            if col not in out_df.columns:
                raise ValueError(f"{col} is required")
        out_df = out_df[out_df["order_date"].notna()]
        out_df["order_date"] = pd.to_datetime(out_df["order_date"], errors="coerce")
        out_df = out_df[out_df["order_date"].notna()]
        out_df["revenue"] = (
            out_df["revenue"].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        )
        out_df["revenue"] = pd.to_numeric(out_df["revenue"], errors="coerce").fillna(0.0)
    else:
        required = ["expense_date","category","amount"]
        for col in required:
            if col not in out_df.columns:
                raise ValueError(f"{col} is required")
        out_df["expense_date"] = pd.to_datetime(out_df["expense_date"], errors="coerce")
        out_df = out_df[out_df["expense_date"].notna()]
        out_df["amount"] = (
            out_df["amount"].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        )
        out_df["amount"] = pd.to_numeric(out_df["amount"], errors="coerce").fillna(0.0)
    return out_df

def insert_orders(df: pd.DataFrame, data_source_id: str):
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
                data_source_id,
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

def insert_expenses(df: pd.DataFrame, data_source_id: str):
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
                data_source_id,
                row["expense_date"].date().isoformat(),
                str(row["category"]),
                float(row["amount"]),
                now,
            ),
        )
    conn.commit()
    conn.close()

def rebuild_customers(workspace_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT o.customer_name
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (workspace_id,))
    names = [r[0] for r in c.fetchall() if r[0]]
    c.execute("SELECT name FROM customer WHERE workspace_id = ?", (workspace_id,))
    existing = {r[0] for r in c.fetchall()}
    now = datetime.utcnow().isoformat()
    for name in names:
        if name not in existing:
            c.execute("INSERT INTO customer (id, workspace_id, name, created_at) VALUES (?,?,?,?)",
                      (str(uuid.uuid4()), workspace_id, name, now))
    conn.commit()
    conn.close()

def calc_metrics(workspace_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT o.order_date, o.revenue
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (workspace_id,))
    orders_df = pd.DataFrame(c.fetchall(), columns=["order_date","revenue"])
    if not orders_df.empty:
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])

    c.execute("""
        SELECT e.expense_date, e.amount
        FROM expense e
        JOIN data_source ds ON ds.id = e.data_source_id
        WHERE ds.workspace_id = ?
    """, (workspace_id,))
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
    """, (workspace_id,))
    distinct_orders = c.fetchone()[0]
    avg_check = (revenue / distinct_orders) if distinct_orders else None

    # series
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
    """, (workspace_id,))
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
        ins.append("70%+ выручки дают несколько клиентов — усиливайте удержание и апсейл.")
    if exp_g > 0.3:
        ins.append("Расходы растут быстрее выручки. Проверьте маркетинг/операционные траты.")
    if not ins:
        ins.append("Метрики стабильны, отклонений не найдено.")
    return ins[:3]

# ---------- SIDEBAR: workspaces ----------
st.sidebar.title("ClarityOS")
workspaces = list_workspaces()
ws_names = {ws_id: name for ws_id, name in workspaces}

if "current_ws" not in st.session_state:
    st.session_state.current_ws = workspaces[0][0] if workspaces else None

st.sidebar.subheader("Рабочие области")
selected_ws_name = st.sidebar.selectbox(
    "Выбери область",
    options=[ws_names[w[0]] for w in workspaces],
    index=0 if workspaces else None,
)
# получить id по имени
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

page = st.sidebar.radio("Навигация", ["1. Загрузка", "2. Маппинг", "3. Дашборд"], index=0)

# держим в сессии последний сырой df и id источника
if "latest_df" not in st.session_state:
    st.session_state.latest_df = None
if "latest_data_source_id" not in st.session_state:
    st.session_state.latest_data_source_id = None
if "latest_mapping_suggest" not in st.session_state:
    st.session_state.latest_mapping_suggest = None
if "latest_upload" not in st.session_state:
    st.session_state.latest_upload = None

current_ws = st.session_state.current_ws

# ----------- PAGE 1: upload -----------
if page == "1. Загрузка":
    st.title("Загрузка данных")
    st.write(f"Текущая рабочая область: **{ws_names[current_ws]}**")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded = st.file_uploader("CSV (оплаты / расходы)", type=["csv"])
    with col_up2:
        gsheet_url = st.text_input("Google Sheets (публичный)")

    category = st.selectbox("Что загружаем?", ["orders (оплаты)", "expenses (расходы)"])

    if uploaded is not None:
        content = uploaded.read()
        # попытка ; потом ,
        try:
            df = pd.read_csv(io.BytesIO(content), sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(io.BytesIO(content), sep=",")
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=",")
        detected = list(df.columns)
        st.success(f"Файл загружен, колонок: {len(detected)}")
        st.dataframe(df.head())

        conn = get_conn()
        c = conn.cursor()
        ds_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        c.execute("""INSERT INTO data_source
            (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (ds_id, current_ws, "csv", uploaded.name, None, "uploaded", "orders" if "orders" in category else "expenses", now, now))
        up_id = str(uuid.uuid4())
        c.execute("""INSERT INTO data_upload
            (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (up_id, ds_id, uploaded.name, "", ",".join(detected), len(df), now))
        conn.commit()
        conn.close()

        st.session_state.latest_df = df
        st.session_state.latest_data_source_id = ds_id
        st.session_state.latest_mapping_suggest = suggest_mapping(detected)
        st.session_state.latest_upload = {
            "data_source_id": ds_id,
            "upload_id": up_id,
            "detected_schema": detected,
        }
        st.info("Теперь открой «2. Маппинг» и сопоставь поля.")
    elif gsheet_url:
        try:
            csv_url = parse_google_sheet_to_csv_url(gsheet_url)
            if not csv_url:
                st.error("Неверная ссылка на Google Sheets")
            else:
                df = pd.read_csv(csv_url)
                detected = list(df.columns)
                st.success(f"Таблица загружена, колонок: {len(detected)}")
                st.dataframe(df.head())

                conn = get_conn()
                c = conn.cursor()
                ds_id = str(uuid.uuid4())
                now = datetime.utcnow().isoformat()
                c.execute("""INSERT INTO data_source
                    (id, workspace_id, type, title, source_url, status, category, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (ds_id, current_ws, "google_sheets", "Google Sheet", gsheet_url, "uploaded", "orders" if "orders" in category else "expenses", now, now))
                up_id = str(uuid.uuid4())
                c.execute("""INSERT INTO data_upload
                    (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                    VALUES (?,?,?,?,?,?,?)""",
                    (up_id, ds_id, "sheet", "", ",".join(detected), len(df), now))
                conn.commit()
                conn.close()

                st.session_state.latest_df = df
                st.session_state.latest_data_source_id = ds_id
                st.session_state.latest_mapping_suggest = suggest_mapping(detected)
                st.session_state.latest_upload = {
                    "data_source_id": ds_id,
                    "upload_id": up_id,
                    "detected_schema": detected,
                }
                st.info("Теперь открой «2. Маппинг» и сопоставь поля.")
        except Exception as e:
            st.error(f"Не удалось загрузить Google Sheets: {e}")

    # список источников в этом workspace
    st.subheader("Источники в этой области")
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, title, type, status, category, created_at
        FROM data_source
        WHERE workspace_id = ?
        ORDER BY created_at DESC
    """, (current_ws,))
    rows = c.fetchall()
    conn.close()
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["id","title","type","status","category","created_at"]))
    else:
        st.info("Пока нет загруженных документов в этой области.")

# ------------- PAGE 2: mapping -------------
elif page == "2. Маппинг":
    st.title("Маппинг полей")
    st.write(f"Рабочая область: **{ws_names[current_ws]}**")

    # выберем data_source, который хотим замаппить
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, title, category, status, created_at
        FROM data_source
        WHERE workspace_id = ?
        ORDER BY created_at DESC
    """, (current_ws,))
    sources = c.fetchall()
    conn.close()

    if not sources:
        st.warning("Сначала загрузите документы на шаге 1.")
        st.stop()

    source_labels = [f"{s[1]} ({s[2]}) [{s[0][:6]}]" for s in sources]
    selected_label = st.selectbox("Выберите источник для маппинга", source_labels)
    # находим id
    selected_source_id = None
    for i, s in enumerate(sources):
        if source_labels[i] == selected_label:
            selected_source_id = s[0]
            selected_source_category = s[2]
            break

    # достанем последнюю загрузку этого источника
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT detected_schema
        FROM data_upload
        WHERE data_source_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (selected_source_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        st.warning("Для этого источника нет загруженного файла.")
        st.stop()

    detected = row[0].split(",") if row[0] else []
    mapping_suggest = suggest_mapping(detected)

    # покажем сырой df, если он именно сейчас в сессии и это тот же источник
    if (
        st.session_state.latest_df is not None
        and st.session_state.latest_data_source_id == selected_source_id
    ):
        st.caption("Первые строки:")
        st.dataframe(st.session_state.latest_df.head())
        df_raw = st.session_state.latest_df.copy()
    else:
        # читаем заново из файла мы не можем (мы его не сохранили), поэтому просто маппим структуру
        df_raw = None
        st.info("Этот источник был загружен ранее — есть только схема. Для ре-ETL перезагрузите файл на шаге 1.")

    # строим UI маппинга
    if selected_source_category == "orders":
        st.subheader("Orders")
        order_mapping = {}
        for f in mapping_suggest["orders"]:
            col = st.selectbox(
                f'{f["label"]} ({f["target"]}) {"*" if f["required"] else ""}',
                options=["— не выбрано —"] + detected,
                index=(detected.index(f["suggested_column"]) + 1) if f.get("suggested_column") in detected else 0,
                key=f'ord_{selected_source_id}_{f["target"]}',
            )
            order_mapping[f["target"]] = None if col == "— не выбрано —" else col
        # expenses можно не показывать
        expense_mapping = {}
    else:
        st.subheader("Expenses")
        expense_mapping = {}
        for f in mapping_suggest["expenses"]:
            col = st.selectbox(
                f'{f["label"]} ({f["target"]}) {"*" if f["required"] else ""}',
                options=["— не выбрано —"] + detected,
                index=(detected.index(f["suggested_column"]) + 1) if f.get("suggested_column") in detected else 0,
                key=f'exp_{selected_source_id}_{f["target"]}',
            )
            expense_mapping[f["target"]] = None if col == "— не выбрано —" else col
        order_mapping = {}

    if st.button("Сохранить и запустить ETL"):
        conn = get_conn()
        c = conn.cursor()
        # очистить прошлую нормализацию именно этого источника
        c.execute('DELETE FROM "order" WHERE data_source_id = ?', (selected_source_id,))
        c.execute('DELETE FROM expense WHERE data_source_id = ?', (selected_source_id,))
        conn.commit()
        conn.close()

        try:
            if selected_source_category == "orders":
                if df_raw is None:
                    st.error("Нужно заново загрузить файл для этого источника на шаге 1, чтобы выполнить ETL.")
                    st.stop()
                req = ["order_id","order_date","customer_name","product","revenue"]
                miss = [x for x in req if not order_mapping.get(x)]
                if miss:
                    st.error(f"Не заполнены обязательные поля: {', '.join(miss)}")
                    st.stop()
                df_orders = apply_mapping_to_df(df_raw, order_mapping, "orders")
                insert_orders(df_orders, selected_source_id)
            else:
                if df_raw is None:
                    st.error("Нужно заново загрузить файл для этого источника на шаге 1, чтобы выполнить ETL.")
                    st.stop()
                req = ["expense_date","category","amount"]
                miss = [x for x in req if not expense_mapping.get(x)]
                if miss:
                    st.error(f"Не заполнены обязательные поля: {', '.join(miss)}")
                    st.stop()
                df_exp = apply_mapping_to_df(df_raw, expense_mapping, "expenses")
                insert_expenses(df_exp, selected_source_id)

            rebuild_customers(current_ws)
            metrics = calc_metrics(current_ws)
            insights = generate_insights(metrics)

            conn = get_conn()
            c = conn.cursor()
            snap_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            payload = {"metrics": metrics, "insights": insights}
            c.execute("""INSERT INTO metrics_snapshot
                (id, workspace_id, period_from, period_to, payload_json, created_at)
                VALUES (?,?,?,?,?,?)""",
                (snap_id, current_ws, metrics["period"]["from"], metrics["period"]["to"], json.dumps(payload), now))
            for ins in insights:
                c.execute("""INSERT INTO insight
                    (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                    VALUES (?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), current_ws, snap_id, ins, "rule", now))
            c.execute("UPDATE data_source SET status = ?, updated_at = ? WHERE id = ?",
                      ("processed", now, selected_source_id))
            conn.commit()
            conn.close()

            st.success("ETL выполнен, данные обновлены ✅")
        except Exception as e:
            st.error(f"Ошибка ETL: {e}")

# ------------- PAGE 3: dashboard -------------
elif page == "3. Дашборд":
    st.title("Дашборд")
    st.write(f"Рабочая область: **{ws_names[current_ws]}**")

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
    data = []
    for p in periods:
        rev = next((x["revenue"] for x in metrics["revenue_series"] if x["period"] == p), 0)
        exp = next((x["expenses"] for x in metrics["expenses_series"] if x["period"] == p), 0)
        data.append({"period": p, "Revenue": rev, "Expenses": exp})
    if data:
        df_chart = pd.DataFrame(data).set_index("period")
        st.line_chart(df_chart)
    else:
        st.info("Пока нет данных для графика.")

    st.subheader("Топ клиентов")
    if metrics["top_customers"]:
        st.dataframe(pd.DataFrame(metrics["top_customers"]))
    else:
        st.info("Клиенты появятся после загрузки оплат.")

    st.subheader("AI-инсайты")
    for ins in generate_insights(metrics):
        st.markdown(f'<div class="insight">{ins}</div>', unsafe_allow_html=True)
