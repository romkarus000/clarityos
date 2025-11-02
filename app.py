# app.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
from urllib.parse import urlparse, parse_qs
import io
import uuid
import re
import numpy as np

DB_PATH = "clarityos.db"

# ============ UI THEME ============

PRIMARY_COLOR = "#007AFF"
st.set_page_config(page_title="ClarityOS MVP", layout="wide")

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

# ============ DB ============

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # user / workspace (упрощаем: один дефолтный пользователь и workspace)
    c.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id TEXT PRIMARY KEY,
        email TEXT,
        name TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS workspace (
        id TEXT PRIMARY KEY,
        owner_id TEXT,
        name TEXT,
        created_at TEXT
    )
    """)
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
    c.execute("""
    CREATE TABLE IF NOT EXISTS data_upload (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        original_filename TEXT,
        storage_path TEXT,
        detected_schema TEXT,
        rows_count INTEGER,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS mapping_profile (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        target_table TEXT,
        mapping_json TEXT,
        status TEXT,
        created_at TEXT
    )
    """)
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
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS expense (
        id TEXT PRIMARY KEY,
        data_source_id TEXT,
        expense_date TEXT,
        category TEXT,
        amount REAL,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS customer (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        name TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS metrics_snapshot (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        period_from TEXT,
        period_to TEXT,
        payload_json TEXT,
        created_at TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS insight (
        id TEXT PRIMARY KEY,
        workspace_id TEXT,
        metrics_snapshot_id TEXT,
        text TEXT,
        rule_code TEXT,
        created_at TEXT
    )
    """)
    conn.commit()

    # ensure default user/workspace
    c.execute("SELECT COUNT(*) FROM user")
    if c.fetchone()[0] == 0:
        user_id = str(uuid.uuid4())
        ws_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        c.execute("INSERT INTO user (id, email, name, created_at) VALUES (?,?,?,?)",
                  (user_id, "demo@clarityos.app", "Demo User", now))
        c.execute("INSERT INTO workspace (id, owner_id, name, created_at) VALUES (?,?,?,?)",
                  (ws_id, user_id, "Demo Workspace", now))
        conn.commit()
    conn.close()

init_db()

def get_default_workspace_id():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM workspace LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row[0]

WORKSPACE_ID = get_default_workspace_id()

# ============ HELPERS ============

def parse_google_sheet_to_csv_url(url: str):
    # https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0 → export?format=csv&gid=0
    if "docs.google.com/spreadsheets" not in url:
        return None
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    try:
        sheet_id = parts[3]
    except IndexError:
        return None
    qs = parse_qs(parsed.fragment)
    gid = qs.get("gid", ["0"])[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return csv_url

def detect_schema_from_df(df: pd.DataFrame):
    return list(df.columns)

def load_raw_df(upload_bytes: bytes, sep=","):
    return pd.read_csv(io.BytesIO(upload_bytes), sep=sep)

def load_df_from_gsheet(url: str):
    csv_url = parse_google_sheet_to_csv_url(url)
    if not csv_url:
        raise ValueError("Неверная ссылка Google Sheets")
    df = pd.read_csv(csv_url)
    return df

def suggest_mapping(detected_cols):
    # target fields
    orders_targets = [
        {"target":"order_id","label":"ID заказа","required":True,"synonyms":["id","order","orderno","номер","id заказа"]},
        {"target":"order_date","label":"Дата заказа","required":True,"synonyms":["date","order_date","дата","date order"]},
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

    def find_suggest(synonyms):
        for col in detected_cols:
            col_low = col.lower().strip()
            for s in synonyms:
                if s in col_low:
                    return col
        return None

    for t in orders_targets:
        t["suggested_column"] = find_suggest(t["synonyms"])
    for t in expenses_targets:
        t["suggested_column"] = find_suggest(t["synonyms"])

    return {
        "orders": orders_targets,
        "expenses": expenses_targets
    }

def apply_mapping_to_df(df: pd.DataFrame, mapping: dict, target_table: str):
    out = {}
    for target, source in mapping.items():
        if source and source in df.columns:
            out[target] = df[source]
        else:
            out[target] = None
    out_df = pd.DataFrame(out)

    # validation
    if target_table == "orders":
        required = ["order_id","order_date","customer_name","product","revenue"]
    else:
        required = ["expense_date","category","amount"]

    for col in required:
        if col not in out_df.columns:
            raise ValueError(f"{col} is required in mapping")
        out_df = out_df[out_df[col].notna()]

    # normalize types
    if target_table == "orders":
        out_df["order_date"] = pd.to_datetime(out_df["order_date"], errors="coerce")
        out_df = out_df[out_df["order_date"].notna()]
        out_df["revenue"] = (
            out_df["revenue"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        out_df["revenue"] = pd.to_numeric(out_df["revenue"], errors="coerce").fillna(0.0)
    else:
        out_df["expense_date"] = pd.to_datetime(out_df["expense_date"], errors="coerce")
        out_df = out_df[out_df["expense_date"].notna()]
        out_df["amount"] = (
            out_df["amount"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        out_df["amount"] = pd.to_numeric(out_df["amount"], errors="coerce").fillna(0.0)

    return out_df

def insert_orders(df: pd.DataFrame, data_source_id: str):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    for _, row in df.iterrows():
        order_id = str(uuid.uuid4())
        c.execute("""
        INSERT INTO "order" (id, data_source_id, external_id, order_date, customer_name, product, revenue, channel, customer_id, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            order_id,
            data_source_id,
            str(row["order_id"]),
            row["order_date"].date().isoformat(),
            str(row["customer_name"]),
            str(row["product"]),
            float(row["revenue"]),
            str(row.get("channel") if pd.notna(row.get("channel")) else None),
            None,
            now
        ))
    conn.commit()
    conn.close()

def insert_expenses(df: pd.DataFrame, data_source_id: str):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    for _, row in df.iterrows():
        expense_id = str(uuid.uuid4())
        c.execute("""
        INSERT INTO expense (id, data_source_id, expense_date, category, amount, created_at)
        VALUES (?,?,?,?,?,?)
        """,
        (
            expense_id,
            data_source_id,
            row["expense_date"].date().isoformat(),
            str(row["category"]),
            float(row["amount"]),
            now
        ))
    conn.commit()
    conn.close()

def rebuild_customers(workspace_id: str):
    conn = get_conn()
    c = conn.cursor()
    # customers from orders.customer_name
    c.execute("""
    SELECT DISTINCT o.customer_name
    FROM "order" o
    JOIN data_source ds ON ds.id = o.data_source_id
    WHERE ds.workspace_id = ?
    """, (workspace_id,))
    names = [r[0] for r in c.fetchall() if r[0]]
    # existing
    c.execute("SELECT name FROM customer WHERE workspace_id = ?", (workspace_id,))
    existing = {r[0] for r in c.fetchall()}
    now = datetime.utcnow().isoformat()
    for name in names:
        if name not in existing:
            c.execute("INSERT INTO customer (id, workspace_id, name, created_at) VALUES (?,?,?,?)",
                      (str(uuid.uuid4()), workspace_id, name, now))
    conn.commit()
    conn.close()

def calc_metrics(workspace_id: str, period_from: str = None, period_to: str = None):
    conn = get_conn()
    c = conn.cursor()
    # orders
    q_orders = """
    SELECT o.order_date, o.revenue
    FROM "order" o
    JOIN data_source ds ON ds.id = o.data_source_id
    WHERE ds.workspace_id = ?
    """
    params = [workspace_id]
    if period_from:
        q_orders += " AND o.order_date >= ?"
        params.append(period_from)
    if period_to:
        q_orders += " AND o.order_date <= ?"
        params.append(period_to)
    c.execute(q_orders, params)
    rows = c.fetchall()
    orders_df = pd.DataFrame(rows, columns=["order_date","revenue"])
    if not orders_df.empty:
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])
    # expenses
    q_exp = """
    SELECT e.expense_date, e.amount
    FROM expense e
    JOIN data_source ds ON ds.id = e.data_source_id
    WHERE ds.workspace_id = ?
    """
    params = [workspace_id]
    if period_from:
        q_exp += " AND e.expense_date >= ?"
        params.append(period_from)
    if period_to:
        q_exp += " AND e.expense_date <= ?"
        params.append(period_to)
    c.execute(q_exp, params)
    rows_e = c.fetchall()
    exp_df = pd.DataFrame(rows_e, columns=["expense_date","amount"])
    if not exp_df.empty:
        exp_df["expense_date"] = pd.to_datetime(exp_df["expense_date"])

    revenue = float(orders_df["revenue"].sum()) if not orders_df.empty else 0.0
    expenses = float(exp_df["amount"].sum()) if not exp_df.empty else 0.0
    profit = revenue - expenses
    margin = (profit / revenue) if revenue > 0 else None

    avg_check = None
    if not orders_df.empty:
        # external_id count
        conn2 = get_conn()
        c2 = conn2.cursor()
        c2.execute("""
        SELECT COUNT(DISTINCT o.external_id)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        """, (workspace_id,))
        distinct_orders = c2.fetchone()[0]
        conn2.close()
        if distinct_orders > 0:
            avg_check = revenue / distinct_orders

    # series month
    revenue_series = []
    expense_series = []
    if not orders_df.empty:
        s = orders_df.groupby(orders_df["order_date"].dt.to_period("M"))["revenue"].sum().reset_index()
        for _, r in s.iterrows():
            revenue_series.append({"period": str(r["order_date"]), "revenue": float(r["revenue"])})
    if not exp_df.empty:
        s = exp_df.groupby(exp_df["expense_date"].dt.to_period("M"))["amount"].sum().reset_index()
        for _, r in s.iterrows():
            expense_series.append({"period": str(r["expense_date"]), "expenses": float(r["amount"])})

    # top customers
    c.execute("""
    SELECT o.customer_name, SUM(o.revenue) as rev, COUNT(*) as cnt, MAX(o.order_date) as last_order
    FROM "order" o
    JOIN data_source ds ON ds.id = o.data_source_id
    WHERE ds.workspace_id = ?
    GROUP BY o.customer_name
    ORDER BY rev DESC
    LIMIT 20
    """, (workspace_id,))
    top_customers = []
    for row in c.fetchall():
        top_customers.append({
            "customer_name": row[0],
            "revenue": float(row[1]),
            "orders": int(row[2]),
            "ltv": float(row[1]),
            "last_order_date": row[3]
        })
    conn.close()

    top_share = 0.0
    if revenue > 0 and top_customers:
        top_total = sum([c["revenue"] for c in top_customers[:5]])
        top_share = top_total / revenue

    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "margin": margin,
        "avg_check": avg_check,
        "revenue_series": revenue_series,
        "expenses_series": expense_series,
        "top_customers": top_customers,
        "top_customers_share": top_share,
        "period": {
            "from": period_from,
            "to": period_to
        }
    }

def growth(series_list, key="revenue"):
    # series_list: [{period, revenue}, ...] ordered by period asc
    if not series_list or len(series_list) < 2:
        return 0.0
    first = series_list[-2].get(key, 0)
    last = series_list[-1].get(key, 0)
    if first == 0:
        return 0.0
    return (last - first) / first

def generate_insights(m):
    insights = []
    rev_growth = growth(m["revenue_series"], "revenue")
    exp_growth = growth(m["expenses_series"], "expenses")
    profit_growth = 0  # simplified = revenue-expenses dynamics is seen

    if rev_growth < 0 and exp_growth > 0:
        insights.append("Прибыль снизилась из-за роста расходов при падении выручки.")
    if m["top_customers_share"] > 0.6:
        insights.append("Бизнес сильно зависит от небольшого числа клиентов: 5 клиентов дают >60% выручки.")
    if exp_growth > 0.3:
        insights.append("Расходы растут быстрее выручки. Проверьте маркетинг и операционные затраты.")
    if not insights:
        insights.append("Метрики стабильны. Продолжайте в том же духе.")
    return insights[:3]

# ============ STREAMLIT UI ============

st.sidebar.title("ClarityOS MVP")
page = st.sidebar.radio("Навигация", ["1. Загрузка", "2. Маппинг", "3. Дашборд"], index=0)
st.sidebar.markdown(f"<small>Workspace: {WORKSPACE_ID}</small>", unsafe_allow_html=True)

if "latest_upload" not in st.session_state:
    st.session_state.latest_upload = None
if "latest_df" not in st.session_state:
    st.session_state.latest_df = None
if "latest_mapping_suggest" not in st.session_state:
    st.session_state.latest_mapping_suggest = None
if "latest_data_source_id" not in st.session_state:
    st.session_state.latest_data_source_id = None

# 1. ЗАГРУЗКА
if page == "1. Загрузка":
    st.title("Импорт данных")
    st.write("Загрузи CSV **или** укажи ссылку на Google Sheets. Мы определим колонки и перейдём к маппингу 👇")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("CSV / UTF-8 / ; или ,", type=["csv"])
    with col2:
        gsheet_url = st.text_input("Google Sheets (публичная)")

    if uploaded is not None:
        content = uploaded.read()
        # пробуем ; и ,
        try:
            df = load_raw_df(content, sep=";")
            if df.shape[1] == 1:
                df = load_raw_df(content, sep=",")
        except Exception:
            df = load_raw_df(content, sep=",")
        detected = detect_schema_from_df(df)
        st.success(f"Определили {len(detected)} колонок")
        st.dataframe(df.head())
        # сохраняем как data_source / data_upload
        conn = get_conn()
        c = conn.cursor()
        ds_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        c.execute("""INSERT INTO data_source (id, workspace_id, type, title, source_url, status, created_at, updated_at)
                     VALUES (?,?,?,?,?,?,?,?)""",
                  (ds_id, WORKSPACE_ID, "csv", uploaded.name, None, "uploaded", now, now))
        up_id = str(uuid.uuid4())
        c.execute("""INSERT INTO data_upload (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                     VALUES (?,?,?,?,?,?,?)""",
                  (up_id, ds_id, uploaded.name, "", ",".join(detected), len(df), now))
        conn.commit()
        conn.close()

        st.session_state.latest_upload = {
            "data_source_id": ds_id,
            "upload_id": up_id,
            "detected_schema": detected
        }
        st.session_state.latest_df = df
        st.session_state.latest_data_source_id = ds_id
        st.session_state.latest_mapping_suggest = suggest_mapping(detected)
        st.info("Данные загружены. Перейдите в «2. Маппинг».")
    elif gsheet_url:
        try:
            df = load_df_from_gsheet(gsheet_url)
            detected = detect_schema_from_df(df)
            st.success(f"Google Sheets загружен, колонок: {len(detected)}")
            st.dataframe(df.head())

            conn = get_conn()
            c = conn.cursor()
            ds_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            c.execute("""INSERT INTO data_source (id, workspace_id, type, title, source_url, status, created_at, updated_at)
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (ds_id, WORKSPACE_ID, "google_sheets", "Google Sheet", gsheet_url, "uploaded", now, now))
            up_id = str(uuid.uuid4())
            c.execute("""INSERT INTO data_upload (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
                         VALUES (?,?,?,?,?,?,?)""",
                      (up_id, ds_id, "sheet", "", ",".join(detected), len(df), now))
            conn.commit()
            conn.close()

            st.session_state.latest_upload = {
                "data_source_id": ds_id,
                "upload_id": up_id,
                "detected_schema": detected
            }
            st.session_state.latest_df = df
            st.session_state.latest_data_source_id = ds_id
            st.session_state.latest_mapping_suggest = suggest_mapping(detected)
            st.info("Данные загружены. Перейдите в «2. Маппинг».")
        except Exception as e:
            st.error(f"Не удалось загрузить Google Sheets: {e}")

# 2. МАППИНГ
elif page == "2. Маппинг":
    st.title("Маппинг полей")

    has_mapping = (
        "latest_mapping_suggest" in st.session_state
        and st.session_state.latest_mapping_suggest is not None
    )
    has_df = (
        "latest_df" in st.session_state
        and st.session_state.latest_df is not None
        and isinstance(st.session_state.latest_df, pd.DataFrame)
        and not st.session_state.latest_df.empty
    )

    if not has_mapping or not has_df:
        st.warning("Сначала загрузите CSV/Sheets на шаге 1.")
        st.stop()

    detected = st.session_state.latest_upload["detected_schema"]
    mapping_suggest = st.session_state.latest_mapping_suggest

    st.caption("Первые строки загруженного файла:")
    st.dataframe(st.session_state.latest_df.head())

    st.subheader("Orders")
    order_mapping = {}
    for f in mapping_suggest["orders"]:
        col = st.selectbox(
            f'{f["label"]} ({f["target"]}) {"*" if f["required"] else ""}',
            options=["— не выбрано —"] + detected,
            index=(detected.index(f["suggested_column"]) + 1) if f.get("suggested_column") in detected else 0,
            key=f'ord_{f["target"]}',
        )
        order_mapping[f["target"]] = None if col == "— не выбрано —" else col

    st.subheader("Expenses (опционально)")
    expense_mapping = {}
    for f in mapping_suggest["expenses"]:
        col = st.selectbox(
            f'{f["label"]} ({f["target"]}) {"*" if f["required"] else ""}',
            options=["— не выбрано —"] + detected,
            index=(detected.index(f["suggested_column"]) + 1) if f.get("suggested_column") in detected else 0,
            key=f'exp_{f["target"]}',
        )
        expense_mapping[f["target"]] = None if col == "— не выбрано —" else col

    if st.button("Сохранить и запустить ETL"):
        # 1. валидация orders
        req_orders = ["order_id", "order_date", "customer_name", "product", "revenue"]
        miss = [x for x in req_orders if not order_mapping.get(x)]
        if miss:
            st.error(f"Не заполнены обязательные поля Orders: {', '.join(miss)}")
            st.stop()

        df_raw = st.session_state.latest_df.copy()
        ds_id = st.session_state.latest_data_source_id

        # 2. очистка старых данных ЭТОГО источника
        conn = get_conn()
        c = conn.cursor()
        c.execute('DELETE FROM "order" WHERE data_source_id = ?', (ds_id,))
        c.execute('DELETE FROM expense WHERE data_source_id = ?', (ds_id,))
        conn.commit()
        conn.close()

        # 3. загрузка orders
        try:
            df_orders = apply_mapping_to_df(df_raw, order_mapping, "orders")
            insert_orders(df_orders, ds_id)
        except Exception as e:
            st.error(f"Ошибка загрузки Orders: {e}")
            st.stop()

        # 4. загрузка expenses (если маппинг полноценный)
        if (
            expense_mapping.get("expense_date")
            and expense_mapping.get("category")
            and expense_mapping.get("amount")
        ):
            try:
                df_exp = apply_mapping_to_df(df_raw, expense_mapping, "expenses")
                insert_expenses(df_exp, ds_id)
            except Exception as e:
                st.warning(f"Расходы не загружены: {e}")

        # 5. пересчёт
        rebuild_customers(WORKSPACE_ID)
        metrics = calc_metrics(WORKSPACE_ID)
        insights = generate_insights(metrics)

        # (сохранение снапшота как у тебя было)
        import json
        conn = get_conn()
        c = conn.cursor()
        snap_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        payload = {"metrics": metrics, "insights": insights}
        c.execute(
            """INSERT INTO metrics_snapshot
               (id, workspace_id, period_from, period_to, payload_json, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                snap_id,
                WORKSPACE_ID,
                metrics["period"]["from"],
                metrics["period"]["to"],
                json.dumps(payload),
                now,
            ),
        )
        for ins in insights:
            c.execute(
                """INSERT INTO insight
                   (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (str(uuid.uuid4()), WORKSPACE_ID, snap_id, ins, "rule", now),
            )
        c.execute(
            "UPDATE data_source SET status = ?, updated_at = ? WHERE id = ?",
            ("processed", now, ds_id),
        )
        conn.commit()
        conn.close()

        st.success("Данные перезаписаны для этого файла и метрики обновлены ✅")


# 3. ДАШБОРД
elif page == "3. Дашборд":
    st.title("Обзор бизнеса")

    metrics = calc_metrics(WORKSPACE_ID)

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
    # строим серию
    series_df = pd.DataFrame()
    periods = sorted({s["period"] for s in metrics["revenue_series"]} | {s["period"] for s in metrics["expenses_series"]})
    data = []
    for p in periods:
        rev = next((x["revenue"] for x in metrics["revenue_series"] if x["period"] == p), 0)
        exp = next((x["expenses"] for x in metrics["expenses_series"] if x["period"] == p), 0)
        data.append({"period": p, "Revenue": rev, "Expenses": exp})
    if data:
        series_df = pd.DataFrame(data).set_index("period")
        st.line_chart(series_df)
    else:
        st.info("Нет данных для графика — сначала загрузите файл и выполните маппинг.")

    st.subheader("Топ клиентов")
    if metrics["top_customers"]:
        st.dataframe(pd.DataFrame(metrics["top_customers"]))
    else:
        st.info("Клиенты появятся после загрузки заказов.")

    st.subheader("Инсайты")
    insights = generate_insights(metrics)
    for ins in insights:
        st.markdown(f'<div class="insight">{ins}</div>', unsafe_allow_html=True)

    st.caption("MVP ClarityOS • данные из SQLite • для демонстрации.")

# ============ END APP ============
