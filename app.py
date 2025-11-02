import streamlit as st
import sqlite3
import os
import uuid
import json
from datetime import datetime
import pandas as pd

# =========================================================
# 0. НАСТРОЙКИ
# =========================================================

st.set_page_config(page_title="ClarityOS", layout="wide")
PRIMARY = "#007AFF"
DB_PATH = "clarityos.db"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"]  {{
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .stButton>button {{
        background:{PRIMARY};
        color:white;
        border-radius:8px;
        height:42px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1. БАЗА
# =========================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    c = conn.cursor()

    # рабочие области
    c.execute("""
    CREATE TABLE IF NOT EXISTS workspace (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    # источники
    c.execute("""
    CREATE TABLE IF NOT EXISTS data_source (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        type TEXT NOT NULL,          -- 'csv' | 'google_sheets'
        title TEXT,
        source_url TEXT,
        status TEXT,
        category TEXT,               -- 'orders' | 'expenses'
        created_at TEXT NOT NULL,
        updated_at TEXT,
        FOREIGN KEY (workspace_id) REFERENCES workspace (id) ON DELETE CASCADE
    );
    """)

    # факты загрузок
    c.execute("""
    CREATE TABLE IF NOT EXISTS data_upload (
        id TEXT PRIMARY KEY,
        data_source_id TEXT NOT NULL,
        original_filename TEXT,
        storage_path TEXT,
        detected_schema TEXT,
        rows_count INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY (data_source_id) REFERENCES data_source (id) ON DELETE CASCADE
    );
    """)

    # нормализованные заказы
    c.execute("""
    CREATE TABLE IF NOT EXISTS "order" (
        id TEXT PRIMARY KEY,
        data_source_id TEXT NOT NULL,
        external_id TEXT,
        order_date TEXT,
        customer_name TEXT,
        product TEXT,
        revenue REAL,
        channel TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (data_source_id) REFERENCES data_source (id) ON DELETE CASCADE
    );
    """)

    # нормализованные расходы
    c.execute("""
    CREATE TABLE IF NOT EXISTS expense (
        id TEXT PRIMARY KEY,
        data_source_id TEXT NOT NULL,
        expense_date TEXT,
        category TEXT,
        amount REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (data_source_id) REFERENCES data_source (id) ON DELETE CASCADE
    );
    """)

    # клиенты
    c.execute("""
    CREATE TABLE IF NOT EXISTS customer (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspace (id) ON DELETE CASCADE
    );
    """)

    # снапшоты
    c.execute("""
    CREATE TABLE IF NOT EXISTS metrics_snapshot (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        period_from TEXT,
        period_to TEXT,
        payload_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspace (id) ON DELETE CASCADE
    );
    """)

    # инсайты
    c.execute("""
    CREATE TABLE IF NOT EXISTS insight (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        metrics_snapshot_id TEXT,
        text TEXT,
        rule_code TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (workspace_id) REFERENCES workspace (id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# 2. ХЕЛПЕРЫ
# =========================================================

def parse_google_sheet_to_csv_url(url: str) -> str:
    if "docs.google.com/spreadsheets" not in url:
        return url
    if "export?format=csv" in url:
        return url
    # https://docs.google.com/spreadsheets/d/<id>/edit#gid=0
    base = url.split("/edit")[0]
    return base + "/export?format=csv"


def suggest_mapping(detected_cols):
    # упрощённая версия твоего ТЗ
    return {
        "orders": [
            {"target": "order_id", "label": "ID заказа (order_id)", "required": True,
             "suggested_column": "order_id" if "order_id" in detected_cols else (detected_cols[0] if detected_cols else None)},
            {"target": "order_date", "label": "Дата заказа (order_date)", "required": True,
             "suggested_column": "date" if "date" in [c.lower() for c in detected_cols] else None},
            {"target": "customer_name", "label": "Клиент (customer_name)", "required": True,
             "suggested_column": "customer" if "customer" in [c.lower() for c in detected_cols] else None},
            {"target": "product", "label": "Продукт (product)", "required": True,
             "suggested_column": None},
            {"target": "revenue", "label": "Выручка (revenue)", "required": True,
             "suggested_column": "amount" if "amount" in [c.lower() for c in detected_cols] else None},
            {"target": "channel", "label": "Канал (channel)", "required": False,
             "suggested_column": None},
        ],
        "expenses": [
            {"target": "expense_date", "label": "Дата расхода (expense_date)", "required": True,
             "suggested_column": "date" if "date" in [c.lower() for c in detected_cols] else None},
            {"target": "category", "label": "Категория (category)", "required": True,
             "suggested_column": None},
            {"target": "amount", "label": "Сумма (amount)", "required": True,
             "suggested_column": "amount" if "amount" in [c.lower() for c in detected_cols] else None},
        ]
    }


def apply_mapping_to_df(df: pd.DataFrame, mapping: dict, target: str) -> pd.DataFrame:
    out = {}
    for tgt, src in mapping.items():
        if src is None:
            out[tgt] = None
        else:
            out[tgt] = df[src] if src in df.columns else None
    out_df = pd.DataFrame(out)
    # нормализация
    if target == "orders":
        if "revenue" in out_df.columns:
            out_df["revenue"] = (
                out_df["revenue"]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            out_df["revenue"] = pd.to_numeric(out_df["revenue"], errors="coerce").fillna(0.0)
    if target == "expenses":
        if "amount" in out_df.columns:
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
        c.execute(
            """
            INSERT INTO "order"
            (id, data_source_id, external_id, order_date, customer_name, product, revenue, channel, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                data_source_id,
                str(row.get("order_id") or ""),
                str(row.get("order_date") or ""),
                str(row.get("customer_name") or ""),
                str(row.get("product") or ""),
                float(row.get("revenue") or 0.0),
                str(row.get("channel") or ""),
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
            """
            INSERT INTO expense
            (id, data_source_id, expense_date, category, amount, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                data_source_id,
                str(row.get("expense_date") or ""),
                str(row.get("category") or ""),
                float(row.get("amount") or 0.0),
                now,
            ),
        )
    conn.commit()
    conn.close()


def rebuild_customers(workspace_id: str):
    conn = get_conn()
    c = conn.cursor()
    # удалить и пересобрать
    c.execute("DELETE FROM customer WHERE workspace_id = ?", (workspace_id,))
    c.execute(
        """
        INSERT INTO customer (id, workspace_id, name, created_at)
        SELECT DISTINCT
            ? || '_' || COALESCE(o.customer_name,'') AS id,
            ? AS workspace_id,
            COALESCE(o.customer_name,'') AS name,
            ?
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
          AND COALESCE(o.customer_name,'') <> ''
        """,
        (workspace_id, workspace_id, datetime.utcnow().isoformat(), workspace_id),
    )
    conn.commit()
    conn.close()


def calc_metrics(workspace_id: str) -> dict:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT COALESCE(SUM(o.revenue),0)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        """,
        (workspace_id,),
    )
    revenue = c.fetchone()[0]

    c.execute(
        """
        SELECT COALESCE(SUM(e.amount),0)
        FROM expense e
        JOIN data_source ds ON ds.id = e.data_source_id
        WHERE ds.workspace_id = ?
        """,
        (workspace_id,),
    )
    expenses = c.fetchone()[0]

    c.execute(
        """
        SELECT COUNT(DISTINCT o.external_id)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        """,
        (workspace_id,),
    )
    orders_cnt = c.fetchone()[0]

    conn.close()

    profit = revenue - expenses
    margin = profit / revenue if revenue > 0 else None
    avg_check = revenue / orders_cnt if orders_cnt > 0 else None

    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "margin": margin,
        "avg_check": avg_check,
    }


def generate_insights(m: dict) -> list[str]:
    ins = []
    if m["revenue"] == 0 and m["expenses"] == 0:
        ins.append("Данных пока нет — загрузите CSV или Google Sheet в эту область.")
    else:
        if m["expenses"] > m["revenue"]:
            ins.append("Расходы превышают выручку — проверьте статьи затрат.")
        if m["margin"] is not None and m["margin"] < 0.3:
            ins.append("Маржа ниже 30%. Подумайте о повышении цен или оптимизации расходов.")
        if not ins:
            ins.append("Метрики стабильны. Можно загрузить дополнительные источники для более точной картины.")
    return ins


def save_uploaded_file(uploaded_file, workspace_id: str, category: str) -> tuple[str, pd.DataFrame]:
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(uploads_dir, f"{file_id}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    df = pd.read_csv(file_path)
    # создаём data_source + data_upload
    conn = get_conn()
    c = conn.cursor()
    ds_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    c.execute(
        """
        INSERT INTO data_source (id, workspace_id, type, title, source_url, status, category, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (ds_id, workspace_id, "csv", uploaded_file.name, None, "uploaded", category, now),
    )
    c.execute(
        """
        INSERT INTO data_upload (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            ds_id,
            uploaded_file.name,
            file_path,
            ",".join(df.columns.tolist()),
            int(df.shape[0]),
            now,
        ),
    )
    conn.commit()
    conn.close()
    return ds_id, df


def save_google_sheet(sheet_url: str, workspace_id: str, category: str) -> str:
    csv_url = parse_google_sheet_to_csv_url(sheet_url)
    df = pd.read_csv(csv_url)
    conn = get_conn()
    c = conn.cursor()
    ds_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    c.execute(
        """
        INSERT INTO data_source (id, workspace_id, type, title, source_url, status, category, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (ds_id, workspace_id, "google_sheets", sheet_url, csv_url, "uploaded", category, now),
    )
    c.execute(
        """
        INSERT INTO data_upload (id, data_source_id, original_filename, storage_path, detected_schema, rows_count, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            ds_id,
            "Google Sheet",
            "",
            ",".join(df.columns.tolist()),
            int(df.shape[0]),
            now,
        ),
    )
    conn.commit()
    conn.close()
    return ds_id


# =========================================================
# 3. СЕССИЯ
# =========================================================

if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = None
if "orders_ds_id" not in st.session_state:
    st.session_state.orders_ds_id = None
if "orders_df" not in st.session_state:
    st.session_state.orders_df = None
if "expenses_ds_id" not in st.session_state:
    st.session_state.expenses_ds_id = None
if "expenses_df" not in st.session_state:
    st.session_state.expenses_df = None


# =========================================================
# 4. UI
# =========================================================

st.sidebar.header("Рабочие области")

# список областей
conn = get_conn()
c = conn.cursor()
c.execute("SELECT id, name FROM workspace ORDER BY created_at DESC")
workspaces = c.fetchall()
conn.close()

ws_names = {row[1]: row[0] for row in workspaces}

selected_ws_name = st.sidebar.selectbox(
    "Выберите область",
    ["— не выбрано —"] + list(ws_names.keys()),
    index=0 if not st.session_state.current_workspace else 1 + list(ws_names.keys()).index(
        next(name for name, wid in ws_names.items() if wid == st.session_state.current_workspace)
    ) if st.session_state.current_workspace in ws_names.values() else 0,
)

if selected_ws_name == "— не выбрано —":
    current_ws = None
else:
    current_ws = ws_names[selected_ws_name]
    st.session_state.current_workspace = current_ws

st.sidebar.markdown("#### Создать новую")
new_ws = st.sidebar.text_input("Название новой области", key="new_ws_name")
if st.sidebar.button("Создать область"):
    if new_ws.strip():
        conn = get_conn()
        c = conn.cursor()
        ws_id = str(uuid.uuid4())
        c.execute(
            "INSERT INTO workspace (id, name, created_at) VALUES (?,?,?)",
            (ws_id, new_ws.strip(), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        st.session_state.current_workspace = ws_id
        st.experimental_rerun()

st.title(f"ClarityOS — {selected_ws_name if current_ws else 'нет области'}")

if not current_ws:
    st.info("Создайте или выберите рабочую область слева, чтобы продолжить.")
    st.stop()

tab_dashboard, tab_upload, tab_mapping = st.tabs(["Дашборд", "Загрузка данных", "Маппинг"])


# =========================================================
# 4.1 DASHBOARD
# =========================================================

with tab_dashboard:
    st.subheader("Дашборд")

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT COUNT(*), COALESCE(SUM(revenue),0)
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
    """, (current_ws,))
    orders_cnt, orders_sum = c.fetchone()

    c.execute("""
        SELECT COUNT(*), COALESCE(SUM(amount),0)
        FROM expense e
        JOIN data_source ds ON ds.id = e.data_source_id
        WHERE ds.workspace_id = ?
    """, (current_ws,))
    exp_cnt, exp_sum = c.fetchone()

    c.execute("""
        SELECT o.customer_name, COALESCE(SUM(o.revenue),0) AS rev, COUNT(*) AS cnt
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        GROUP BY o.customer_name
        ORDER BY rev DESC
        LIMIT 20
    """, (current_ws,))
    top_customers = c.fetchall()

    c.execute("""
        SELECT o.external_id, o.order_date, o.customer_name, o.product, o.revenue
        FROM "order" o
        JOIN data_source ds ON ds.id = o.data_source_id
        WHERE ds.workspace_id = ?
        ORDER BY o.created_at DESC
        LIMIT 10
    """, (current_ws,))
    last_orders = c.fetchall()

    c.execute("""
        SELECT e.expense_date, e.category, e.amount
        FROM expense e
        JOIN data_source ds ON ds.id = e.data_source_id
        WHERE ds.workspace_id = ?
        ORDER BY e.created_at DESC
        LIMIT 10
    """, (current_ws,))
    last_exp = c.fetchall()

    conn.close()

    st.caption(f"🔎 Диагностика: оплат {orders_cnt}, выручка {orders_sum}; расходов {exp_cnt}, сумма {exp_sum}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Выручка", f"{orders_sum:,.0f} ₽".replace(",", " "))
    col2.metric("Расходы", f"{exp_sum:,.0f} ₽".replace(",", " "))
    profit = orders_sum - exp_sum
    col3.metric("Прибыль", f"{profit:,.0f} ₽".replace(",", " "))
    margin = (profit / orders_sum) if orders_sum else None
    col4.metric("Маржа", f"{margin*100:,.1f} %" if margin is not None else "—")

    st.markdown("#### Топ клиентов")
    if top_customers:
        st.dataframe(
            [
                {"Клиент": r[0], "Выручка": r[1], "Кол-во заказов": r[2]}
                for r in top_customers
            ]
        )
    else:
        st.write("Нет данных по клиентам.")

    st.markdown("#### Последние 10 оплат")
    if last_orders:
        st.dataframe(
            [
                {
                    "ID": r[0],
                    "Дата": r[1],
                    "Клиент": r[2],
                    "Продукт": r[3],
                    "Сумма": r[4],
                }
                for r in last_orders
            ]
        )
    else:
        st.write("Пока нет оплат в этой области.")

    st.markdown("#### Последние 10 расходов")
    if last_exp:
        st.dataframe(
            [
                {
                    "Дата": r[0],
                    "Категория": r[1],
                    "Сумма": r[2],
                }
                for r in last_exp
            ]
        )
    else:
        st.write("Пока нет расходов в этой области.")


# =========================================================
# 4.2 UPLOAD
# =========================================================

with tab_upload:
    st.subheader("Загрузка данных")

    col_left, col_right = st.columns(2)

    # --------------------- ОПЛАТЫ -------------------------
    with col_left:
        st.markdown("### Оплаты")

        up_orders = st.file_uploader("CSV с оплатами", type=["csv"], key="up_orders")
        if up_orders is not None:
            ds_id, df = save_uploaded_file(up_orders, current_ws, "orders")
            st.session_state.orders_ds_id = ds_id
            st.session_state.orders_df = df
            st.success(f"Загружено {df.shape[0]} строк")
        st.markdown("Или Google Sheet (CSV):")
        sheet_url = st.text_input("URL Google Sheet (оплаты)", key="sheet_orders")
        if st.button("Подключить Sheet (оплаты)"):
            ds_id = save_google_sheet(sheet_url, current_ws, "orders")
            st.session_state.orders_ds_id = ds_id
            st.session_state.orders_df = pd.read_csv(parse_google_sheet_to_csv_url(sheet_url))
            st.success("Подключено")

    # --------------------- РАСХОДЫ -------------------------
    with col_right:
        st.markdown("### Расходы")

        up_exp = st.file_uploader("CSV с расходами", type=["csv"], key="up_expenses")
        if up_exp is not None:
            ds_id, df = save_uploaded_file(up_exp, current_ws, "expenses")
            st.session_state.expenses_ds_id = ds_id
            st.session_state.expenses_df = df
            st.success(f"Загружено {df.shape[0]} строк")
        st.markdown("Или Google Sheet (CSV):")
        sheet_url2 = st.text_input("URL Google Sheet (расходы)", key="sheet_exp")
        if st.button("Подключить Sheet (расходы)"):
            ds_id = save_google_sheet(sheet_url2, current_ws, "expenses")
            st.session_state.expenses_ds_id = ds_id
            st.session_state.expenses_df = pd.read_csv(parse_google_sheet_to_csv_url(sheet_url2))
            st.success("Подключено")


# =========================================================
# 4.3 MAPPING
# =========================================================

with tab_mapping:
    st.subheader("Маппинг")
    col_left, col_right = st.columns(2)

    # -------------------- МАППИНГ ОПЛАТ -------------------
    with col_left:
        st.markdown("### Маппинг оплат")

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
            label = st.selectbox("Источник оплат", labels, key="map_orders_source")
            order_source_id = next(r[0] for i, r in enumerate(order_sources) if labels[i] == label)

            # получаем схему
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

            with st.form(f"orders_form_{order_source_id}"):
                order_mapping = {}
                for f in suggest["orders"]:
                    tgt = f["target"]
                    wkey = f"ord_{order_source_id}_{tgt}"
                    options = ["— не выбрано —"] + detected
                    if wkey in st.session_state:
                        chosen = st.selectbox(
                            f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                            options=options,
                            key=wkey,
                        )
                    else:
                        sug_col = f.get("suggested_column")
                        idx = options.index(sug_col) if sug_col in detected else 0
                        chosen = st.selectbox(
                            f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                            options=options,
                            index=idx,
                            key=wkey,
                        )
                    order_mapping[tgt] = None if chosen == "— не выбрано —" else chosen

                submitted_orders = st.form_submit_button("Сохранить маппинг и запустить ETL (оплаты)")

            if submitted_orders:
                try:
                    if orders_df_raw is None:
                        # пробуем подтянуть снова
                        conn = get_conn()
                        c = conn.cursor()
                        c.execute("SELECT type, source_url FROM data_source WHERE id = ?", (order_source_id,))
                        ds_row = c.fetchone()
                        conn.close()
                        if ds_row and ds_row[0] == "google_sheets":
                            orders_df_raw = pd.read_csv(parse_google_sheet_to_csv_url(ds_row[1]))
                        else:
                            st.error("Нет данных этого источника в сессии. Загрузите файл заново.")
                            st.stop()

                    # очистка прошлых
                    conn = get_conn()
                    c = conn.cursor()
                    c.execute('DELETE FROM "order" WHERE data_source_id = ?', (order_source_id,))
                    conn.commit()
                    conn.close()

                    req = ["order_id", "order_date", "customer_name", "product", "revenue"]
                    miss = [r for r in req if not order_mapping.get(r)]
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
                        c.execute(
                            """
                            INSERT INTO metrics_snapshot
                            (id, workspace_id, period_from, period_to, payload_json, created_at)
                            VALUES (?,?,?,?,?,?)
                            """,
                            (snap_id, current_ws, None, None, json.dumps({"metrics": m, "insights": ins}), now),
                        )
                        for txt in ins:
                            c.execute(
                                """
                                INSERT INTO insight
                                (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                                VALUES (?,?,?,?,?,?)
                                """,
                                (str(uuid.uuid4()), current_ws, snap_id, txt, "rule", now),
                            )
                        c.execute(
                            "UPDATE data_source SET status = ?, updated_at = ? WHERE id = ?",
                            ("processed", now, order_source_id),
                        )
                        conn.commit()
                        conn.close()
                        st.success("Оплаты промапплены и загружены ✅")
                except Exception as e:
                    st.error(f"ETL по оплатам упал: {e}")

    # -------------------- МАППИНГ РАСХОДОВ ----------------
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
            label = st.selectbox("Источник расходов", labels, key="map_exp_source")
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

            with st.form(f"expenses_form_{exp_source_id}"):
                expense_mapping = {}
                for f in suggest["expenses"]:
                    tgt = f["target"]
                    wkey = f"exp_{exp_source_id}_{tgt}"
                    options = ["— не выбрано —"] + detected
                    if wkey in st.session_state:
                        chosen = st.selectbox(
                            f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                            options=options,
                            key=wkey,
                        )
                    else:
                        sug_col = f.get("suggested_column")
                        idx = options.index(sug_col) if sug_col in detected else 0
                        chosen = st.selectbox(
                            f'{f["label"]} ({tgt}) {"*" if f["required"] else ""}',
                            options=options,
                            index=idx,
                            key=wkey,
                        )
                    expense_mapping[tgt] = None if chosen == "— не выбрано —" else chosen

                submitted_expenses = st.form_submit_button("Сохранить маппинг и запустить ETL (расходы)")

            if submitted_expenses:
                try:
                    if exp_df_raw is None:
                        conn = get_conn()
                        c = conn.cursor()
                        c.execute("SELECT type, source_url FROM data_source WHERE id = ?", (exp_source_id,))
                        ds_row = c.fetchone()
                        conn.close()
                        if ds_row and ds_row[0] == "google_sheets":
                            exp_df_raw = pd.read_csv(parse_google_sheet_to_csv_url(ds_row[1]))
                        else:
                            st.error("Нет данных этого источника в сессии. Загрузите файл заново.")
                            st.stop()

                    conn = get_conn()
                    c = conn.cursor()
                    c.execute("DELETE FROM expense WHERE data_source_id = ?", (exp_source_id,))
                    conn.commit()
                    conn.close()

                    req = ["expense_date", "category", "amount"]
                    miss = [r for r in req if not expense_mapping.get(r)]
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
                        c.execute(
                            """
                            INSERT INTO metrics_snapshot
                            (id, workspace_id, period_from, period_to, payload_json, created_at)
                            VALUES (?,?,?,?,?,?)
                            """,
                            (snap_id, current_ws, None, None, json.dumps({"metrics": m, "insights": ins}), now),
                        )
                        for txt in ins:
                            c.execute(
                                """
                                INSERT INTO insight
                                (id, workspace_id, metrics_snapshot_id, text, rule_code, created_at)
                                VALUES (?,?,?,?,?,?)
                                """,
                                (str(uuid.uuid4()), current_ws, snap_id, txt, "rule", now),
                            )
                        c.execute(
                            "UPDATE data_source SET status = ?, updated_at = ? WHERE id = ?",
                            ("processed", now, exp_source_id),
                        )
                        conn.commit()
                        conn.close()
                        st.success("Расходы промапплены и загружены ✅")
                except Exception as e:
                    st.error(f"ETL по расходам упал: {e}")
