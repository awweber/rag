"""
app.py – Streamlit Web-Oberfläche für das Industrie-KI-Assistenzsystem

Features:
    - Chat-Interface mit Konversationshistorie (Memory).
    - Sidebar: LLM-Verbindungsstatus, PDF-Auswahl, Datenbank-Info.
    - Transparenz: Expander-Widgets für Agent-Logs und abgerufene Kontexte.
    - Intelligenter Agent orchestriert Document-RAG und SQL-Analytics.

Start:
    streamlit run app.py
"""

import streamlit as st
import os

from rag_engine import RAGSystem, get_available_pdfs, check_llm_connection, DOCS_DIR
from sql_engine import SQLEngine, DB_PATH
from agent import IntelligentAgent

# ---------------------------------------------------------------------------
# Seiten-Konfiguration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KI Assistent",
    page_icon="🏭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-State initialisieren
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "sql_engine" not in st.session_state:
    st.session_state.sql_engine = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []


# ---------------------------------------------------------------------------
# Sidebar: Konfiguration & Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfiguration")

    # --- LLM-Verbindungsstatus ---
    st.subheader("🔌 LLM-Server Status")
    if st.button("Verbindung prüfen", use_container_width=True):
        st.session_state.llm_connected = check_llm_connection()

    llm_connected = st.session_state.get("llm_connected", None)
    if llm_connected is True:
        st.success("✅ LM Studio verbunden (localhost:1234)")
    elif llm_connected is False:
        st.error("❌ LM Studio nicht erreichbar")
        st.caption("Stelle sicher, dass LM Studio läuft und der Server auf Port 1234 gestartet ist.")
    else:
        st.info("Status noch nicht geprüft – klicke oben.")

    st.markdown("---")

    # --- PDF-Auswahl ---
    st.subheader("📄 Dokument-RAG")
    pdf_files = get_available_pdfs()

    if not pdf_files:
        st.warning(f"Keine PDFs in `{DOCS_DIR}/` gefunden.")
        selected_pdf_path = None
    else:
        pdf_labels = [os.path.basename(f) for f in pdf_files]
        selected_label = st.selectbox("PDF-Dokument:", pdf_labels)
        selected_pdf_path = os.path.join(DOCS_DIR, selected_label)

    st.markdown("---")

    # --- Datenbank-Info ---
    st.subheader("🗄️ SQL-Datenbank")
    if os.path.isfile(DB_PATH):
        st.success(f"✅ `{DB_PATH}` vorhanden")
        # Tabellen-Übersicht laden
        try:
            if st.session_state.sql_engine is None:
                st.session_state.sql_engine = SQLEngine()
            summary = st.session_state.sql_engine.get_table_summary()
            for table, count in summary.items():
                st.caption(f"  • {table}: {count} Einträge")
        except Exception as e:
            st.caption(f"Fehler beim Lesen: {e}")
    else:
        st.error(f"❌ `{DB_PATH}` nicht gefunden")
        st.caption("Führe `python setup_db.py` aus.")

    st.markdown("---")

    # --- Chat zurücksetzen ---
    if st.button("🗑️ Chat zurücksetzen", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_logs = []
        if st.session_state.agent:
            st.session_state.agent.clear_history()
        st.rerun()


# ---------------------------------------------------------------------------
# RAG-System initialisieren / aktualisieren
# ---------------------------------------------------------------------------
def initialize_rag(pdf_path: str) -> bool:
    """Initialisiert oder aktualisiert das RAG-System für das gewählte PDF."""
    if st.session_state.current_pdf == pdf_path and st.session_state.rag is not None:
        return True  # Bereits geladen

    with st.spinner(f"📚 Indexiere '{os.path.basename(pdf_path)}'..."):
        try:
            st.session_state.rag = RAGSystem(pdf_path)
            st.session_state.current_pdf = pdf_path
            return True
        except Exception as e:
            st.error(f"Fehler beim Laden des PDF: {e}")
            return False


def initialize_agent() -> bool:
    """Erstellt den Agenten mit den aktuell verfügbaren Tools."""
    try:
        # SQL-Engine initialisieren (falls noch nicht geschehen)
        if st.session_state.sql_engine is None and os.path.isfile(DB_PATH):
            st.session_state.sql_engine = SQLEngine()

        st.session_state.agent = IntelligentAgent(
            rag_system=st.session_state.rag,
            sql_engine=st.session_state.sql_engine,
        )
        return True
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Agenten: {e}")
        return False


# ---------------------------------------------------------------------------
# Hauptbereich: Titel & Info
# ---------------------------------------------------------------------------
st.title("🏭 Industrie-KI Assistent")
st.caption(
    "Verknüpft Wissen aus PDF-Dokumenten mit strukturierten Messdaten. "
    "Alle Daten werden lokal verarbeitet – maximale Datensouveränität."
)

# RAG initialisieren wenn ein PDF ausgewählt ist
if selected_pdf_path and selected_pdf_path != st.session_state.current_pdf:
    if initialize_rag(selected_pdf_path):
        # Agent mit neuem RAG-System neu erstellen
        initialize_agent()
        st.toast(f"'{os.path.basename(selected_pdf_path)}' geladen!", icon="📄")

# Agent erstellen falls noch nicht vorhanden
if st.session_state.agent is None:
    if selected_pdf_path:
        if st.session_state.rag is None:
            initialize_rag(selected_pdf_path)
    initialize_agent()


# ---------------------------------------------------------------------------
# Hilfsfunktion: Agent-Details rendern
# ---------------------------------------------------------------------------
def _render_agent_details(msg: dict):
    """Zeigt Agent-Details (genutzte Tools, Zwischenschritte) in Expandern."""
    if "tools_used" in msg and msg["tools_used"]:
        tool_names = ", ".join(msg["tools_used"])
        st.caption(f"🔧 Genutzte Tools: {tool_names}")

    if "logs" in msg and msg["logs"]:
        with st.expander("🧠 Gedankengang des Agenten", expanded=False):
            for i, step in enumerate(msg["logs"], 1):
                action = step.get("action", "?")
                action_input = step.get("action_input", "")
                observation = step.get("observation", "")

                st.markdown(f"**Schritt {i}: {action}**")
                if action_input:
                    st.code(action_input, language="text")
                if observation:
                    st.text_area(
                        f"Ergebnis Schritt {i}",
                        observation,
                        height=120,
                        disabled=True,
                        key=f"log_{msg.get('idx', i)}_{i}",
                    )
                st.markdown("---")

    if "sql_query" in msg:
        with st.expander("📊 SQL-Abfrage", expanded=False):
            st.code(msg["sql_query"], language="sql")
            if "sql_result" in msg:
                st.text(msg["sql_result"])

    if "source_documents" in msg:
        with st.expander("📄 Quellen-Chunks (Kontext)", expanded=False):
            for j, doc_info in enumerate(msg["source_documents"], 1):
                st.markdown(f"**Chunk {j}** (Seite {doc_info.get('page', '?')})")
                st.caption(doc_info.get("content", ""))
                st.markdown("---")


# ---------------------------------------------------------------------------
# Chat-Verlauf anzeigen
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Agent-Logs für diese Nachricht anzeigen (falls vorhanden)
        if msg["role"] == "assistant" and "logs" in msg:
            _render_agent_details(msg)


# ---------------------------------------------------------------------------
# Chat-Eingabe
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Stelle eine Frage zu Dokumenten oder Messdaten..."):

    # Nutzernachricht anzeigen und speichern
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent-Antwort generieren
    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            st.error(
                "Agent konnte nicht initialisiert werden. "
                "Bitte prüfe die LLM-Verbindung und lade ein PDF-Dokument."
            )
        else:
            with st.spinner("🤔 Agent denkt nach..."):
                try:
                    response = st.session_state.agent.ask(prompt)
                    result = response["result"]

                    # Antwort anzeigen
                    st.markdown(result)

                    # Nachricht mit Metadaten speichern
                    msg_data = {
                        "role": "assistant",
                        "content": result,
                        "tools_used": response.get("tools_used", []),
                        "idx": len(st.session_state.messages),
                    }

                    # Intermediate Steps aufbereiten (Format: list[dict])
                    logs = []
                    for step in response.get("intermediate_steps", []):
                        # Neues Format: step ist bereits ein dict mit action/action_input/observation
                        log_entry = {
                            "action": step.get("action", "?"),
                            "action_input": step.get("action_input", ""),
                            "observation": str(step.get("observation", ""))[:2000],
                        }
                        logs.append(log_entry)
                        observation_str = str(step.get("observation", ""))

                        # SQL-Query extrahieren falls vorhanden
                        if "SQL-Abfrage:" in observation_str:
                            try:
                                sql_part = observation_str.split("SQL-Abfrage:")[1]
                                sql_q = sql_part.split("\n")[0].strip()
                                msg_data["sql_query"] = sql_q
                                rest = sql_part.split("Ergebnis:\n", 1)
                                if len(rest) > 1:
                                    msg_data["sql_result"] = rest[1][:1000]
                            except (IndexError, ValueError):
                                pass

                        # Quellen extrahieren falls vorhanden
                        if step.get("action") == "vector_search":
                            if "Quellen:" in observation_str:
                                msg_data["source_documents"] = [
                                    {"page": "siehe Agent-Log", "content": observation_str[:500]}
                                ]

                    msg_data["logs"] = logs
                    st.session_state.messages.append(msg_data)

                    # Genutzte Tools anzeigen
                    if response.get("tools_used"):
                        tool_names = ", ".join(response["tools_used"])
                        st.caption(f"🔧 Genutzte Tools: {tool_names}")

                    # Agent-Logs anzeigen
                    if logs:
                        with st.expander("🧠 Gedankengang des Agenten", expanded=False):
                            for k, log in enumerate(logs, 1):
                                st.markdown(f"**Schritt {k}: {log['action']}**")
                                if log["action_input"]:
                                    inp = log["action_input"]
                                    if isinstance(inp, dict):
                                        inp = str(inp)
                                    st.code(inp, language="text")
                                if log["observation"]:
                                    st.text_area(
                                        f"Ergebnis Schritt {k}",
                                        log["observation"],
                                        height=120,
                                        disabled=True,
                                        key=f"current_log_{k}",
                                    )
                                st.markdown("---")

                    # SQL-Details anzeigen
                    if "sql_query" in msg_data:
                        with st.expander("📊 SQL-Abfrage", expanded=False):
                            st.code(msg_data["sql_query"], language="sql")
                            if "sql_result" in msg_data:
                                st.text(msg_data["sql_result"])

                except Exception as e:
                    error_msg = (
                        f"Fehler bei der Verarbeitung. Ist LM Studio auf localhost:1234 aktiv?\n\n"
                        f"Details: {e}"
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {error_msg}",
                    })