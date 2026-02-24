"""
sql_engine.py – SQL-Analytics Modul (Strukturierte Daten)

Verantwortlich für:
    1. Verbindung zur SQLite-Datenbank data/industrie_ki.db.
    2. Text-to-SQL: Natürliche Sprache → SQL-Query via LLM.
    3. Sichere Ausführung von SELECT-Abfragen (kein INSERT/UPDATE/DELETE).
    4. Analyse von Maschinenzuständen, Anomalie-Logs und Sensor-Statistiken.

Sicherheit:
    - Nur SELECT-Statements werden zugelassen.
    - Die Datenbank wird im Read-Only-Modus geöffnet (wo möglich).
    - SQL-Injection wird durch die LLM-generierte Query-Validierung minimiert.
"""

import os
import sqlite3
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
DB_PATH = os.path.join("data", "industrie_ki.db")

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_TEMPERATURE = 0.0  # Deterministisch für SQL-Generierung


# ---------------------------------------------------------------------------
# Datenbankschema-Beschreibung (wird dem LLM als Kontext übergeben)
# ---------------------------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE maschinen_status (
    id                INTEGER PRIMARY KEY,
    maschinen_name    TEXT    NOT NULL,  -- z.B. 'Fräse-01', 'Roboter-Arm-Alpha'
    abteilung         TEXT    NOT NULL,  -- 'Produktion', 'Montage', 'Instandhaltung', 'Umformung'
    llm_optimiert     BOOLEAN DEFAULT 0,
    letzte_anomalie   DATE,              -- Format: 'YYYY-MM-DD'
    rag_zugriff_aktiv BOOLEAN DEFAULT 0,
    betriebsstunden   INTEGER DEFAULT 0,
    zustand           TEXT    DEFAULT 'Normal'  -- 'Normal', 'Warnung', 'Kritisch'
);

CREATE TABLE anomalie_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    maschinen_id    INTEGER NOT NULL REFERENCES maschinen_status(id),
    zeitstempel     DATETIME NOT NULL,   -- Format: 'YYYY-MM-DD HH:MM:SS'
    schweregrad     TEXT,                 -- 'Niedrig', 'Mittel', 'Hoch', 'Kritisch'
    sensor_typ      TEXT    NOT NULL,     -- 'Vibration', 'Temperatur', 'Druck', 'Stromstärke'
    messwert        REAL    NOT NULL,
    schwellenwert   REAL    NOT NULL,
    beschreibung    TEXT
);

CREATE TABLE ki_projekte (
    id                          INTEGER PRIMARY KEY,
    projekt_name                TEXT    NOT NULL,
    technologie                 TEXT,
    effizienz_steigerung_prozent REAL,
    status                      TEXT   -- 'Abgeschlossen', 'In Arbeit', 'Testphase', 'Geplant'
);

CREATE TABLE sensor_statistiken (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    maschinen_id    INTEGER NOT NULL REFERENCES maschinen_status(id),
    monat           TEXT    NOT NULL,    -- Format: 'YYYY-MM'
    sensor_typ      TEXT    NOT NULL,
    mittelwert      REAL,
    maximum         REAL,
    minimum         REAL,
    standardabweichung REAL
);
""".strip()


# ---------------------------------------------------------------------------
# Haupt-Klasse: SQLEngine
# ---------------------------------------------------------------------------

class SQLEngine:
    """
    Ermöglicht natürlichsprachliche Abfragen auf der SQLite-Datenbank.

    Workflow:
        1. Nutzeranfrage → LLM generiert SQL-Query.
        2. SQL-Query wird validiert (nur SELECT erlaubt).
        3. Query wird gegen die Datenbank ausgeführt.
        4. Ergebnisse werden vom LLM in natürlicher Sprache zusammengefasst.
    """

    def __init__(self):
        """Initialisiert die SQL-Engine mit LLM-Verbindung."""
        if not os.path.isfile(DB_PATH):
            raise FileNotFoundError(
                f"Datenbank nicht gefunden: {DB_PATH}\n"
                "Bitte zuerst 'python setup_db.py' ausführen."
            )

        self.db_path = DB_PATH

        # LLM für Text-to-SQL (niedrige Temperatur für präzise Queries)
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=SecretStr(LLM_API_KEY),
            temperature=LLM_TEMPERATURE,
        )

        # Prompt: Natürliche Sprache → SQL (mit Few-Shot-Beispielen)
        self.sql_prompt = ChatPromptTemplate.from_template(
            "Du bist ein SQLite-Experte. Du schreibst SQL-Abfragen für genau diese Datenbank.\n\n"
            "=== DATENBANKSCHEMA (nur diese Tabellen und Spalten existieren!) ===\n\n"
            "{schema}\n\n"
            "=== BEISPIELE ===\n\n"
            "Frage: Welche Anomalien gab es zuletzt?\n"
            "SQL: SELECT a.id, m.maschinen_name, a.zeitstempel, a.schweregrad, a.sensor_typ, a.messwert, a.beschreibung FROM anomalie_log a JOIN maschinen_status m ON a.maschinen_id = m.id ORDER BY a.zeitstempel DESC LIMIT 10\n\n"
            "Frage: Welche Maschinen haben einen kritischen Zustand?\n"
            "SQL: SELECT maschinen_name, abteilung, zustand, betriebsstunden FROM maschinen_status WHERE zustand = 'Kritisch'\n\n"
            "Frage: Wie viele Anomalien hatte jede Maschine?\n"
            "SQL: SELECT m.maschinen_name, COUNT(*) AS anzahl FROM anomalie_log a JOIN maschinen_status m ON a.maschinen_id = m.id GROUP BY m.maschinen_name ORDER BY anzahl DESC\n\n"
            "Frage: Was ist der durchschnittliche Vibrationswert?\n"
            "SQL: SELECT mittelwert, maximum, minimum FROM sensor_statistiken WHERE sensor_typ = 'Vibration'\n\n"
            "=== REGELN ===\n"
            "- Verwende NUR die oben definierten Tabellen und Spalten.\n"
            "- Die Tabellen heißen: maschinen_status, anomalie_log, ki_projekte, sensor_statistiken.\n"
            "- Es gibt KEINE Tabelle 'anomalies', 'machines' oder andere englische Namen.\n"
            "- Es gibt KEINE Spalte 'anomalie_id', 'timestamp', 'type'. Die richtigen Namen stehen im Schema.\n"
            "- Gib NUR die SQL-Abfrage zurück. Kein Markdown, keine Erklärungen, keine Code-Blöcke.\n\n"
            "Frage: {question}\n"
            "SQL:"
        )

        # Prompt: SQL-Ergebnis → Natürliche Sprache
        self.answer_prompt = ChatPromptTemplate.from_template(
            "Du bist ein hilfreicher Assistent für Industriedaten. "
            "Beantworte die Frage basierend auf den SQL-Abfrageergebnissen. "
            "Formuliere eine klare und verständliche Antwort auf Deutsch.\n\n"
            "Frage: {question}\n\n"
            "SQL-Abfrage:\n{sql_query}\n\n"
            "Ergebnis:\n{sql_result}\n\n"
            "Antwort:"
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_sql(sql: str) -> str:
        """
        Validiert und bereinigt die generierte SQL-Abfrage.

        Erlaubt nur SELECT-Statements. Entfernt Markdown-Artefakte.

        Args:
            sql: Die vom LLM generierte SQL-Abfrage.

        Returns:
            Die bereinigte SQL-Abfrage.

        Raises:
            ValueError: Wenn die Abfrage kein SELECT-Statement ist.
        """
        # Markdown-Code-Blöcke entfernen
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql)
        sql = sql.strip().rstrip(";") + ";"

        # Sicherheits-Check: Nur SELECT erlauben
        first_keyword = sql.strip().split()[0].upper() if sql.strip() else ""
        if first_keyword != "SELECT":
            raise ValueError(
                f"Nur SELECT-Abfragen sind erlaubt. Erkannt: '{first_keyword}'"
            )

        # Gefährliche Schlüsselwörter prüfen
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
        sql_upper = sql.upper()
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(
                    f"Unzulässiges SQL-Schlüsselwort erkannt: {keyword}"
                )

        return sql

    def _execute_sql(self, sql: str) -> tuple[list[str], list[tuple]]:
        """
        Führt eine validierte SQL-Abfrage gegen die Datenbank aus.

        Args:
            sql: Die SQL-Abfrage (muss bereits validiert sein).

        Returns:
            Tuple aus (Spaltennamen, Ergebniszeilen).
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return columns, rows
        finally:
            conn.close()

    @staticmethod
    def _format_result(columns: list[str], rows: list[tuple]) -> str:
        """Formatiert SQL-Ergebnisse als lesbare Tabelle."""
        if not rows:
            return "Keine Ergebnisse gefunden."

        # Header
        header = " | ".join(columns)
        separator = "-+-".join("-" * max(len(c), 8) for c in columns)
        lines = [header, separator]

        # Zeilen (maximal 50 für die Ausgabe)
        for row in rows[:50]:
            lines.append(" | ".join(str(v) for v in row))

        if len(rows) > 50:
            lines.append(f"... ({len(rows) - 50} weitere Zeilen)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Beantwortet eine natürlichsprachliche Frage über die Datenbank.

        Workflow: Frage → SQL-Generierung → Validierung → Ausführung → Antwort.

        Args:
            question: Die Nutzerfrage (z.B. "Welche Maschinen haben kritische Anomalien?").

        Returns:
            Dictionary mit:
                - 'result': Natürlichsprachliche Antwort.
                - 'sql_query': Die generierte SQL-Abfrage.
                - 'sql_result': Rohe Ergebnistabelle.
                - 'columns': Spaltennamen.
                - 'rows': Ergebniszeilen.
        """
        # Schritt 1: SQL-Query generieren
        sql_chain = self.sql_prompt | self.llm | StrOutputParser()
        raw_sql = sql_chain.invoke({
            "schema": DB_SCHEMA,
            "question": question,
        })

        # Schritt 2: Validierung
        sql_query = self._validate_sql(raw_sql)

        # Schritt 3: Ausführung
        columns, rows = self._execute_sql(sql_query)
        formatted_result = self._format_result(columns, rows)

        # Schritt 4: Natürlichsprachliche Antwort generieren
        answer_chain = self.answer_prompt | self.llm | StrOutputParser()
        result = answer_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "sql_result": formatted_result,
        })

        return {
            "result": result,
            "sql_query": sql_query,
            "sql_result": formatted_result,
            "columns": columns,
            "rows": rows,
        }

    def get_schema_info(self) -> str:
        """Gibt die Schema-Beschreibung zurück (nützlich für Debugging / UI)."""
        return DB_SCHEMA

    def get_table_summary(self) -> dict[str, int]:
        """Gibt die Anzahl der Einträge pro Tabelle zurück."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            tables = {}
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                tables[table_name] = cursor.fetchone()[0]
            return tables
        finally:
            conn.close()
