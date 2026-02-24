"""
agent.py – Intelligenter Agent (Orchestrierung)

Verantwortlich für:
    1. Bereitstellung von zwei spezialisierten Tools:
       - vector_search: Beantwortet Fragen zu PDF-Dokumenten (Document-RAG).
       - sql_query:     Beantwortet Fragen zu Messwerten/Zahlen (SQL-Analytics).
    2. Automatische Entscheidung, welches Tool basierend auf der Nutzeranfrage
       am besten geeignet ist.
    3. Rückgabe strukturierter Ergebnisse inklusive Agent-Logs für Transparenz.

Hinweis:
    Der Agent nutzt eine manuelle Orchestrierung (keine LangGraph-Agenten), da
    lokale LLMs via LM Studio ausschließlich 'user'- und 'assistant'-Rollen
    unterstützen. SystemMessage und ToolMessage werden NICHT verwendet.
"""

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from rag_engine import RAGSystem
from sql_engine import SQLEngine


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_TEMPERATURE = 0.1

# Prompt für die Tool-Auswahl (als User-Nachricht, nicht als System-Nachricht)
ROUTER_INSTRUCTION = (
    "Du bist ein intelligenter KI-Assistent für wissenschaftliche Mitarbeiter in der Industrie. "
    "Du hast Zugriff auf folgende Werkzeuge:\n\n"
    "{tool_descriptions}\n\n"
    "Entscheide, welches Werkzeug für die folgende Frage am besten geeignet ist. "
    "Antworte NUR mit einer JSON-Zeile in diesem Format (keine weiteren Erklärungen):\n"
    '  {{"tool": "<tool_name>", "input": "<die Eingabe für das Tool>"}}\n\n'
    "Falls kein Werkzeug nötig ist, antworte direkt mit:\n"
    '  {{"tool": "none", "input": ""}}\n\n'
    "Frage: {question}"
)

# Prompt für die finale Antwort nach Tool-Ausführung
ANSWER_INSTRUCTION = (
    "Du bist ein hilfreicher KI-Assistent für wissenschaftliche Mitarbeiter. "
    "Beantworte die Frage basierend auf dem folgenden Tool-Ergebnis. "
    "Antworte auf Deutsch, präzise und verständlich.\n\n"
    "Frage: {question}\n\n"
    "Tool-Ergebnis:\n{tool_result}\n\n"
    "Antwort:"
)


# ---------------------------------------------------------------------------
# Tool-Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Verwaltet die verfügbaren Tools und deren Ausführung."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str, func):
        """Registriert ein Tool mit Name, Beschreibung und Ausführungsfunktion."""
        self._tools[name] = {"description": description, "func": func}

    def get_descriptions(self) -> str:
        """Gibt eine formatierte Übersicht aller Tools zurück."""
        lines = []
        for name, info in self._tools.items():
            lines.append(f"- **{name}**: {info['description']}")
        return "\n".join(lines)

    def execute(self, name: str, input_text: str) -> str:
        """Führt ein Tool aus und gibt das Ergebnis zurück."""
        if name not in self._tools:
            return f"Unbekanntes Tool: {name}"
        try:
            return self._tools[name]["func"](input_text)
        except Exception as e:
            return f"Fehler bei {name}: {str(e)}"

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())


# ---------------------------------------------------------------------------
# Tool-Funktionen
# ---------------------------------------------------------------------------

def _make_vector_search(rag_system: RAGSystem):
    """Erstellt die vector_search Funktion."""
    def vector_search(query: str) -> str:
        try:
            result = rag_system.ask(query)
            sources = []
            for doc in result["source_documents"]:
                page = doc.metadata.get("page", "?")
                sources.append(f"[Seite {page}]")
            source_info = ", ".join(sorted(set(sources)))
            return f"{result['result']}\n\nQuellen: {source_info}"
        except Exception as e:
            return f"Fehler bei der Dokumentensuche: {str(e)}"
    return vector_search


def _make_sql_query(sql_engine: SQLEngine):
    """Erstellt die sql_query Funktion."""
    def sql_query(question: str) -> str:
        try:
            result = sql_engine.ask(question)
            return (
                f"{result['result']}\n\n"
                f"SQL-Abfrage: {result['sql_query']}\n"
                f"Ergebnis:\n{result['sql_result']}"
            )
        except ValueError as e:
            return f"SQL-Validierungsfehler: {str(e)}"
        except Exception as e:
            return f"Fehler bei der SQL-Abfrage: {str(e)}"
    return sql_query


# ---------------------------------------------------------------------------
# Haupt-Klasse: IntelligentAgent
# ---------------------------------------------------------------------------

class IntelligentAgent:
    """
    Orchestriert die Nutzung von Document-RAG und SQL-Analytics.

    Die Orchestrierung erfolgt manuell in zwei LLM-Aufrufen:
        1. Router-Call:  LLM wählt das passende Tool und formuliert den Input.
        2. Answer-Call:  LLM formuliert die finale Antwort aus dem Tool-Ergebnis.

    Alle Nachrichten verwenden ausschließlich 'user'/'assistant'-Rollen,
    damit lokale LLMs (LM Studio) fehlerfrei arbeiten.
    """

    def __init__(
        self,
        rag_system: RAGSystem | None = None,
        sql_engine: SQLEngine | None = None,
    ):
        """
        Initialisiert den Agent mit den verfügbaren Tools.

        Args:
            rag_system: Optionale RAG-System-Instanz (für Dokumentensuche).
            sql_engine: Optionale SQL-Engine-Instanz (für Datenbankabfragen).
        """
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=SecretStr(LLM_API_KEY),
            temperature=LLM_TEMPERATURE,
        )

        # Tool-Registry aufbauen
        self.registry = ToolRegistry()
        if rag_system is not None:
            self.registry.register(
                "vector_search",
                "Durchsucht PDF-Dokumente (Forschungsberichte, Handbücher) nach "
                "Konzepten, Methoden, Erklärungen und technischen Beschreibungen.",
                _make_vector_search(rag_system),
            )
        if sql_engine is not None:
            self.registry.register(
                "sql_query",
                "Greift auf die Industriedatenbank zu für Messwerte, Statistiken, "
                "Maschinenzustände, Anomalie-Logs und KI-Projekte.",
                _make_sql_query(sql_engine),
            )

        # Prompts (nur user-Rolle, kein SystemMessage)
        self.router_prompt = ChatPromptTemplate.from_template(ROUTER_INSTRUCTION)
        self.answer_prompt = ChatPromptTemplate.from_template(ANSWER_INSTRUCTION)

        # Chat-Historie für Memory
        self.chat_history: list[dict] = []

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _parse_tool_choice(self, llm_output: str) -> tuple[str, str]:
        """
        Parst die JSON-Antwort des Routers.

        Returns:
            Tuple (tool_name, tool_input). Bei Parse-Fehler: ("none", "").
        """
        # JSON aus der Antwort extrahieren (auch wenn Umgebungstext vorhanden)
        json_match = re.search(r'\{[^}]+\}', llm_output)
        if json_match:
            try:
                data = json.loads(json_match.group())
                tool_name = data.get("tool", "none").strip()
                tool_input = data.get("input", "").strip()
                return tool_name, tool_input
            except json.JSONDecodeError:
                pass
        return "none", ""

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Beantwortet eine Frage über den zweistufigen Agent-Workflow.

        Schritt 1: Router – LLM wählt Tool + Input.
        Schritt 2: Tool-Ausführung.
        Schritt 3: Answer – LLM formuliert finale Antwort.

        Args:
            question: Die Nutzeranfrage in natürlicher Sprache.

        Returns:
            Dictionary mit:
                - 'result':              Die endgültige Antwort (str).
                - 'intermediate_steps':  Agent-Logs (list[dict]).
                - 'tools_used':          Liste der genutzten Tool-Namen.
        """
        steps = []
        tools_used = []

        try:
            # --- Schritt 1: Tool-Auswahl durch LLM ---
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            router_output = router_chain.invoke({
                "tool_descriptions": self.registry.get_descriptions(),
                "question": question,
            })

            tool_name, tool_input = self._parse_tool_choice(router_output)

            # Routing-Schritt protokollieren
            steps.append({
                "action": "router",
                "action_input": question,
                "observation": f"Gewählt: {tool_name}" + (f" → {tool_input}" if tool_input else ""),
            })

            # --- Schritt 2: Tool ausführen (falls nötig) ---
            if tool_name != "none" and tool_name in self.registry.names:
                tool_result = self.registry.execute(tool_name, tool_input or question)
                tools_used.append(tool_name)

                steps.append({
                    "action": tool_name,
                    "action_input": tool_input or question,
                    "observation": tool_result[:2000],  # Für Logs kürzen
                })

                # --- Schritt 3: Finale Antwort formulieren ---
                answer_chain = self.answer_prompt | self.llm | StrOutputParser()
                final_answer = answer_chain.invoke({
                    "question": question,
                    "tool_result": tool_result,
                })

            else:
                # Kein Tool nötig – direkte Antwort vom LLM
                direct_chain = ChatPromptTemplate.from_template(
                    "Beantworte die folgende Frage auf Deutsch, präzise und hilfreich.\n\n"
                    "Frage: {question}\n\nAntwort:"
                ) | self.llm | StrOutputParser()
                final_answer = direct_chain.invoke({"question": question})

            # Chat-History aktualisieren
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": final_answer})

            return {
                "result": final_answer,
                "intermediate_steps": steps,
                "tools_used": tools_used,
            }

        except Exception as e:
            error_msg = f"Agent-Fehler: {str(e)}"
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": error_msg})
            return {
                "result": error_msg,
                "intermediate_steps": steps,
                "tools_used": tools_used,
            }

    def get_chat_history(self) -> list[dict]:
        """Gibt die bisherige Chat-Historie zurück."""
        return self.chat_history

    def clear_history(self) -> None:
        """Löscht die Chat-Historie."""
        self.chat_history = []
