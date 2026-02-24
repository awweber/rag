# Industrie-KI Assistent – RAG + SQL Agent

Ein Assistenzsystem für wissenschaftliche Mitarbeiter, das Wissen aus unstrukturierten PDFs (Forschungsberichte) mit strukturierten Messdaten (SQL) verknüpft. Der Fokus liegt auf **Datensouveränität** durch lokale LLM-Inferenz via LM Studio.

## Architektur

```
Nutzeranfrage
      │
      ▼
┌─────────────┐
│  Streamlit   │  ← Chat-Interface mit Memory
│   (app.py)   │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│  Intelligenter   │  ← Zweistufiger Router-Agent
│  Agent (agent.py)│
└───┬─────────┬───┘
    │         │
    ▼         ▼
┌────────┐ ┌───────────┐
│ vector │ │ sql_query  │
│_search │ │            │
└───┬────┘ └─────┬─────┘
    │            │
    ▼            ▼
┌────────────┐ ┌──────────────┐
│ ChromaDB   │ │ SQLite       │
│ (PDFs)     │ │ (Messdaten)  │
│rag_engine  │ │ sql_engine   │
└────────────┘ └──────────────┘
```

## Projektstruktur

```
RAG/
├── data/
│   ├── docs/              # PDF-Dokumente hier ablegen
│   ├── industrie_ki.db    # SQLite-Datenbank (wird von setup_db.py erstellt)
│   └── chroma_db/         # Persistente Vektordatenbank (wird automatisch erstellt)
├── app.py                 # Streamlit Web-Oberfläche (UI)
├── rag_engine.py          # Modul A: Document-RAG (unstrukturierte Daten)
├── sql_engine.py          # Modul B: SQL-Analytics (strukturierte Daten)
├── agent.py               # Modul C: Intelligenter Agent (Orchestrierung)
├── setup_db.py            # SQLite-Datenbank erstellen & befüllen
├── requirements.txt       # Python-Abhängigkeiten
└── README.md
```

## Installation

### 1. Python-Umgebung erstellen

```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3. SQLite-Datenbank erstellen

```bash
python setup_db.py
```

### 4. PDF-Dokumente ablegen

Lege deine PDF-Dateien (Forschungsberichte, Handbücher) in den Ordner `data/docs/`:

```bash
cp mein_dokument.pdf data/docs/
```

## Nutzung

### 1. LM Studio starten

1. **LM Studio** öffnen
2. Modell **Mistral-7B-Instruct** herunterladen (falls nötig)
3. **Local Server** Tab → Modell laden → **Start Server** (Port 1234)
4. Verifizieren: „Server started" in den LM Studio Logs

### 2. Streamlit-App starten

```bash
conda activate rag_env
streamlit run app.py
```

Die App öffnet sich unter `http://localhost:8501`.

## Features

### Modul A: Document-RAG (Unstrukturiert)
- **Loader**: `PyPDFLoader` liest PDFs aus `./data/docs/`
- **Chunking**: `RecursiveCharacterTextSplitter` (600 Zeichen, 100 Overlap)
- **Embeddings**: `all-MiniLM-L6-v2` (lokal, keine API nötig)
- **Vektordatenbank**: ChromaDB mit Persistenz auf Disk
- **Quellenangaben**: Seitenzahlen werden in Antworten referenziert

### Modul B: SQL-Analytics (Strukturiert)
- **Datenbank**: SQLite (`data/industrie_ki.db`)
- **Text-to-SQL**: LLM generiert SQL aus natürlicher Sprache
- **Sicherheit**: Nur SELECT-Abfragen erlaubt, gefährliche Keywords blockiert
- **Tabellen**: Maschinenstatus, Anomalie-Logs, KI-Projekte, Sensor-Statistiken

### Modul C: Intelligenter Agent (Orchestrierung)
- **Zweistufiger Router-Agent** (kompatibel mit lokalen LLMs, die nur user/assistant-Rollen unterstützen)
  - **Schritt 1 – Router**: LLM wählt das passende Tool und formuliert den Input (JSON-Antwort)
  - **Schritt 2 – Answer**: LLM formuliert die finale Antwort aus dem Tool-Ergebnis
- **Tool 1** (`vector_search`): Für Fragen zu Dokumenten/Konzepten
- **Tool 2** (`sql_query`): Für Fragen zu Messwerten/Zahlen
- **Transparenz**: Agent-Logs zeigen Routing-Entscheidung und Tool-Ergebnisse

### UI/UX
- **Chat-Interface** mit Konversationshistorie
- **LLM-Verbindungsstatus** in der Sidebar
- **Expander-Widgets** für Agent-Logs und Quellen-Chunks
- **PDF-Auswahl** und Datenbank-Übersicht in der Sidebar

## Routing-Ansätze im Überblick

Der Agent muss eingehende Fragen dem richtigen Tool zuweisen. Dafür existieren verschiedene Routing-Strategien:

| Ansatz | Funktionsweise | Latenz | Genauigkeit | Benötigt LLM? |
|---|---|---|---|---|
| **Keyword-basiert** | Regelwerk mit Schlüsselwörtern (z. B. *„Sensor"* → SQL, *„Konzept"* → RAG) | ⚡ Sehr gering | Niedrig – versagt bei Umschreibungen | Nein |
| **LLM-basiert** ⭐ | LLM analysiert die Frage und wählt das Tool per JSON-Antwort | 🐢 Hoch (LLM-Call) | Hoch – versteht Kontext und Nuancen | Ja |
| **Semantisches Routing** | Embedding der Frage wird per Kosinus-Ähnlichkeit mit Referenz-Embeddings verglichen | ⚡ Gering | Mittel–Hoch | Nein (nur Embedding-Modell) |
| **Klassifikation (trainiert)** | Supervised-Modell (z. B. Logistic Regression, BERT) auf gelabelten Beispielen | ⚡ Gering | Sehr hoch (mit guten Trainingsdaten) | Nein |
| **Zero-Shot-Klassifikation** | Vortrainiertes NLI-Modell ordnet Fragen Kategorien zu (z. B. `bart-large-mnli`) | 🔶 Mittel | Mittel–Hoch | Nein (NLI-Modell) |
| **Hybrid** | Kombination mehrerer Ansätze (z. B. Keyword-Vorfilter + Semantic Fallback) | 🔶 Variabel | Hoch | Optional |

### Aktuelle Implementierung

Dieses Projekt verwendet **LLM-basiertes Routing** (Zweistufiger Router-Agent in `agent.py`):

1. **Router-Schritt**: Ein `ChatPromptTemplate` beschreibt die verfügbaren Tools. Das LLM antwortet mit `{"tool": "<name>", "input": "<input>"}`.
2. **Answer-Schritt**: Das Tool-Ergebnis wird dem LLM übergeben, das die finale Nutzerantwort formuliert.

**Warum LLM-basiert?** Für einen Prototyp mit zwei Tools bietet LLM-Routing die beste Balance aus Genauigkeit und Implementierungsaufwand. Es erfordert keine Trainingsdaten und versteht auch umformulierte oder mehrdeutige Fragen zuverlässig. Bei Skalierung auf viele Tools wäre ein Wechsel zu semantischem oder hybridem Routing sinnvoll.

## Technischer Stack

| Komponente | Technologie | Paket |
|---|---|---|
| Sprache | Python 3.11+ | – |
| Frontend | Streamlit | `streamlit` |
| LLM | LM Studio (Mistral-7B, lokal) | `langchain-openai` |
| LLM-Framework | LangChain (OpenAI-kompatible API) | `langchain`, `langchain-core` |
| Vektordatenbank | ChromaDB | `langchain-chroma`, `chromadb` |
| Relationale DB | SQLite | Python-Standard |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | `langchain-huggingface`, `sentence-transformers` |
| PDF-Loader | PyPDFLoader | `langchain-community`, `pypdf` |
| Text-Splitting | RecursiveCharacterTextSplitter | `langchain-text-splitters` |

## Pipeline-Details

| Schritt | Library | Import-Pfad | Zweck |
|---|---|---|---|
| PDF-Laden | `PyPDFLoader` | `langchain_community.document_loaders` | PDF-Seiten extrahieren |
| Text-Splitting | `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Chunks erzeugen (600 Zeichen, 100 Overlap) |
| Embeddings | `HuggingFaceEmbeddings` | `langchain_huggingface` | Lokale Vektorisierung mit `all-MiniLM-L6-v2` |
| Vektorspeicher | `Chroma` | `langchain_chroma` | Ähnlichkeitssuche auf Dokumenten-Chunks |
| LLM-Anbindung | `ChatOpenAI` | `langchain_openai` | Verbindung zu LM Studio (localhost:1234) |
| Prompts | `ChatPromptTemplate` | `langchain_core.prompts` | Prompt-Templates für alle LLM-Aufrufe |
| Text-to-SQL | `ChatOpenAI` + Few-Shot Prompt | `langchain_openai` | Natürliche Sprache → SQL SELECT |
| Agent | Manueller Router-Agent | `agent.py` (eigene Implementierung) | Zweistufige Tool-Auswahl (Router → Answer) |
| UI | `Streamlit` | `streamlit` | Chat, Sidebar, Expander-Widgets |
