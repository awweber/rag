"""
rag_engine.py – Document-RAG Modul (Unstrukturierte Daten)

Verantwortlich für:
    1. Einlesen von PDF-Dokumenten aus ./data/docs/ via PyPDFLoader.
    2. Chunking mit RecursiveCharacterTextSplitter (600 Zeichen, 100 Overlap).
    3. Lokale Einbettung der Chunks mit sentence-transformers/all-MiniLM-L6-v2.
    4. Persistente Vektordatenbank (ChromaDB) für schnelle Ähnlichkeitssuche.
    5. Antwortgenerierung mit Quellenangaben (Source Citations).

Das Modul wird sowohl vom Agenten (agent.py) als auch direkt von der UI genutzt.
"""

import os
import glob
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
DOCS_DIR = os.path.join("data", "docs")
CHROMA_DIR = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
RETRIEVER_K = 3  # Anzahl zurückgegebener Chunks

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _collection_name_for(pdf_path: str) -> str:
    """Erzeugt einen stabilen Collection-Namen aus dem Dateipfad."""
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    # ChromaDB erlaubt max. 63 Zeichen und nur bestimmte Zeichen
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:50]
    suffix = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    return f"{safe}_{suffix}"


def get_available_pdfs() -> list[str]:
    """Gibt eine sortierte Liste aller PDF-Dateien in DOCS_DIR zurück."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    return sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))


def check_llm_connection() -> bool:
    """Prüft, ob der lokale LLM-Server (LM Studio) erreichbar ist."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{LLM_BASE_URL}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Embeddings (Singleton – wird nur einmal geladen)
# ---------------------------------------------------------------------------

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Gibt eine gemeinsame Embedding-Instanz zurück (Lazy-Singleton)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_instance


# ---------------------------------------------------------------------------
# Haupt-Klasse: RAGSystem
# ---------------------------------------------------------------------------

class RAGSystem:
    """
    Verwaltet den gesamten Document-RAG-Workflow für ein einzelnes PDF:
    Laden → Chunking → Embedding → Vektorsuche → LLM-Antwort mit Quellen.
    """

    def __init__(self, pdf_path: str):
        """
        Initialisiert das RAG-System für das gegebene PDF.

        Args:
            pdf_path: Pfad zur PDF-Datei (z.B. 'data/docs/report.pdf').
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")

        self.pdf_path = pdf_path
        self.collection_name = _collection_name_for(pdf_path)

        # 1. PDF laden und in Chunks aufteilen
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.chunks = text_splitter.split_documents(documents)

        # 2. Lokale Embeddings (läuft effizient auf CPU / Apple Silicon)
        self.embeddings = get_embeddings()

        # 3. Vektordatenbank – persistiert auf Disk für schnellen Neustart
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=CHROMA_DIR,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )

        # 4. LLM-Verbindung zu LM Studio (lokaler Server)
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=SecretStr(LLM_API_KEY),
            temperature=LLM_TEMPERATURE,
        )

        # 5. Prompt-Template mit klarer Anweisung zu Quellenangaben
        self.prompt = ChatPromptTemplate.from_template(
            "Du bist ein hilfreicher KI-Tutor für wissenschaftliche Mitarbeiter. "
            "Beantworte die Frage ausschließlich auf Basis des folgenden Kontexts "
            "aus technischen Dokumenten. Gib am Ende die verwendeten Seitenzahlen "
            "als Quellenangabe an. Wenn der Kontext die Antwort nicht enthält, "
            "sage dies ehrlich.\n\n"
            "Kontext:\n{context}\n\n"
            "Frage: {question}\n\n"
            "Antwort:"
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _format_docs(docs) -> str:
        """Formatiert Dokument-Chunks zu einem zusammenhängenden Text mit Quellenangaben."""
        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            parts.append(f"[Seite {page}]\n{doc.page_content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def search(self, query: str, k: int | None = None) -> list:
        """
        Führt eine reine Vektorsuche durch und gibt die relevantesten Chunks zurück.

        Args:
            query: Die Suchanfrage.
            k:     Anzahl der Ergebnisse (Standard: RETRIEVER_K).

        Returns:
            Liste von LangChain-Document-Objekten.
        """
        if k and k != RETRIEVER_K:
            return self.vectorstore.similarity_search(query, k=k)
        return self.retriever.invoke(query)

    def ask(self, query: str) -> dict:
        """
        Beantwortet eine Frage mittels RAG-Pipeline.

        Args:
            query: Die Nutzerfrage in natürlicher Sprache.

        Returns:
            Dictionary mit 'result' (Antworttext) und 'source_documents' (Chunks).
        """
        # Relevante Dokumente abrufen
        source_documents = self.retriever.invoke(query)

        # LCEL-Chain: Prompt → LLM → String-Parser
        chain = self.prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            "context": self._format_docs(source_documents),
            "question": query,
        })

        return {
            "result": result,
            "source_documents": source_documents,
        }