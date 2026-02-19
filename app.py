import streamlit as st
from rag_engine import RAGSystem
import os
import glob

st.set_page_config(page_title="RAG Demo for AI Learning", page_icon="🤖")

st.title("AI Book Assistant")
st.markdown("---")

# Sidebar for PDF selection
st.sidebar.header("Configuration")

if not os.path.exists("data"):
    os.makedirs("data")

# Find all PDFs in the data folder
pdf_files = sorted(glob.glob("data/*.pdf"))

if not pdf_files:
    st.error("No PDF files found in the `data/` folder. Please add PDF books there.")
    st.stop()

pdf_labels = [os.path.basename(f) for f in pdf_files]
selected_pdf = st.sidebar.selectbox("Select a PDF book:", pdf_labels)
selected_path = os.path.join("data", selected_pdf)

st.info("This system uses a local LLM via LM Studio for maximum data privacy.")

# Initialize or re-initialize RAG when PDF selection changes
if "rag" not in st.session_state or st.session_state.get("current_pdf") != selected_path:
    with st.spinner(f"Indexing '{selected_pdf}'... This may take a moment."):
        try:
            st.session_state.rag = RAGSystem(selected_path)
            st.session_state.current_pdf = selected_path
            st.success(f"'{selected_pdf}' ready!")
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            st.stop()

query = st.text_input("Ask a question about the document:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.rag.ask(query)
            st.write("### Answer:")
            st.write(response["result"])

            with st.expander("Source chunks (context used for this answer)"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Chunk {i}** (Page {doc.metadata.get('page', '?')})")
                    st.caption(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error querying LLM. Is LM Studio running on localhost:1234?\n\n{e}")