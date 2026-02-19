# RAG (Retrieval-Augmented Generation)

This demo project allows you to upload PDF books and ask questions about their content. The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on the uploaded documents.

## Project Structure
The following structure is a recommendation for organizing your RAG project. You can of course adjust it to your needs, but it provides a clear separation between data, logic, and user interface.

```
RAG/
├── data/               # Store your PDF books here
├── app.py              # The main script (Streamlit UI)
├── rag_engine.py       # Logic for PDF processing & RAG
└── requirements.txt    # Required libraries
```

## Installation 

### Installation of python environment
To set up your Python environment, you can use conda to create a virtual environment:
```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

### Install required libraries

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the local LLM server (LM Studio)
1. Open **LM Studio**
2. Download the model **Mistral-7B-Instruct** (if not already downloaded)
3. Go to the **Local Server** tab (left sidebar)
4. Load the Mistral-7B-Instruct model
5. Click **Start Server** — it will run on `http://localhost:1234`
6. Verify the server is running: you should see "Server started" in the LM Studio logs

### 2. Run the Streamlit app
Make sure your conda environment is activated, then start the app:
```bash
conda activate rag_env
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`. You can now ask questions about the PDF books stored in the `data/` folder.

## Files Overview

### Main Script: app.py
This file contains the Streamlit user interface, allowing the user to upload PDFs, ask questions, and receive answers. It communicates with `rag_engine.py` to handle the logic for processing the PDFs and generating the answers.

### Backend: rag_engine.py
This file handles the "backend" – corresponding to the interface between the user and the backend mentioned in the job posting.