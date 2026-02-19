from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

class RAGSystem:
    def __init__(self, pdf_path):
        # 1. Load document & split into chunks
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        self.chunks = text_splitter.split_documents(documents)
        
        # 2. Local embeddings (runs efficiently on Mac)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Vector database in memory
        self.vectorstore = Chroma.from_documents(documents=self.chunks, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. Connection to LM Studio (local server must be running!)
        self.llm = ChatOpenAI(
            base_url="http://localhost:1234/v1", 
            api_key=SecretStr("lm-studio"),
            temperature=0.2
        )

        # 5. RAG prompt template
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful AI tutor. Answer the question based ONLY on the following context "
            "from AI/ML textbooks. If the context doesn't contain the answer, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, query):
        # Retrieve relevant documents
        source_documents = self.retriever.invoke(query)

        # Build LCEL chain
        chain = self.prompt | self.llm | StrOutputParser()

        # Run chain with context and question
        result = chain.invoke({
            "context": self._format_docs(source_documents),
            "question": query
        })

        return {
            "result": result,
            "source_documents": source_documents
        }