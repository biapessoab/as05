import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_pdf_text(pdf_docs):
    """
    extrai texto de uma lista de pdfs
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    divide o texto em chunks para processamento pela llm
    define o tamanho dos chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    cria banco de vetores a partir dos chunks de texto.
    cada chunk é convertido em um embedding (vetor numérico)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain():
    """
    cadeia de conversação (RAG - Retrieval Augmented Generation).
    """
    # Define o template do prompt para o LLM.
    prompt = ChatPromptTemplate.from_template(
        """
    responda à pergunta do usuário com base apenas no contexto fornecido.
    se a resposta não for encontrada no contexto fornecido, diga que você não tem informações suficientes.

    contexto: {context}

    pergunta: {input}
    """
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        huggingfacehub_api_token=token,
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


st.set_page_config(
    page_title="Leitor e Analisador de PDF com LLM na Nuvem", layout="wide"
)
st.header("Analise seus PDFs")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Faça uma pergunta sobre o PDF")

if user_question:
    if "vector_store" not in st.session_state:
        st.error("Faça o upload e processe seus PDFs na barra lateral.")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        document_chain = get_conversational_chain()
        retriever = st.session_state.vector_store.as_retriever()

        rag_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Gerando resposta..."):  #
            response = rag_chain.invoke({"input": user_question})
            answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.title("Seus Documentos")
    pdf_docs = st.file_uploader(
        "Carregue seus arquivos PDF aqui e clique em 'Processar'",
        accept_multiple_files=True,
        type="pdf",
    )

    if st.button("Processar PDFs"):
        if pdf_docs:
            with st.spinner("Processando PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success(
                    "PDFs processados com sucesso! Agora você pode fazer perguntas."
                )
        else:
            st.warning("Carregue pelo menos um arquivo PDF para processar.")
