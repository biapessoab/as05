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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    logger.warning(
        "Token da HuggingFace não encontrado no .env! Certifique-se de configurá-lo."
    )


def get_pdf_text(pdf_docs):
    """
    Extrai texto de uma lista de PDFs.
    """
    logger.info("Extraindo texto de %d PDFs", len(pdf_docs))
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    logger.info("Extração de texto finalizada. Tamanho: %d caracteres", len(text))
    return text


def get_text_chunks(text):
    """
    Divide o texto em pedaços (chunks) para processamento pelo LLM.
    Define o tamanho dos chunks.
    """
    logger.info("Dividindo texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    logger.info("Número de chunks gerados: %d", len(chunks))
    return chunks


def get_vector_store(text_chunks):
    """
    Cria um banco de vetores a partir dos chunks de texto.
    """
    logger.info("Gerando embeddings e criando vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    logger.info("Vector store criado com sucesso.")
    return vector_store


def get_conversational_chain():
    """
    Define a cadeia de conversação (RAG - Retrieval Augmented Generation).
    Isso inclui o prompt para o LLM e a configuração do modelo do Hugging Face Hub.
    """
    logger.info("Inicializando cadeia de conversação com LLM da HuggingFace...")

    prompt_template = """Você é um assistente útil. Responda à pergunta com base no contexto, **em português**, de forma breve e precisa.
Se não souber, diga que não sabe.

[Contexto]
{context}

[Pergunta]
{input}

[Resposta]
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    try:
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        )
        logger.info("LLM carregada com sucesso: HuggingFaceH4/zephyr-7b-beta.")
    except Exception as e:
        logger.error("Erro ao carregar HuggingFaceHub LLM: %s", e)
        raise

    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


# --- STREAMLIT APP ---
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

        try:
            logger.info("Criando cadeia de resposta RAG...")
            document_chain = get_conversational_chain()
            retriever = st.session_state.vector_store.as_retriever()
            rag_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Gerando resposta..."):
                logger.info("Chamando cadeia RAG com pergunta: %s", user_question)
                response = rag_chain.invoke({"input": user_question})
                answer_raw = response["answer"]
                final_answer = answer_raw.strip()

            with st.chat_message("assistant"):
                st.markdown(final_answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )
        except Exception as e:
            logger.exception("Erro ao gerar resposta: %s", e)
            st.error(f"Erro ao gerar resposta: {e}")

with st.sidebar:
    st.title("Seus Documentos")
    pdf_docs = st.file_uploader(
        "Carregue seus arquivos PDF aqui e clique em 'Processar'",
        accept_multiple_files=True,
        type="pdf",
    )

    if st.button("Processar PDFs"):
        if pdf_docs:
            try:
                with st.spinner("Processando PDFs..."):
                    logger.info("Iniciando processamento de PDFs.")
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success(
                        "PDFs processados com sucesso! Agora você pode fazer perguntas."
                    )
            except Exception as e:
                logger.exception("Erro ao processar PDFs: %s", e)
                st.error(f"Erro ao processar PDFs: {e}")
        else:
            st.warning("Carregue pelo menos um arquivo PDF para processar.")
