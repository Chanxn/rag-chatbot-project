import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

st.title("RAG Chatbot (Gemini)")
st.caption("ถาม-ตอบ จากเอกสารด้วย Google Gemini (Free Version)")

# 1. Setup Embeddings (ต้องใช้ตัวเดิมกับตอน ingest)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load Database
if os.path.exists("./chroma_db"):
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Setup LLM (Gemini)
    # แนะนำสำหรับทดสอบ model="models/gemini-flash-latest" 
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # UI Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("ถามอะไรดีครับ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini กำลังอ่านเอกสาร..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    source_docs = response['source_documents']
                    
                    st.markdown(answer)
                    
                    with st.expander("อ้างอิงจากเอกสาร"):
                        for doc in source_docs:
                            st.write(f"- {doc.page_content[:100]}...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.warning("⚠️ ไม่พบฐานข้อมูล กรุณารันไฟล์ ingest.py ก่อนครับ")