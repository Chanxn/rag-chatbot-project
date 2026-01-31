import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# กำหนดโฟลเดอร์เก็บข้อมูล
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def create_vector_db():
    # 1. เช็คว่ามีโฟลเดอร์ data ไหม
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"สร้างโฟลเดอร์ {DATA_PATH} ให้แล้ว เอาไฟล์ PDF มาใส่ได้เลย")
        return

    # 2. ล้างข้อมูลเก่าทิ้ง (Reset DB) เพื่อไม่ให้ข้อมูลซ้ำซ้อน
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"ล้างฐานข้อมูลเก่าเรียบร้อย")

    print("mb กำลังโหลดเอกสารทุกไฟล์ในโฟลเดอร์ data/ ...")
    
    # --- จุดที่แก้ไข: อ่านทุกไฟล์ที่เป็น .pdf ในโฟลเดอร์ ---
    loader = DirectoryLoader(
        DATA_PATH,           # โฟลเดอร์เป้าหมาย
        glob="*.pdf",        # อ่านเฉพาะไฟล์ที่ลงท้ายด้วย .pdf
        loader_cls=PyPDFLoader # ใช้ตัวอ่าน PDF
    )
    documents = loader.load()
    
    if len(documents) == 0:
        print("ไม่พบไฟล์ PDF ในโฟลเดอร์ data/ เลยครับ")
        return

    print(f"พบเอกสารทั้งหมด {len(documents)} หน้า")

    # 3. หั่นข้อมูล (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"แยกข้อมูลได้ทั้งหมด {len(chunks)} ชิ้น")

    # 4. แปลงเป็น Vector และบันทึก
    print("กำลังสร้าง Vector Database (ใช้ HuggingFace Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("สร้างฐานข้อมูลสำเร็จ! พร้อมใช้งาน")

if __name__ == "__main__":
    create_vector_db()