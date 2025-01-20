from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

# Muat file konfigurasi dari .env
load_dotenv()
pdf_file = "dataset.pdf"  # Nama file PDF yang akan diproses

# List untuk menyimpan konten PDF
pdf_content = []

try:
    # Baca file PDF
    pdf_loader = PyPDFLoader(pdf_file)
    loaded_data = pdf_loader.load()
    pdf_content.extend(loaded_data)
    print(f"Sukses memuat {len(loaded_data)} halaman dari {pdf_file}")
except Exception as error:
    print(f"Terjadi kesalahan saat memuat PDF {pdf_file}: {error}")

if pdf_content:
    # Potong dokumen menjadi bagian-bagian kecil
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    split_docs = splitter.split_documents(pdf_content)
    print(f"Total bagian dokumen yang diproses: {len(split_docs)}")
else:
    print("Tidak ada data yang dapat diproses.")

# Inisialisasi embeddings dan penyimpanan vektor
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")

    # Simpan vektor ke direktori 'data'
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="data"
    )
    retriever_tool = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    print("Vektor berhasil dibuat dan disimpan.")
except Exception as error:
    print(f"Kesalahan saat inisialisasi embeddings atau penyimpanan vektor: {error}")

# Buat LLM dan chain untuk RAG
try:
    generative_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

    # Siapkan memori percakapan
    chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Buat template prompt dengan memori
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Anda adalah pakar yang fokus pada pengelolaan limbah rumah tangga berbahaya, menggunakan pendekatan NLP. "
                "Tugas Anda adalah memberikan jawaban faktual dan praktis untuk membantu pengguna mengelola limbah dengan aman. "
                "Pastikan keselamatan dan perlindungan lingkungan dalam setiap saran yang Anda berikan. "
                "Jika informasi tidak tersedia, berikan langkah-langkah umum yang aman dengan penjelasan singkat dan langsung ke pokok permasalahan. "
            ),
            MessagesPlaceholder(variable_name="chat_history"),  # Memori otomatis dimasukkan di sini
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Buat chain percakapan
    conversation_chain = LLMChain(
        llm=generative_model,
        prompt=prompt_template,
        memory=chat_memory,
        verbose=True
    )

    # Pertanyaan pengguna
    user_query = "Apa itu limbah rumah tangga beracun?"

    # Proses pertanyaan menggunakan chain
    response = conversation_chain.run(question=user_query)
    print("Jawaban:", response)

    # Ambil dokumen relevan berdasarkan similarity
    relevant_docs = retriever_tool.invoke(user_query)

    # Hitung kesamaan antara jawaban dan dokumen yang diambil
    similarities = []
    query_vector = embedding_model.embed_query(response)

    for doc in relevant_docs:
        doc_vector = embedding_model.embed_query(doc.page_content)
        similarity_score = cosine_similarity([query_vector], [doc_vector])[0][0]
        similarities.append((doc.page_content, similarity_score))

    # Urutkan hasil berdasarkan nilai kemiripan
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    print("Hasil pencarian berdasarkan tingkat kemiripan:")
    for content, score in similarities:
        print(f"Kemiripan: {score:.4f}")

except Exception as error:
    print(f"Terjadi kesalahan saat menginisialisasi model atau memori: {error}")
