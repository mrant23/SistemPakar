from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

def initialize_rag_system():
    # Load and process the PDF
    loader = PyPDFLoader("dataset.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vectorstore and retriever
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=500
    )

    # Set up the system prompt
    system_template = (
    "Anda adalah sistem pakar yang ahli dalam pengelolaan limbah rumah tangga beracun, menggunakan teknologi Natural Language Processing (NLP). "
    "Tugas Anda adalah memberikan jawaban yang akurat, relevan, dan berbasis fakta untuk membantu pengguna mengelola limbah beracun dengan aman dan bertanggung jawab terhadap lingkungan. "
    "Selalu utamakan keselamatan manusia dan perlindungan lingkungan dalam setiap rekomendasi yang Anda berikan. "
    "Jika informasi tidak tersedia, sampaikan dengan jujur dan sarankan langkah-langkah umum yang aman. "
    "Berikan penjelasan yang ringkas, langsung ke inti masalah, dan sertakan langkah-langkah praktis jika memungkinkan. "
    "Jawaban Anda harus mencakup aspek pencegahan risiko, penanganan limbah yang tepat, dan dampak lingkungan dari setiap tindakan. "
    "Jika relevan, sertakan sumber daya tambahan seperti aturan hukum atau panduan umum terkait pengelolaan limbah. "
    "\n\n"
    "Konteks saat ini: {context}\n"
    "Riwayat percakapan sebelumnya: {chat_history}"
)


    # Create the prompt template with memory
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question"
    )

    # Create conversation chain
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    
    return retriever, conversation, memory

# Initialize the RAG system
retriever, conversation_chain, memory = initialize_rag_system()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/konsultasi', methods=['GET', 'POST'])
def konsultasi():
    if request.method == 'POST':
        query = request.form['query']
        
        try:
            # Get relevant documents
            retrieved_docs = retriever.get_relevant_documents(query)
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            
            # Get response using the conversation chain with memory
            response = conversation_chain.predict(
                question=query,
                context=context
            )
            
            # Get chat history for display
            chat_history = memory.load_memory_variables({})["chat_history"]
            
            return render_template('konsultasi.html', 
                                query=query, 
                                answer=response,
                                chat_history=chat_history,
                                include_meta=True)
        except Exception as e:
            print(f"Error: {str(e)}")  # For debugging
            return render_template('konsultasi.html', 
                                query=query, 
                                answer="Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.", 
                                include_meta=True)
    
    return render_template('konsultasi.html')

if __name__ == '__main__':
    app.run(debug=True)