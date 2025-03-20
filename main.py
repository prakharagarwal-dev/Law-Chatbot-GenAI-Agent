import os

import fitz
import groq
from pinecone import Pinecone, ServerlessSpec
from twilio.rest import Client
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import re
import langdetect

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Required to avoid tokenization warnings

# Pinecone Initialization
pc = Pinecone(api_key="pcsk_25hjbe_LfS2dWQeKo9FfzCPuhviPtdMD76trCXwN9w1DdQcfsvLs2eSd4CjTKjM7wMge1h")

# Check if index already exists (it shouldn't if this is first time running the demo)
if not pc.has_index(name="groq"):
    # If does not exist, create index
    pc.create_index(
        name="groq",  # Replace with your preferred index name
        dimension=384,  # Use 1536 for OpenAI embeddings
        metric="cosine",  # Can also be "dotproduct" or "euclidean"
        spec = ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
    )

# Initialize index client
index = pc.Index(name="groq")

# ✅ Load Hugging Face Embedding Model (Use a lightweight model for efficiency)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dimension

def get_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings

def retrieve_relevant_documents(query):
    query_embedding = get_embeddings([query])[0]
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)  # Get top 3 most relevant docs
    return result['matches']

client = groq.Groq(api_key="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")

def query_groq_model(user_query, context):
    messages = [
        {"role": "system", "content": context},  # Context as system instruction
        {"role": "user", "content": user_query}  # User's actual question
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    answer = response.choices[0].message.content
    return answer


def get_answer(query):
    relevant_docs = retrieve_relevant_documents(query)
    context = "\n".join([doc['metadata']['text'] for doc in relevant_docs])
    query = f"""Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information. 
- Answer this query now: {query}"""
    response = query_groq_model(query, context)
    return response


def send_whatsapp_message(to, message):
    client = Client("your-twilio-sid", "your-twilio-auth-token")
    message = client.messages.create(
        body=message,
        from_="whatsapp:+14155238886",
        to=f"whatsapp:{to}"
    )
    return message.sid


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract text as plain text
    return text


def clean_text(text):
    # 1. Remove URLs, Emails, and Phone Numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'\b\d{10,}\b', '', text)  # Remove phone numbers (10-digit)

    # 2. Remove non-alphanumeric characters (except spaces and some punctuations)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\?\!\-\_]', '', text)  # Remove unwanted characters

    # 3. Normalize whitespace (remove extra spaces, newlines, etc.)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing spaces

    # 5. Normalize to lowercase
    text = text.lower()

    # 6. Remove non-English content (optional)
    try:
        if langdetect.detect(text) != 'en':  # If the text is not in English, discard it
            text = ""
    except langdetect.lang_detect_exception.LangDetectException:
        text = ""  # If language detection fails, discard the text

    return text


def chunk_with_overlap(text, chunk_size, overlap_size):
    text_length = len(text)
    chunks = []

    for i in range(0, text_length, chunk_size - overlap_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


def index_pdfs_to_pinecone(pdf_folder_path):
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Indexing PDFs"):
        pdf_path = os.path.join(pdf_folder_path, pdf_file)

        text = extract_text_from_pdf(pdf_path)
        ct = clean_text(text)
        chunk_size = 7500
        overlap_size = 1000
        text_chunks = chunk_with_overlap(ct, chunk_size, overlap_size)

        embeddings = get_embeddings(text_chunks)

        ids = [f"{pdf_file}_{i}" for i in range(len(text_chunks))]

        metadata = [{'text': chunk} for chunk in text_chunks]
        index.upsert(vectors=zip(ids, embeddings, metadata))

pdf_folder_path = "./data"

# Index PDFs into Pinecone
# index_pdfs_to_pinecone(pdf_folder_path)

# print(retrieve_relevant_documents("how much is the Payment of compensation in case of loss of human life in Tamil Nadu?"))
# print(get_answer("how much is the Payment of compensation in case of loss of human life in Tamil Nadu?"))

# print(retrieve_relevant_documents("how much is the Payment of compensation in case of loss of human life in Andhra Pradesh?"))
print(get_answer("how much is the Payment of compensation in case of loss of human life in Andhra Pradesh?"))

# print(extract_text_from_pdf("./data/MH.pdf"))
# print(clean_text(extract_text_from_pdf("./data/MH.pdf")))
