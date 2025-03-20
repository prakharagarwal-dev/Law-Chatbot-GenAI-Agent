import os
import time
import uuid

import fitz
import groq
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import re
import langdetect

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather

app = Flask(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Required to avoid tokenization warnings

# Pinecone Initialization
pc = Pinecone(api_key="PINECONE_API_KEY")

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

if not pc.has_index(name="memory"):
    # If does not exist, create index
    pc.create_index(
        name="memory",  # Replace with your preferred index name
        dimension=384,  # Use 1536 for OpenAI embeddings
        metric="cosine",  # Can also be "dotproduct" or "euclidean"
        spec = ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
    )

# Initialize index client
index = pc.Index(name="groq")
mem_index = pc.Index(name="memory")

# ✅ Load Hugging Face Embedding Model (Use a lightweight model for efficiency)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dimension

def get_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings

def retrieve_relevant_documents(query):
    query_embedding = get_embeddings([query])[0]
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)  # Get top 3 most relevant docs
    return result['matches']

client = groq.Groq(api_key="GROQ_API_KEY")

def query_groq_model(user_query, context):
    messages = [
        {"role": "system", "content": context},  # Context as system instruction
        {"role": "user", "content": user_query}  # User's actual question
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return answer


def get_answer(query):
    past_memory, retrieved_docs = retrieve_context(query)

    system_context = f"""
    You are an AI assistant specializing in Indian wildlife and environmental laws, particularly in human-wildlife conflict resolution. Your goal is to provide accurate legal guidance based on the given context.

    ## Instructions:
    - **Be concise** and provide **direct, legally accurate answers**.
    - **If the context does not contain enough information, say 'I don't know'.** Do not make up answers.
    - **Use relevant laws** and references from the provided documents to support your response.
    - **Maintain continuity**: If the user follows up, consider past conversations to ensure a coherent response.
    - **Output must be clear, structured, and easy to understand**.

    ## Context from past conversations:
    {past_memory}

    ## Relevant Legal Documents:
    {retrieved_docs}
    """

    response = query_groq_model(query, system_context)  # Generate response from LLM

    store_memory_interaction(query, response)  # Store user query + response in memory

    return response

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
        chunk_size = 5000
        overlap_size = 1000
        text_chunks = chunk_with_overlap(ct, chunk_size, overlap_size)

        embeddings = get_embeddings(text_chunks)

        ids = [f"{pdf_file}_{i}" for i in range(len(text_chunks))]

        metadata = [{'text': chunk} for chunk in text_chunks]
        index.upsert(vectors=zip(ids, embeddings, metadata))

# Load Hugging Face Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_response(response):
    """Summarizes long chatbot responses to store in memory efficiently."""
    if len(response.split()) > 50:  # Only summarize if response is long
        summary = summarizer(response, max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    return response  # Return as is if already short

def store_memory_interaction(query, response):
    """Stores user query and summarized chatbot response in Pinecone memory."""
    summarized_response = summarize_response(response)

    # Generate unique ID for this memory entry
    memory_id = str(uuid.uuid4())

    # Get embedding for the query
    query_embedding = get_embeddings([query])[0]

    # Properly structure metadata
    metadata = {"query": query, "response": summarized_response}

    # Store in Pinecone
    mem_index.upsert(vectors=[(memory_id, query_embedding, metadata)])

# def store_memory_interaction(user_query, bot_response):
#     """
#     Store user query + bot response in Pinecone memory index.
#     """
#     embedding = embedding_model.encode(user_query).tolist()
#     mem_index.upsert(
#         vectors=[(f"chat_{user_query}", embedding, {"query": user_query, "response": bot_response})]
#     )

def retrieve_context(user_query, top_k=3):
    """
    Retrieve past memory + relevant RAG documents.
    """
    embedding = embedding_model.encode(user_query).tolist()

    # Retrieve past conversations
    memory_results = mem_index.query(vector=embedding, top_k=top_k, include_metadata=True)
    past_contexts = [
        f"User: {match['metadata']['query']}\nAI: {match['metadata']['response']}"
        for match in memory_results["matches"]
    ]

    # Retrieve RAG documents
    doc_results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    doc_contexts = [match["metadata"]["text"] for match in doc_results["matches"]]

    return "\n".join(past_contexts), "\n".join(doc_contexts)

pdf_folder_path = "./data"

from twilio.rest import Client

# Twilio credentials
# account_sid = "TWILIO_ACCOUNT_SID"
# auth_token = "TWILIO_AUTH_TOKEN"
#
# # Initialize Twilio client
# twilio_client = Client(account_sid, auth_token)
#
# # Make a call
# call = twilio_client.calls.create(
#     to="PHONE_NUMBER",  # Replace with the recipient's phone number
#     from_="+17069481798",  # Replace with your Twilio phone number
#     url="https://7c17-2001-420-5448-1302-95dc-8e19-3261-d029.ngrok-free.app/ivr"
# )
#
# print(f"Call SID: {call.sid}")

def split_text(text, max_length=1000):
    """Splits text into smaller chunks that fit within Twilio's limits."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def convert_markdown_to_whatsapp(text):
    """Converts standard Markdown to WhatsApp-supported formatting."""

    # Convert bold (**text**) → *text*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)

    # Convert italic (*text* or _text_) → _text_
    text = re.sub(r'\*(.*?)\*', r'_\1_', text)

    # Convert headings (# Heading) → *Heading*
    text = re.sub(r'#+\s*(.*)', r'*\1*', text)

    # Convert unordered lists (- item) → • item
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)

    # Remove inline code blocks (```code```) → (not supported in WhatsApp)
    text = re.sub(r'```(.*?)```', r'\1', text, flags=re.DOTALL)

    # Remove single-line inline code (`code`) → (not supported in WhatsApp)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    return text

from langdetect import detect
from deep_translator import GoogleTranslator

# Function to detect the language
def detect_language(text):
    try:
        return detect(text)  # Returns language code (e.g., 'hi' for Hindi, 'ta' for Tamil)
    except:
        return "en"  # Default to English if detection fails

# Function to translate text
def translate_text(text, src_lang, target_lang):
    if src_lang == target_lang:
        return text  # No translation needed
    return GoogleTranslator(source=src_lang, target=target_lang).translate(text)

from twilio.twiml.messaging_response import MessagingResponse

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()

    detected_lang = detect_language(incoming_msg)
    print("Detected Language:", detected_lang)

    # If not English, translate to English
    if detected_lang != "en":
        translated_query = translate_text(incoming_msg, detected_lang, "en")
    else:
        translated_query = incoming_msg

    # Send message to chatbot and get a response
    chatbot_response = get_answer(translated_query)
    print("Chatbot response: ", chatbot_response)

    # Translate back to the user's language
    if detected_lang != "en":
        chatbot_response = translate_text(chatbot_response, "en", detected_lang)

    # Convert Markdown to WhatsApp format
    formatted_response = convert_markdown_to_whatsapp(chatbot_response)

    # Split long responses
    response_chunks = split_text(formatted_response)
    # Create Twilio response
    resp = MessagingResponse()
    for chunk in response_chunks:
        resp.message(chunk)
        time.sleep(1)

    return str(resp)


@app.route("/ivr", methods=["POST"])
def ivr():
    """Handles incoming Twilio calls and processes user input."""
    response = VoiceResponse()

    # Step 1: Ask user a question
    gather = Gather(input="speech", action="/process_query", timeout=5)
    gather.say("Welcome to the Wildlife Legal Assistant. Please ask your question.")
    response.append(gather)

    return Response(str(response), mimetype="text/xml")


@app.route("/process_query", methods=["POST"])
def process_query():
    """Processes user input and responds with chatbot output."""
    response = VoiceResponse()

    # Capture user query
    user_query = request.form.get("SpeechResult")
    print("User Query:", user_query)

    if user_query:
        # Detect if user wants to exit
        exit_phrases = ["no", "goodbye", "exit", "that's all", "stop", "end", "thank you"]
        if user_query.lower().strip() in exit_phrases:
            response.say("Thank you for using the Wildlife Legal Assistant. Goodbye!")
            response.hangup()
            return Response(str(response), mimetype="text/xml")

        # Get chatbot response
        chatbot_response = get_answer(user_query)

        # Split long response into chunks
        response_chunks = split_text(chatbot_response, max_length=1000)

        # Speak each chunk
        for chunk in response_chunks:
            response.say(chunk)

        # Ask if user has another question
        gather = Gather(input="speech", action="/process_query", timeout=10)
        gather.say("Do you have another question? Please ask now.")
        response.append(gather)

    else:
        response.say("I didn't catch that. Please try again.")

    return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    app.run(port=8000)

# Index PDFs into Pinecone
# index_pdfs_to_pinecone(pdf_folder_path)

# print(retrieve_relevant_documents("how much is the Payment of compensation in case of loss of human life in Tamil Nadu?"))
# print(get_answer("how much is the Payment of compensation in case of loss of human life in Tamil Nadu?"))

# print(retrieve_relevant_documents("how much is the Payment of compensation in case of loss of human life in Andhra Pradesh?"))
# print(get_answer("how much is the Payment of compensation in case of loss of human life in Andhra Pradesh?"))

# print(retrieve_relevant_documents("how much is the Payment of compensation in case of loss of human life in Andhra Pradesh?"))
# print(get_answer("I am a farmer in kerala, how much compensation can I get for the loss of my crops?"))

# while True:
#     query = input("Enter your query (or type 'exit' to quit): ")
#     if query.lower() == 'exit':
#         break
#     answer = get_answer(query)
#     print("Answer:", answer)


# print(extract_text_from_pdf("./data/MH.pdf"))
# print(clean_text(extract_text_from_pdf("./data/MH.pdf")))
