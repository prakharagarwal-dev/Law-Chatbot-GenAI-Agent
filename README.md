# Wildlife Legal Assistant

This project is an AI-powered assistant specializing in Indian wildlife and environmental laws, particularly in human-wildlife conflict resolution. It provides accurate legal guidance based on the given context.

## Features

- **Query Handling**: Prompts users to write queries and prints the answers by calling the `get_answer` function.
- **Pinecone Integration**: Uses Pinecone for vector storage and retrieval.
- **Hugging Face Integration**: Utilizes Hugging Face models for embeddings and summarization.
- **Flask Web Server**: Provides endpoints for WhatsApp and IVR interactions using Twilio.
- **PDF Processing**: Extracts and cleans text from PDFs, then indexes them into Pinecone.

## Setup

1. **Create a virtual environment**:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Set environment variables**:
    ```sh
    export TOKENIZERS_PARALLELISM=true
    ```

4. **Configure Pinecone**:
    - Set your Pinecone API key in the `main.py` file.

5. **Configure Twilio**:
    - Set your Twilio account SID and auth token in the `main.py` file.

## Usage

1. **Run the Flask server**:
    ```sh
    python main.py
    ```

2. **Interact with the assistant**:
    - Use the `/whatsapp` endpoint for WhatsApp interactions.
    - Use the `/ivr` endpoint for IVR interactions.

## Functions

- **get_answer(query)**: Retrieves the answer to a given query.
- **extract_text_from_pdf(pdf_path)**: Extracts text from a PDF file.
- **clean_text(text)**: Cleans the extracted text.
- **chunk_with_overlap(text, chunk_size, overlap_size)**: Splits text into chunks with overlap.
- **index_pdfs_to_pinecone(pdf_folder_path)**: Indexes PDFs into Pinecone.
- **summarize_response(response)**: Summarizes long chatbot responses.
- **store_memory_interaction(query, response)**: Stores user query and summarized chatbot response in Pinecone memory.
- **retrieve_context(user_query, top_k=3)**: Retrieves past memory and relevant documents.

## License

This project is licensed under the MIT License.