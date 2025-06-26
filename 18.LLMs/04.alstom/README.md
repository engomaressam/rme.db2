# Alstom Project Assistant

## Overview
The Alstom Project Assistant is a chatbot designed to help users access and query project documents. It uses a vector-based approach to find relevant documents based on user questions and provides answers using the Ollama API.

## Network Update (May 2025)
The application has been updated to use network paths for document access. This allows users from different computers to access the same documents through the shared network drive.

### Network Path
Documents are now stored at:
```
\\fileserver2\Head Office Server\Projects Control (PC)\10 Backup\05 Models\alstom
```

## Components

### 1. Document Processor (14.document_processor.py)
- Extracts text from PDF documents using PyPDF2 with Tesseract OCR fallback
- Creates document embeddings using SentenceTransformer
- Stores the vector database in a local directory

### 2. Web Interface (15.vector_chatbot.py)
- Provides a user interface with a left sidebar (1/4) and main chat area (3/4)
- Connects to the Ollama API running on a GPU machine
- Loads pre-computed document embeddings for efficient retrieval
- Shows GPU connection status in the sidebar

### 3. Public Interface (16.chatbot.public.py)
- Simplified version for public access
- Displays document information without direct file access

## Development Versions
Multiple versions of each component have been created to allow for development without affecting the production version:
- 14.document_processor.rev.03.py
- 15.vector_chatbot.rev.03.py
- 16.chatbot.public.rev.02.py

## Running the Application
1. Process documents with the document processor
2. Start the web interface on port 7860 (production) or 7862 (development)
3. Connect to the interface via web browser

## Technical Details
- Uses Gradio for the web interface
- Connects to Ollama API at http://10.10.12.202:11434
- Uses SentenceTransformer for document embeddings
- Extracts text from PDFs using PyPDF2 and Tesseract OCR
