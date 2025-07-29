# RAG Q&A: Conversational PDF Assistant

![RAG Q&A](https://img.shields.io/badge/RAG-Q%26A-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)

A powerful conversational question-answering application that allows users to upload PDF documents and chat with their content using Retrieval-Augmented Generation (RAG).

## Features

- 📄 **PDF Upload**: Support for single or multiple PDF documents
- 💬 **Conversational Memory**: Maintains chat history for contextual conversations
- 🔍 **Smart Retrieval**: Uses history-aware retrieval to understand context from previous exchanges
- 🧠 **RAG Architecture**: Combines retrieval with generative AI for accurate answers
- 🔄 **Session Management**: Supports multiple chat sessions with unique identifiers

## Tech Stack

- **Frontend**: Streamlit
- **Embedding Model**: Google Generative AI Embeddings (Gemini)
- **LLM**: Perplexity AI (Sonar-Pro model)
- **Vector Store**: Chroma
- **Framework**: LangChain

## Setup Instructions

### Prerequisites

- Python 3.8+
- Required API keys:
  - Perplexity AI API key
  - Google API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-qa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   PERPLEXITY_API_KEY=your_perplexity_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser (typically at http://localhost:8501)

3. Enter a session ID (or use the default)

4. Upload one or more PDF documents

5. Start asking questions about the content of your documents

## How It Works

1. **Document Processing**: PDFs are loaded and split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to embeddings using Google's Gemini model
3. **Vector Storage**: Embeddings are stored in a Chroma vector database
4. **Query Processing**: 
   - User questions are processed considering chat history
   - Relevant document chunks are retrieved
   - The LLM generates concise answers based on retrieved content

## Example Interactions

- "What is the main topic of this document?"
- "Can you summarize the third section?"
- "What did the author say about [specific topic]?"
- "Based on what we discussed earlier, how does [concept] relate to [another concept]?"

## License

[MIT License](LICENSE) 