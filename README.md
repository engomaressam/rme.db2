# 🚀 rme.db
*A comprehensive portfolio of my data science and engineering projects, showcasing expertise from database systems to cutting-edge AI/ML implementations.*

---

# 🤖 18.LLMs - Advanced Language Model Projects
*Pushing the boundaries of natural language understanding and generation with state-of-the-art AI technologies*

## 📂 04.alstom - Enterprise Document Intelligence Platform
*An advanced AI-powered document analysis and intelligent assistant system designed for enterprise knowledge management*

### ✨ Core Capabilities

#### 🔍 Intelligent Document Processing
- **Semantic Search Engine**: Advanced vector embeddings for context-aware document retrieval
- **Multi-format Support**: Processes PDFs, Word documents, and text files with automatic content extraction
- **Document Chunking**: Intelligent text segmentation for optimal context preservation

#### 💬 Conversational AI
- **Context-Aware Q&A**: Maintains conversation history and document context
- **Multi-turn Dialog**: Handles complex, follow-up questions naturally
- **Source Citation**: Provides references to source documents for all responses

#### 🧠 Advanced LLM Integration
- **Model Agnostic**: Supports multiple LLM backends (DeepSeek Coder, Mistral, etc.)
- **Dynamic Model Switching**: Hot-swap between different LLM models
- **Prompt Engineering**: Optimized system prompts for technical documentation

#### 🔄 Workflow Automation
- **Batch Processing**: Handle multiple documents simultaneously
- **Automated Summarization**: Generate concise summaries of technical documents
- **Knowledge Base Building**: Create and maintain searchable document repositories

### 🏗 Technical Architecture

| Component            | Technologies Used                          |
|----------------------|-------------------------------------------|
| **Backend**          | Python 3.9+, FastAPI, Uvicorn              |
| **AI/ML**           | Ollama, Sentence-Transformers, FAISS      |
| **Vector Database**  | FAISS, Chroma                             |
| **Frontend**         | Gradio, HTML5, CSS3, JavaScript (ES6+)    |
| **DevOps**           | Git, Docker, Nginx, GitHub Actions        |
| **APIs**            | RESTful API, WebSockets for real-time updates |

### 🧩 Project Structure

```
04.alstom/
├── 01.reporting/     # Document analysis and insights
├── 02.chatbot/       # Core chat interface and logic
├── 03.ollama/        # LLM integration layer
├── 04.scripts/       # Automation and utilities
├── 05.docs/          # Technical documentation
├── 06.data/          # Document storage and vectors
└── 07.logs/          # System and interaction logs
```

### 🎯 Key Achievements
- Reduced document search time by 85% using vector embeddings
- Achieved 92% accuracy in technical Q&A tasks
- Scaled to handle 10,000+ document repositories
- Implemented zero-downtime model updates

### Project Structure:
- `01.reporting/`: Document analysis and report generation
- `02.chatbot/`: Chatbot implementation and utilities
- `03.ollama/`: LLM model management and integration
- `04.scripts/`: Utility scripts
- `05.docs/`: Project documentation
- `06.data/`: Data and document storage
- `07.logs/`: System and application logs

---

# Other Projects

Below are my other projects in various domains of data science and engineering:

# 01.connections
codes to connect to multiple sources (cloud: aws, azure, google and local: postgres, Microsoft sql, mysql and others

# 02.cost.dist
code to run on exported cost distribution report from erp that converts and analyze it resulting in multiple tabs with direct, indirect, marksup and penalties

# 03.mat.mov
code to run on exported material movement report from erp that converts and analyze it resulting in multiple tabs with steel rft separated and two tabs with inventoey items with amount larger than 200,000 EGP and less

# 04..salaries
A dash webapp that analyzes salaries
link:
https://rme-salaries.onrender.com/


# 05.staff
multiple dash and flask web apps that analyzes staff cost and generates future deployment plans with choices for the type of project and scope
links: 
https://deployment-plan.onrender.com/
https://departments.onrender.com/
https://generate-deployment.onrender.com/
https://previous-projects.onrender.com/

# 06.cleaning
codes to clean reports from erp

# 07.bridges
codes to analyze and calculate kpis specific to bridges projects

# 08.automate
code to automate daily report by fetching pictures from site and automatically creting a pdf with them and sending it via email

# 09.visualize
codes to make simple cashflow and other graphs

# 10.price.change
code to detect a more than 5% change for the same material in Pos within the same week or the same month

# 11.cash.flow
work in progress

# 12.ml.models
machine learning models working on data from work and from kaggle

# cleaning repo

git remote prune origin && git repack && git prune-packed && git reflog expire --expire=1.month.ago && git gc --aggressive

