# GraphRAG Hybrid System for Purchase Order Analysis

This system combines graph databases (Neo4j) and vector databases (Qdrant) to create a powerful hybrid approach for analyzing purchase orders.

## Features

- **PDF Processing**: Extract purchase order data from PDF files using OCR
- **Graph Database**: Store relationships between purchase orders, projects, suppliers, and items
- **Vector Database**: Enable semantic search for similar purchase orders
- **Visualization**: Generate graph visualizations of purchase order relationships
- **Hybrid Search**: Combine graph traversal and semantic search for powerful queries

## Architecture

The system consists of the following components:

1. **Database Managers**:
   - `Neo4jManager`: Handles graph database operations
   - `QdrantManager`: Manages vector embeddings for semantic search

2. **Processors**:
   - `DocumentProcessor`: Extracts purchase order data from PDF files
   - `EmbeddingProcessor`: Generates vector embeddings for purchase orders and search queries

3. **Main GraphRAG class**:
   - Integrates all components and provides high-level functionality
   - Supports visualization-only mode for working with existing graph data

## Setup

### Prerequisites

- Python 3.8+
- Neo4j database (local or cloud)
- Qdrant database (local or cloud)
- Tesseract OCR
- Poppler (for PDF processing)

### Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up Neo4j and Qdrant databases:

Option 1: Using Docker (recommended):
```bash
docker-compose up -d
```

Option 2: Using Neo4j Desktop and Qdrant Cloud:
- Install [Neo4j Desktop](https://neo4j.com/download/)
- Create a free account on [Qdrant Cloud](https://cloud.qdrant.io/)

3. Configure environment variables in `config.py`

## Usage

### Process Purchase Orders

```bash
python scripts/process_pos.py --directory /path/to/pdfs --recursive
```

### Visualize Graph

```bash
python scripts/visualize_graph.py --graph-file /path/to/graph.json
```

### Search Purchase Orders

```bash
python scripts/search_pos.py --query "purchase orders for project X from supplier Y" --limit 5
```

## Running in Different Modes

The system can run in different modes:

- **Full Mode**: Uses both Neo4j and Qdrant for hybrid search and graph analysis
- **Visualization-Only Mode**: Works with existing graph data without requiring database connections
- **With/Without LLM**: Can operate with or without the Ollama LLM component

## Directory Structure

```
.
├── config.py                # Configuration settings
├── docker-compose.yml       # Docker setup for Neo4j and Qdrant
├── src/                     # Source code
│   ├── graphrag.py          # Main GraphRAG class
│   ├── database/            # Database managers
│   │   ├── neo4j_manager.py # Neo4j database manager
│   │   └── qdrant_manager.py# Qdrant database manager
│   └── processors/          # Data processors
│       ├── document_processor.py # PDF processing
│       └── embedding_processor.py # Embedding generation
├── scripts/                 # Utility scripts
│   ├── process_pos.py       # Process purchase orders
│   ├── visualize_graph.py   # Visualize graph
│   └── search_pos.py        # Search purchase orders
└── data/                    # Data directory
    └── output/              # Output files (graphs, visualizations)
```
