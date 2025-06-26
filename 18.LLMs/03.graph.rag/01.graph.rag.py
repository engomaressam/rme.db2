import os
import re
import json
import networkx as nx
import pytesseract
from typing import List, Dict, Any
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO
from dataclasses import dataclass

# Langchain imports
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.graphs import NetworkxEntityGraph
from langchain.callbacks.manager import CallbackManager

@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str
    object_: str  # Note the underscore to match NetworkxEntityGraph's expectation

class OCRPDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path
        # Set Tesseract path - update this to your Tesseract installation path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def load(self):
        # Convert PDF to images
        try:
            # Set poppler path - update this to your Poppler installation path
            images = convert_from_path(self.file_path, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            return []
            
        documents = []
        for i, image in enumerate(images):
            try:
                # Perform OCR with support for multiple languages (English and Arabic)
                text = pytesseract.image_to_string(image, lang='eng+ara')
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": self.file_path, "page": i + 1}
                    )
                    documents.append(doc)
                    print(f"  Page {i+1}: {len(text)} characters")
                else:
                    print(f"  Page {i+1}: No text extracted")
            except Exception as e:
                print(f"Error processing page {i+1}: {str(e)}")
                
        return documents

class GraphRAG:
    def __init__(self, pdf_directory, model_name="mistral"):
        """Initialize the GraphRAG system"""
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        
        # Initialize LLM with timeout
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([]),
            temperature=0,
            timeout=30  # 30 second timeout
        )
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            self.embeddings = None
        
        # Initialize graph
        self.graph = NetworkxEntityGraph()
        
        # Initialize vector store
        self.vector_store = None
        self.po_data = {}
        
        print("Initialized GraphRAG system successfully")
        
    def extract_po_data(self, text):
        """Extract key business entities from text with error handling and timeout"""
        default_data = {
            'project_po': {
                'po_number': 'unknown',
                'project_name': 'unknown',
                'issue_date': 'unknown',
                'status': 'unknown'
            },
            'supplier': {
                'name': 'unknown',
                'id': 'unknown',
                'contact': 'unknown'
            },
            'items': [],
            'terms': {
                'delivery': 'unknown',
                'warranty': 'unknown',
                'other': 'unknown'
            },
            'payment': {
                'terms': 'unknown',
                'total_amount': 'unknown',
                'currency': 'unknown',
                'schedule': 'unknown'
            }
        }
        
        try:
            # Use regex to extract PO number first (faster than LLM)
            po_match = re.search(r'P[O]?[-.\s]?\d{2,}[-.]?\d{2,}', text)
            if po_match:
                default_data['project_po']['po_number'] = po_match.group().strip()
            
            # Try to extract dates (faster than LLM)
            date_matches = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)
            if date_matches:
                default_data['project_po']['issue_date'] = date_matches[0]
            
            # Use LLM with a shorter prompt for better reliability
            prompt = """Extract from the text (return 'unknown' if not found):
            1. Project name
            2. Supplier name and ID
            3. Main items (max 3)
            4. Payment terms
            5. Total amount
            Format as Python dict.
            
            Text: {text}
            """
            
            try:
                response = self.llm.predict(prompt, timeout=10)  # 10 second timeout
                extracted = eval(response)
                
                # Update default data with extracted information
                if isinstance(extracted, dict):
                    if 'project_name' in extracted:
                        default_data['project_po']['project_name'] = extracted['project_name']
                    if 'supplier' in extracted:
                        default_data['supplier'].update(extracted['supplier'])
                    if 'items' in extracted:
                        default_data['items'] = extracted['items'][:3]  # Limit to 3 items
                    if 'payment_terms' in extracted:
                        default_data['payment']['terms'] = extracted['payment_terms']
                    if 'total_amount' in extracted:
                        default_data['payment']['total_amount'] = extracted['total_amount']
            
            except Exception as llm_error:
                print(f"LLM extraction failed: {str(llm_error)}. Using regex extracted data.")
            
            return default_data
            
        except Exception as e:
            print(f"Error in data extraction: {str(e)}")
            return default_data
            
    def load_pdfs(self):
        """Load PDF files from the specified directory"""
        if not os.path.exists(self.pdf_directory):
            raise ValueError(f"PDF directory not found: {self.pdf_directory}")
            
        documents = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
            
        for file in pdf_files:
            try:
                file_path = os.path.join(self.pdf_directory, file)
                loader = OCRPDFLoader(file_path)
                docs = loader.load()
                
                if docs:
                    print(f"\nLoaded {file} with {len(docs)} pages")
                    # Debug: Check content of each page
                    for i, doc in enumerate(docs):
                        content = doc.page_content.strip()
                        if content:
                            print(f"  Page {i+1}: {len(content)} characters")
                            documents.append(doc)
                        else:
                            print(f"  Page {i+1}: Empty content")
                            
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
        if not documents:
            raise ValueError("No content could be loaded from PDF files")
            
        print(f"\nSuccessfully loaded {len(documents)} pages with content")
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        if not documents:
            raise ValueError("No documents to split")
            
        try:
            # Debug: Check content before splitting
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"\nTotal characters before splitting: {total_chars}")
            
            # Clean and normalize text
            for doc in documents:
                # Remove left-to-right and right-to-left marks
                doc.page_content = doc.page_content.replace('\u200e', '').replace('\u200f', '')
                # Normalize Arabic text
                doc.page_content = doc.page_content.strip()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                is_separator_regex=False
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Debug: Check content after splitting
            if chunks:
                print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks")
                try:
                    preview = chunks[0].page_content[:100].encode('utf-8', errors='ignore').decode('utf-8')
                    print(f"First chunk preview: {preview}...")
                except Exception as e:
                    print(f"Could not display preview: {str(e)}")
            else:
                raise ValueError("Splitting produced no chunks")
                
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            raise
    
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
            
        try:
            texts = [doc.page_content for doc in documents]
            if not any(texts):
                raise ValueError("Documents contain no text content")
                
            print(f"Creating vector store from {len(documents)} documents")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise
        
    def extract_entities_and_relationships(self, text, po_data=None):
        """Extract entities and relationships with business context"""
        context = ""
        if po_data:
            context = f"\nThis is from PO number {po_data.get('po_number', 'unknown')} "
            context += f"issued to {po_data.get('supplier', 'unknown')} "
            context += f"on {po_data.get('date', 'unknown')}"
        
        prompt = f"""Extract key business entities and their relationships from this purchase order text.
        Consider these relationship types:
        - supplies_to (between supplier and buyer)
        - ordered_by (between items and company)
        - has_value (between PO and amount)
        - dated_on (between PO and date)
        - includes (between PO and items)
        
        Format: Entity1 | Relationship | Entity2
        Text: {text}{context}
        
        Entities and Relationships:"""
        
        response = self.llm.predict(prompt)
        relationships = []
        
        for line in response.split('\n'):
            if '|' in line:
                try:
                    entity1, relation, entity2 = [x.strip() for x in line.split('|')]
                    relationships.append((entity1, relation, entity2))
                except:
                    continue
                
        return relationships
    
    def build_knowledge_graph(self, documents):
        """Build knowledge graph from documents focusing on key business entities"""
        for doc in documents:
            # Extract PO data
            data = self.extract_po_data(doc.page_content)
            if not data:
                continue
                
            # 1. Project/PO Information
            po_info = data['project_po']
            po_number = po_info['po_number']
            if po_number == 'unknown':
                continue
                
            # Create PO node and link to project
            if po_info['project_name'] != 'unknown':
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='belongs_to_project',
                    object_=po_info['project_name']
                ))
            
            # Add PO status and date
            if po_info['status'] != 'unknown':
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='has_status',
                    object_=po_info['status']
                ))
            
            if po_info['issue_date'] != 'unknown':
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='issued_on',
                    object_=po_info['issue_date']
                ))
            
            # 2. Supplier Information
            supplier = data['supplier']
            if supplier['name'] != 'unknown':
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='issued_to_supplier',
                    object_=supplier['name']
                ))
                
                # Add supplier details
                if supplier['id'] != 'unknown':
                    self.graph.add_triple(KnowledgeTriple(
                        subject=supplier['name'],
                        predicate='has_id',
                        object_=supplier['id']
                    ))
                
                if supplier['contact'] != 'unknown':
                    self.graph.add_triple(KnowledgeTriple(
                        subject=supplier['name'],
                        predicate='has_contact',
                        object_=supplier['contact']
                    ))
            
            # 3. Items Information
            for item in data['items']:
                if 'item_code' in item and item['item_code'] != 'unknown':
                    item_code = item['item_code']
                    
                    # Link item to PO
                    self.graph.add_triple(KnowledgeTriple(
                        subject=po_number,
                        predicate='includes_item',
                        object_=item_code
                    ))
                    
                    # Add item details
                    for key, value in item.items():
                        if key != 'item_code' and value != 'unknown':
                            self.graph.add_triple(KnowledgeTriple(
                                subject=item_code,
                                predicate=f'has_{key}',
                                object_=str(value)
                            ))
            
            # 4. Terms and Conditions
            terms = data['terms']
            for term_type, value in terms.items():
                if value != 'unknown':
                    self.graph.add_triple(KnowledgeTriple(
                        subject=po_number,
                        predicate=f'has_{term_type}_terms',
                        object_=value
                    ))
            
            # 5. Payment Information
            payment = data['payment']
            for pay_type, value in payment.items():
                if value != 'unknown':
                    self.graph.add_triple(KnowledgeTriple(
                        subject=po_number,
                        predicate=f'has_{pay_type}',
                        object_=str(value)
                    ))
    
    def query(self, question, use_graph=True, k=3):
        """Query the system using both vector store and knowledge graph"""
        try:
            # Get relevant documents from vector store
            docs = self.vector_store.similarity_search(question, k=k)
            
            # Get relevant subgraph if using graph
            if use_graph:
                # Extract relevant nodes and edges from the graph
                graph_context = []
                for node in self.graph._graph.nodes():
                    # Add node information
                    edges = list(self.graph._graph.edges(node, data=True))
                    if edges:
                        graph_context.append(f"Node: {node}")
                        for _, target, data in edges:
                            graph_context.append(f"  -> {data.get('predicate', 'related_to')} -> {target}")
                
                graph_context = "\n".join(graph_context)
            else:
                graph_context = ""
            
            # Combine document and graph context
            context = "\n\n".join([doc.page_content for doc in docs])
            if graph_context:
                context += "\n\nGraph Context:\n" + graph_context
            
            # Create prompt
            prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say 'I don't have enough information to answer that.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            
            # Get response from LLM
            response = self.llm.predict(prompt)
            return response
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "Sorry, I encountered an error while processing your query."
    
    def save_graph(self, output_dir):
        """Save the knowledge graph to disk"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the graph in GEXF format for visualization
            gexf_path = os.path.join(output_dir, 'knowledge_graph.gexf')
            nx.write_gexf(self.graph._graph, gexf_path)
            print(f"\nSaved graph to {gexf_path}")
            
            # Save graph data as JSON for easier loading
            graph_data = {
                'nodes': list(self.graph._graph.nodes()),
                'edges': [(u, v, d) for u, v, d in self.graph._graph.edges(data=True)]
            }
            json_path = os.path.join(output_dir, 'knowledge_graph.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            print(f"Saved graph data to {json_path}")
            
        except Exception as e:
            print(f"Error saving graph: {str(e)}")
            raise
    
    def process_documents(self):
        """Main method to process documents and build both vector store and knowledge graph"""
        try:
            # Load PDFs
            documents = self.load_pdfs()
            print(f"\nTotal documents loaded: {len(documents)}")
            
            # Split into chunks
            print("\nSplitting documents into chunks...")
            chunks = self.split_documents(documents)
            
            # Create vector store
            print("\nCreating vector store...")
            self.create_vector_store(chunks)
            
            # Build knowledge graph
            print("\nBuilding knowledge graph...")
            self.build_knowledge_graph(chunks)
            
            # Save the graph
            output_dir = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\05.llm\graph.rag"
            self.save_graph(output_dir)
            
            print("\nDocument processing complete!")
            
        except Exception as e:
            print(f"\nError in document processing: {str(e)}")
            raise
        
def main():
    try:
        # Initialize GraphRAG
        print("Initializing GraphRAG...")
        rag = GraphRAG(model_name="mistral", pdf_directory="D:\\OEssam\\01.pdfs")
        
        # Process documents
        print("\nProcessing documents...")
        rag.process_documents()
        
        # Example queries
        print("\nRunning example queries...")
        questions = [
            "List all suppliers and their total order values",
            "What are the most commonly ordered items?",
            "What is the total value of all purchase orders?",
            "Show me the details of the latest purchase order"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            try:
                answer = rag.query(question)
                print(f"A: {answer}")
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()