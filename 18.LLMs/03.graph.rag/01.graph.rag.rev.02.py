import os
import os
import json
import networkx as nx
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO
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

# LangChain imports
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain_community.graphs import NetworkxEntityGraph
import matplotlib.pyplot as plt

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
    def __init__(self, model_name="mistral", pdf_directory=None, skip_llm=False):
        self.model_name = model_name
        self.pdf_directory = pdf_directory or "D:\\OEssam\\01.pdfs"
        self.skip_llm = skip_llm
        
        try:
            # Use a multilingual model for embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize LLM with timeout (if not skipped)
            if not skip_llm:
                try:
                    self.llm = Ollama(
                        model=model_name,
                        temperature=0.0,
                        stop=["\n\n"],
                        timeout=30.0  # 30 second timeout
                    )
                    print("Successfully initialized Ollama LLM")
                except Exception as e:
                    print(f"Warning: Could not initialize Ollama LLM: {str(e)}")
                    print("Some functionality will be limited.")
                    self.llm = None
            else:
                self.llm = None
                print("Skipping LLM initialization as requested")
            
            self.graph = NetworkxEntityGraph()
            self.vector_store = None
            self.po_data = {}
            
            print("Successfully initialized GraphRAG")
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise
        
    def extract_po_details(self, text):
        """Extract purchase order details using structured prompts with project focus"""
        prompt = f"""Extract the following information from this purchase order text. Return as JSON with these keys:
        {{
            "po_number": "The purchase order number",
            "project": {{
                "name": "The project name",
                "code": "The project code if available"
            }},
            "date": "The PO date",
            "supplier": {{
                "name": "The supplier/company name",
                "id": "The supplier ID if available",
                "category": "The supplier category/type if mentioned"
            }},
            "items": [
                {{
                    "name": "The item name/description",
                    "quantity": "The quantity ordered",
                    "unit": "The unit of measurement",
                    "value": "The total value for this item"
                }}
            ],
            "total_value": "The total PO value",
            "payment_terms": "Any payment terms mentioned",
            "delivery_terms": "Any delivery terms or conditions"
        }}

        Return only the JSON, no other text.

        Text: {text}
        """
        
        try:
            response = self.llm.predict(prompt, timeout=10)
            
            # Parse response as JSON
            try:
                po_data = json.loads(response)
                return po_data
            except json.JSONDecodeError:
                print("Error parsing LLM response as JSON")
                return {
                    'po_number': None,
                    'project': {'name': None, 'code': None},
                    'date': None,
                    'supplier': {'name': None, 'id': None, 'category': None},
                    'total_value': None,
                    'payment_terms': None,
                    'delivery_terms': None,
                    'items': []
                }
                
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return {
                'po_number': None,
                'project': {'name': None, 'code': None},
                'date': None,
                'supplier': {'name': None, 'id': None, 'category': None},
                'total_value': None,
                'payment_terms': None,
                'delivery_terms': None,
                'items': []
            }
        
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
        """Build hierarchical knowledge graph focused on projects and suppliers"""
        # Track unique entities to avoid duplication
        projects = {}
        suppliers = {}
        items = {}
        
        for doc in documents:
            # Extract structured PO data
            po_data = self.extract_po_details(doc.page_content)
            if not po_data or not po_data.get('po_number'):
                continue
                
            po_number = po_data['po_number']
            self.po_data[po_number] = po_data
            
            # 1. Handle Project
            project = po_data.get('project', {})
            if project.get('name'):
                project_name = project['name']
                projects[project_name] = project
                
                # Link PO to Project
                self.graph.add_triple(KnowledgeTriple(
                    subject=project_name,
                    predicate='has_po',
                    object_=po_number
                ))
                
                # Add project metadata
                if project.get('code'):
                    self.graph.add_triple(KnowledgeTriple(
                        subject=project_name,
                        predicate='has_code',
                        object_=project['code']
                    ))
            
            # 2. Handle Supplier
            supplier = po_data.get('supplier', {})
            if supplier.get('name'):
                supplier_name = supplier['name']
                suppliers[supplier_name] = supplier
                
                # Link PO to Supplier
                self.graph.add_triple(KnowledgeTriple(
                    subject=supplier_name,
                    predicate='issued_po',
                    object_=po_number
                ))
                
                # Add supplier metadata
                if supplier.get('id'):
                    self.graph.add_triple(KnowledgeTriple(
                        subject=supplier_name,
                        predicate='has_id',
                        object_=supplier['id']
                    ))
                if supplier.get('category'):
                    self.graph.add_triple(KnowledgeTriple(
                        subject=supplier_name,
                        predicate='has_category',
                        object_=supplier['category']
                    ))
            
            # 3. Add PO details
            if po_data.get('date'):
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='issued_on',
                    object_=po_data['date']
                ))
            
            if po_data.get('total_amount'):
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='has_value',
                    object_=str(po_data['total_amount'])
                ))
                
            if po_data.get('payment_terms'):
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='has_payment_terms',
                    object_=po_data['payment_terms']
                ))
            
            # 4. Handle Items (max 3 per PO to reduce nodes)
            for item in po_data.get('items', [])[:3]:
                if not item.get('code'):
                    continue
                    
                item_code = item['code']
                items[item_code] = item
                
                # Link item to PO
                self.graph.add_triple(KnowledgeTriple(
                    subject=po_number,
                    predicate='includes_item',
                    object_=item_code
                ))
                
                # Add item metadata
                if item.get('category'):
                    self.graph.add_triple(KnowledgeTriple(
                        subject=item_code,
                        predicate='has_category',
                        object_=item['category']
                    ))
                
                # Combine quantity and unit price into a single node
                if item.get('quantity') and item.get('unit_price'):
                    value_node = f"{item.get('quantity')} units at {item.get('unit_price')} each"
                    self.graph.add_triple(KnowledgeTriple(
                        subject=item_code,
                        predicate='has_value',
                        object_=value_node
                    ))
        
        # Print statistics
        print(f"\nGraph Statistics:")
        print(f"Projects: {len(projects)}")
        print(f"Suppliers: {len(suppliers)}")
        print(f"POs: {len(self.po_data)}")
        print(f"Items: {len(items)}")
        print(f"Total Nodes: {len(self.graph._graph.nodes())}")
    
    def query(self, question, use_graph=True, k=3):
        """Query the system focusing on business entities and relationships"""
        try:
            # Get relevant documents from vector store
            docs = self.vector_store.similarity_search(question, k=k)
            
            # Get graph context if requested
            graph_context = ""
            if use_graph:
                # Analyze question type
                question_lower = question.lower()
                
                if 'supplier' in question_lower:
                    # Get all suppliers and their relationships
                    suppliers = [n for n, d in self.graph._graph.nodes(data=True) 
                               if any(edge[2].get('predicate') == 'issued_po' 
                                   for edge in self.graph._graph.edges(n, data=True))]
                    
                    supplier_stats = []
                    for supplier in suppliers:
                        # Get POs for this supplier
                        pos = [edge[1] for edge in self.graph._graph.edges(supplier, data=True)
                              if edge[2].get('predicate') == 'issued_po']
                        
                        # Get projects for these POs
                        projects = set()
                        total_value = 0
                        total_items = 0
                        
                        for po in pos:
                            # Get project
                            project_edges = [edge[0] for edge in self.graph._graph.in_edges(po, data=True)
                                           if edge[2].get('predicate') == 'has_po']
                            projects.update(project_edges)
                            
                            # Get value
                            value_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                          if edge[2].get('predicate') == 'has_value']
                            if value_edges:
                                try:
                                    total_value += float(value_edges[0][2].get('object_', '0').split()[0])
                                except:
                                    pass
                            
                            # Get items
                            item_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                         if edge[2].get('predicate') == 'has_item']
                            total_items += len(item_edges)
                        
                        supplier_stats.append({
                            'name': supplier,
                            'pos': len(pos),
                            'projects': len(projects),
                            'total_value': total_value,
                            'total_items': total_items
                        })
                    
                    # Sort by total value and add relationships
                    supplier_stats.sort(key=lambda x: x['total_value'], reverse=True)
                    
                    # Format supplier information with relationships
                    graph_context += "\nSupplier Analysis:\n"
                    graph_context += "-" * 40 + "\n"
                    
                    for stat in supplier_stats:
                        # Basic stats
                        graph_context += f"\nSupplier: {stat['name']}\n"
                        graph_context += f"  - Active POs: {stat['pos']}\n"
                        graph_context += f"  - Projects Involved: {stat['projects']}\n"
                        graph_context += f"  - Total Value: {stat['total_value']:,.2f}\n"
                        graph_context += f"  - Total Items: {stat['total_items']}\n"
                        
                        # Calculate averages
                        avg_po_value = stat['total_value'] / stat['pos'] if stat['pos'] > 0 else 0
                        avg_items_per_po = stat['total_items'] / stat['pos'] if stat['pos'] > 0 else 0
                        
                        graph_context += f"  - Average PO Value: {avg_po_value:,.2f}\n"
                        graph_context += f"  - Average Items/PO: {avg_items_per_po:,.1f}\n"
                
                elif 'project' in question_lower:
                    # Get all projects and their relationships
                    projects = [n for n, d in self.graph._graph.nodes(data=True)
                               if any(edge[2].get('predicate') == 'has_po'
                                   for edge in self.graph._graph.edges(n, data=True))]
                    
                    project_stats = []
                    for project in projects:
                        # Get POs for this project
                        pos = [edge[1] for edge in self.graph._graph.edges(project, data=True)
                              if edge[2].get('predicate') == 'has_po']
                        
                        # Get suppliers and items for these POs
                        suppliers = set()
                        total_value = 0
                        total_items = 0
                        latest_po_date = None
                        po_dates = []
                        
                        for po in pos:
                            # Get supplier and track relationships
                            # Get supplier info and track relationships
                            supplier_edges = [edge[0] for edge in self.graph._graph.in_edges(po, data=True)
                                            if edge[2].get('predicate') == 'issued_po']
                            suppliers.update(supplier_edges)
                            
                            # Get value and payment terms
                            value_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                          if edge[2].get('predicate') == 'has_value']
                            payment_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                           if edge[2].get('predicate') == 'payment_terms']
                            
                            if value_edges:
                                try:
                                    value = float(value_edges[0][2].get('object_', '0').split()[0])
                                    total_value += value
                                except:
                                    pass
                            
                            # Get items and categories
                            item_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                         if edge[2].get('predicate') == 'has_item']
                            total_items += len(item_edges)
                            
                            # Track item categories if available
                            for item_edge in item_edges:
                                category_edges = [edge for edge in self.graph._graph.edges(item_edge[1], data=True)
                                                if edge[2].get('predicate') == 'category']
                            
                            # Get date and track timeline
                            date_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                         if edge[2].get('predicate') == 'issued_on']
                            if date_edges:
                                po_date = date_edges[0][2].get('object_')
                                if po_date:
                                    po_dates.append(po_date)
                                    if not latest_po_date or po_date > latest_po_date:
                                        latest_po_date = po_date
                        
                        # Calculate timeline metrics
                        po_dates.sort()
                        first_po = po_dates[0] if po_dates else 'Unknown'
                        last_po = po_dates[-1] if po_dates else 'Unknown'
                        
                        # Track supplier relationships
                        supplier_pos = {}
                        for supplier in suppliers:
                            supplier_pos[supplier] = len([edge[1] for edge in self.graph._graph.edges(supplier, data=True)
                                                         if edge[2].get('predicate') == 'issued_po' and
                                                         any(edge2[2].get('predicate') == 'has_po' and edge2[0] == project
                                                             for edge2 in self.graph._graph.in_edges(edge[1], data=True))])
                        
                        project_stats.append({
                            'name': project,
                            'pos': len(pos),
                            'suppliers': len(suppliers),
                            'supplier_details': supplier_pos,
                            'total_value': total_value,
                            'total_items': total_items,
                            'first_po': first_po,
                            'last_po': last_po,
                            'po_count': len(po_dates)
                        })
                    
                    # Sort by total value
                    project_stats.sort(key=lambda x: x['total_value'], reverse=True)
                    
                    # Format project information with timeline and relationships
                    graph_context += "\nProject Analysis:\n"
                    graph_context += "-" * 40 + "\n"
                    
                    for stat in project_stats:
                        graph_context += f"\nProject: {stat['name']}\n"
                        graph_context += f"  - Timeline:\n"
                        graph_context += f"    * First PO: {stat['first_po']}\n"
                        graph_context += f"    * Latest PO: {stat['last_po']}\n"
                        graph_context += f"    * Total POs: {stat['po_count']}\n"
                        graph_context += f"\n  - Financial:\n"
                        graph_context += f"    * Total Value: {stat['total_value']:,.2f}\n"
                        graph_context += f"    * Avg PO Value: {stat['total_value']/stat['pos']:,.2f}\n" if stat['pos'] > 0 else "\n"
                        graph_context += f"\n  - Items & Suppliers:\n"
                        graph_context += f"    * Total Items: {stat['total_items']}\n"
                        graph_context += f"    * Items per PO: {stat['total_items']/stat['pos']:,.1f}\n" if stat['pos'] > 0 else "\n"
                        graph_context += f"    * Active Suppliers: {stat['suppliers']}\n"
                        
                        # Show top suppliers
                        if stat['supplier_details']:
                            graph_context += f"\n  - Top Suppliers:\n"
                            sorted_suppliers = sorted(stat['supplier_details'].items(), key=lambda x: x[1], reverse=True)[:3]
                            for supplier, pos in sorted_suppliers:
                                graph_context += f"    * {supplier}: {pos} POs\n"
                
                elif 'item' in question_lower:
                    # Get all items and their relationships
                    items = []
                    for n, d in self.graph._graph.nodes(data=True):
                        item_edges = [edge for edge in self.graph._graph.edges(n, data=True)
                                     if edge[2].get('predicate') == 'has_item']
                        if item_edges:
                            # Get value and quantity
                            value_edges = [edge for edge in self.graph._graph.edges(n, data=True)
                                          if edge[2].get('predicate') == 'has_value']
                            quantity_edges = [edge for edge in self.graph._graph.edges(n, data=True)
                                            if edge[2].get('predicate') == 'has_quantity']
                            
                            value = 0
                            quantity = 0
                            
                            if value_edges:
                                try:
                                    value = float(value_edges[0][2].get('object_', '0').split()[0])
                                except:
                                    pass
                                    
                            if quantity_edges:
                                try:
                                    quantity = float(quantity_edges[0][2].get('object_', '0').split()[0])
                                except:
                                    pass
                            
                            # Get associated PO
                            po_edges = [edge[0] for edge in self.graph._graph.in_edges(n, data=True)
                                       if edge[2].get('predicate') == 'has_item']
                            
                            for po in po_edges:
                                # Get project and supplier
                                project_edges = [edge[0] for edge in self.graph._graph.in_edges(po, data=True)
                                                if edge[2].get('predicate') == 'has_po']
                                supplier_edges = [edge[0] for edge in self.graph._graph.in_edges(po, data=True)
                                                 if edge[2].get('predicate') == 'issued_po']
                                
                                items.append({
                                    'name': n,
                                    'value': value,
                                    'quantity': quantity,
                                    'unit_value': value/quantity if quantity else 0,
                                    'po': po,
                                    'project': project_edges[0] if project_edges else 'Unknown',
                                    'supplier': supplier_edges[0] if supplier_edges else 'Unknown'
                                })
                    
                    # Sort by total value
                    items.sort(key=lambda x: x['value'], reverse=True)
                    
                    # Format item information
                    graph_context += "\nTop Items by Value:\n"
                    for i, item in enumerate(items[:5], 1):
                        graph_context += f"\n{i}. {item['name']}\n"
                        graph_context += f"   Value: {item['value']:,.2f}\n"
                        graph_context += f"   Quantity: {item['quantity']}\n"
                        graph_context += f"   Unit Value: {item['unit_value']:,.2f}\n"
                        graph_context += f"   Project: {item['project']}\n"
                        graph_context += f"   Supplier: {item['supplier']}\n"
                        graph_context += f"   PO: {item['po']}\n"
                    
                    # Add summary statistics
                    total_items = len(items)
                    total_value = sum(item['value'] for item in items)
                    avg_value = total_value / total_items if total_items > 0 else 0
                    
                    graph_context += f"\nSummary Statistics:\n"
                    graph_context += f"Total Items: {total_items}\n"
                    graph_context += f"Total Value: {total_value:,.2f}\n"
                    graph_context += f"Average Value: {avg_value:,.2f}\n"
                
                else:
                    # Default analysis - show overview of all entity types
                    graph_context += "\nOverview of Purchase Order Data:\n"
                    
                    # Count entity types
                    suppliers = [n for n, d in self.graph._graph.nodes(data=True) 
                               if any(edge[2].get('predicate') == 'issued_po' 
                                   for edge in self.graph._graph.edges(n, data=True))]
                    
                    projects = [n for n, d in self.graph._graph.nodes(data=True)
                               if any(edge[2].get('predicate') == 'has_po'
                                   for edge in self.graph._graph.edges(n, data=True))]
                    
                    pos = [n for n, d in self.graph._graph.nodes(data=True)
                          if any(edge[2].get('predicate') == 'has_item'
                              for edge in self.graph._graph.edges(n, data=True))]
                    
                    items = [n for n, d in self.graph._graph.nodes(data=True)
                            if any(edge[2].get('predicate') == 'has_value'
                                for edge in self.graph._graph.edges(n, data=True))]
                    
                    # Calculate total value
                    total_value = 0
                    for po in pos:
                        value_edges = [edge for edge in self.graph._graph.edges(po, data=True)
                                      if edge[2].get('predicate') == 'has_value']
                        if value_edges:
                            try:
                                total_value += float(value_edges[0][2].get('object_', '0').split()[0])
                            except:
                                pass
                    
                    graph_context += f"\nEntity Counts:\n"
                    graph_context += f"  - Projects: {len(projects)}\n"
                    graph_context += f"  - Suppliers: {len(suppliers)}\n"
                    graph_context += f"  - Purchase Orders: {len(pos)}\n"
                    graph_context += f"  - Unique Items: {len(items)}\n"
                    graph_context += f"\nTotal Value: {total_value:,.2f}\n"
                    graph_context += f"Average PO Value: {total_value/len(pos):,.2f}\n" if pos else "\n"
                    graph_context += f"POs per Project: {len(pos)/len(projects):,.1f}\n" if projects else "\n"
                    graph_context += f"POs per Supplier: {len(pos)/len(suppliers):,.1f}\n" if suppliers else "\n"
            
            # Combine document and graph context
            context = "\n\n".join([doc.page_content for doc in docs])
            if graph_context:
                context += "\n\nGraph Analysis:\n" + graph_context
            
            # Create focused prompt
            prompt = f"""Based on the following context about purchase orders, projects, and suppliers, answer this specific question. 
            Focus on numerical facts and business relationships. If you cannot find the exact information, say 'I don't have enough information' 
            rather than making assumptions.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            
            # Get response from LLM with timeout
            try:
                response = self.llm.predict(prompt, timeout=10)
                return response
            except Exception as e:
                print(f"LLM timeout or error: {str(e)}")
                # Fall back to just the graph analysis if we have it
                if graph_context:
                    return f"Based on the graph analysis: {graph_context}"
                return "Sorry, I couldn't process this query in time."
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return "Sorry, I encountered an error while processing your query."
    
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
            
            print("\nDocument processing complete!")
            
            # Print graph statistics
            total_nodes = len(self.graph._graph.nodes())
            total_edges = len(self.graph._graph.edges())
            print(f"\nGraph Statistics:")
            print(f"Total Nodes: {total_nodes}")
            print(f"Total Edges: {total_edges}")
            
            # Count node types
            node_types = {}
            for node in self.graph._graph.nodes():
                node_type = "unknown"
                if any(edge[2].get('predicate') == 'has_po' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "project"
                elif any(edge[2].get('predicate') == 'issued_po' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "supplier"
                elif any(edge[2].get('predicate') == 'has_value' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "item"
                
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print("\nNode Types:")
            for node_type, count in node_types.items():
                print(f"{node_type.title()}: {count}")
            
        except Exception as e:
            print(f"\nError in document processing: {str(e)}")
            raise

    def save_graph(self, output_dir, visualization_options=None):
        """Save the knowledge graph to disk"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save graph in GEXF format for Gephi
            gexf_path = os.path.join(output_dir, "knowledge_graph.gexf")
            nx.write_gexf(self.graph._graph, gexf_path)
            print(f"\nSaved graph to {gexf_path}")
            
            # Save graph statistics
            stats = {
                "total_nodes": len(self.graph._graph.nodes()),
                "total_edges": len(self.graph._graph.edges()),
                "node_types": {},
                "edge_types": {}
            }
            
            # Count node types and prepare node colors for visualization
            node_colors = []
            node_sizes = []
            node_types_dict = {}
            
            for node in self.graph._graph.nodes():
                node_type = "unknown"
                if any(edge[2].get('predicate') == 'has_po' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "project"
                    node_types_dict[node] = "project"
                elif any(edge[2].get('predicate') == 'issued_po' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "supplier"
                    node_types_dict[node] = "supplier"
                elif any(edge[2].get('predicate') == 'has_value' for edge in self.graph._graph.edges(node, data=True)):
                    node_type = "item"
                    node_types_dict[node] = "item"
                else:
                    node_types_dict[node] = "unknown"
                
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count edge types
            for _, _, data in self.graph._graph.edges(data=True):
                edge_type = data.get('predicate', 'unknown')
                stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
            
            # Save statistics
            stats_path = os.path.join(output_dir, "graph_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_path}")

            # Generate and save a visual representation of the graph
            try:
                # Default visualization options
                default_options = {
                    "figsize": (16, 12),
                    "node_size": 100,
                    "font_size": 8,
                    "edge_width": 0.5,
                    "alpha": 0.7,
                    "iterations": 50,
                    "k": 0.2,  # Spring layout parameter
                    "color_scheme": {
                        "project": "#ff7f0e",  # Orange
                        "supplier": "#1f77b4",  # Blue
                        "item": "#2ca02c",     # Green
                        "unknown": "#d62728"   # Red
                    }
                }
                
                # Update with any user-provided options
                if visualization_options:
                    for key, value in visualization_options.items():
                        if key in default_options:
                            if isinstance(value, dict) and isinstance(default_options[key], dict):
                                default_options[key].update(value)
                            else:
                                default_options[key] = value
                
                options = default_options
                
                # Prepare node colors and sizes based on node types
                for node in self.graph._graph.nodes():
                    node_type = node_types_dict.get(node, "unknown")
                    node_colors.append(options["color_scheme"][node_type])
                    
                    # Make project and supplier nodes larger
                    if node_type in ["project", "supplier"]:
                        node_sizes.append(options["node_size"] * 1.5)
                    else:
                        node_sizes.append(options["node_size"])
                
                plt.figure(figsize=options["figsize"])
                pos = nx.spring_layout(self.graph._graph, k=options["k"], iterations=options["iterations"])
                
                # Draw edges
                nx.draw_networkx_edges(
                    self.graph._graph, 
                    pos, 
                    width=options["edge_width"],
                    alpha=options["alpha"]
                )
                
                # Draw nodes
                nx.draw_networkx_nodes(
                    self.graph._graph, 
                    pos, 
                    node_size=node_sizes,
                    node_color=node_colors,
                    alpha=options["alpha"]
                )
                
                # Draw labels
                nx.draw_networkx_labels(
                    self.graph._graph, 
                    pos, 
                    font_size=options["font_size"],
                    font_weight='bold'
                )
                
                # Add a legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type.title())
                    for node_type, color in options["color_scheme"].items()
                    if node_type in node_types_dict.values()
                ]
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.title("Knowledge Graph Visualization")
                plt.axis('off')  # Hide axes
                
                # Save as PNG
                img_path = os.path.join(output_dir, "knowledge_graph.png")
                plt.savefig(img_path, dpi=300, bbox_inches='tight')
                
                # Save as PDF for high-quality vector graphics
                pdf_path = os.path.join(output_dir, "knowledge_graph.pdf")
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
                
                plt.close()  # Close the figure to free memory
                print(f"Saved graph visualizations to {img_path} and {pdf_path}")
            except Exception as e:
                print(f"Error generating graph visualization: {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"Error saving graph: {str(e)}")
            return False


def visualize_only(graph_path, output_dir="output"):
    """Load a saved graph and generate visualizations without running the full system"""
    try:
        print(f"Loading graph from {graph_path}...")
        G = nx.read_gexf(graph_path)
        
        # Create a minimal GraphRAG instance just for visualization
        rag = GraphRAG(skip_llm=True)
        rag.graph._graph = G  # Replace the empty graph with the loaded one
        
        # Custom visualization options
        viz_options = {
            "figsize": (20, 16),  # Larger figure
            "node_size": 150,    # Larger nodes
            "font_size": 10,     # Larger font
            "k": 0.3,           # More spread out
            "iterations": 100    # More iterations for better layout
        }
        
        print(f"\nGenerating visualizations...")
        if rag.save_graph(output_dir, visualization_options=viz_options):
            print("Successfully generated visualizations")
        else:
            print("Failed to generate visualizations")
            
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

def main():
    """Main function to run the GraphRAG system"""
    try:
        # Check for command line arguments
        import sys
        if len(sys.argv) > 1:
            if sys.argv[1] == "--visualize-only" and len(sys.argv) > 2:
                # Just visualize an existing graph
                graph_path = sys.argv[2]
                output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
                visualize_only(graph_path, output_dir)
                return
        
        print("Initializing GraphRAG...")
        rag = GraphRAG(model_name="mistral", pdf_directory="D:\\OEssam\\01.pdfs")
        
        # Process documents
        print("\nProcessing documents...")
        rag.process_documents()
        
        # Example queries focused on business entities
        print("\nRunning example queries...")
        questions = [
            "List all suppliers and their total purchase order values",
            "Which projects have the most purchase orders?",
            "What are the top 5 most valuable items across all POs?",
            "Show me the payment terms for recent purchase orders",
            "Which suppliers are associated with which projects?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            try:
                answer = rag.query(question)
                print(f"A: {answer}")
            except Exception as e:
                print(f"Error processing query: {str(e)}")
        
        # Save the graph and statistics
        print("\nSaving graph and statistics...")
        if rag.save_graph("output"):
            print("Successfully saved graph and statistics")
        else:
            print("Failed to save graph and statistics")
                
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()