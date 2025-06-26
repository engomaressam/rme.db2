"""
Alstom Project Assistant - Document Server
This script provides a simple HTTP server to serve documents from the network path.
Run this alongside the chatbot to enable document access from any browser.
"""

import os
import json
import logging
import http.server
import socketserver
import urllib.parse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DocumentServer")

# Define paths
NETWORK_PATH = r"\\fileserver2\Head Office Server\Projects Control (PC)\10 Backup\05 Models\alstom"
SERVER_PORT = 7861  # Use a different port than the chatbot

# Custom request handler
class DocumentRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        # Check if this is a document request
        if parsed_path.path == '/document':
            if 'path' in query_params:
                doc_path = query_params['path'][0]
                self.serve_document(doc_path)
            else:
                self.send_error(400, "Missing document path parameter")
        elif parsed_path.path == '/':
            # Serve the document index page
            self.serve_index()
        else:
            self.send_error(404, "Not found")
    
    def serve_document(self, doc_path):
        # Construct the full path
        try:
            # Ensure the path is within the allowed directory
            if not doc_path.startswith(NETWORK_PATH):
                full_path = os.path.join(NETWORK_PATH, doc_path.lstrip('\\'))
            else:
                full_path = doc_path
                
            logger.info(f"Attempting to serve document: {full_path}")
            
            # Check if the file exists
            if not os.path.isfile(full_path):
                self.send_error(404, f"Document not found: {os.path.basename(full_path)}")
                return
            
            # Get the file size
            file_size = os.path.getsize(full_path)
            
            # Get the file name for Content-Disposition
            file_name = os.path.basename(full_path)
            
            # Set headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Disposition', f'inline; filename="{file_name}"')
            self.send_header('Content-Length', str(file_size))
            self.end_headers()
            
            # Send the file
            with open(full_path, 'rb') as file:
                self.wfile.write(file.read())
                
            logger.info(f"Successfully served document: {file_name}")
            
        except Exception as e:
            logger.error(f"Error serving document: {str(e)}")
            self.send_error(500, f"Error serving document: {str(e)}")
    
    def serve_index(self):
        # Create a simple HTML index page
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Alstom Document Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .info { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Alstom Project Assistant - Document Server</h1>
            <div class="info">
                <p>This server provides access to Alstom project documents.</p>
                <p>To access a document, use the URL: <code>/document?path=[document_path]</code></p>
                <p>This server works in conjunction with the Alstom Project Assistant chatbot.</p>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

def main():
    try:
        # Create a socket server
        with socketserver.TCPServer(("", SERVER_PORT), DocumentRequestHandler) as httpd:
            logger.info(f"Document server started at port {SERVER_PORT}")
            logger.info(f"Serving documents from: {NETWORK_PATH}")
            logger.info(f"Access the server at: http://localhost:{SERVER_PORT}")
            
            # Serve until interrupted
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == "__main__":
    main()
