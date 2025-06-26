"""
Alstom Project Assistant - Document Server (Improved Version)
This script provides a simple HTTP server to serve documents from the network path.
This version uses chunked transfer to handle large files more efficiently.
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
LOCAL_PATH = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\05.llm\alstom"
SERVER_PORT = 7861  # Use a different port than the chatbot
CHUNK_SIZE = 8192  # Read and send files in smaller chunks

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
            if doc_path.startswith(LOCAL_PATH):
                full_path = doc_path
            elif doc_path.startswith(NETWORK_PATH):
                full_path = doc_path
            else:
                # If it's not a recognized path, try to find it in both locations
                local_relative = os.path.join(LOCAL_PATH, os.path.basename(doc_path))
                network_relative = os.path.join(NETWORK_PATH, os.path.basename(doc_path))
                
                if os.path.isfile(local_relative):
                    full_path = local_relative
                elif os.path.isfile(network_relative):
                    full_path = network_relative
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
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'public, max-age=86400')  # Cache for 24 hours
            self.end_headers()
            
            # Send the file in chunks to avoid memory issues with large files
            with open(full_path, 'rb') as file:
                while True:
                    chunk = file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except ConnectionError as ce:
                        logger.error(f"Connection error while sending file: {str(ce)}")
                        return
                    except Exception as e:
                        logger.error(f"Error sending file chunk: {str(e)}")
                        return
                
            logger.info(f"Successfully served document: {file_name}")
            
        except Exception as e:
            logger.error(f"Error serving document: {str(e)}")
            try:
                self.send_error(500, f"Error serving document: {str(e)}")
            except:
                logger.error("Could not send error response")
    
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

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True
    daemon_threads = True

def main():
    try:
        # Create a threaded socket server
        with ThreadedHTTPServer(("", SERVER_PORT), DocumentRequestHandler) as httpd:
            logger.info(f"Document server started at port {SERVER_PORT}")
            logger.info(f"Serving documents from: {NETWORK_PATH} and {LOCAL_PATH}")
            logger.info(f"Access the server at: http://localhost:{SERVER_PORT}")
            
            # Serve until interrupted
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == "__main__":
    main()
