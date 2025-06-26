import asyncio
import json
import requests
import tkinter as tk
from tkinter import messagebox
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_band_name():
    # Create the main window
    root = tk.Tk()
    root.title('RateYourMusic Search')
    
    # Center the window
    window_width = 300
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Variable to store the input
    band_var = tk.StringVar()
    
    # Create and pack widgets
    tk.Label(root, text='Enter band name:', pady=10).pack()
    entry = tk.Entry(root, textvariable=band_var, width=30)
    entry.pack(pady=5)
    
    # Function to handle submit
    def on_submit():
        root.quit()
    
    tk.Button(root, text='Search', command=on_submit).pack(pady=10)
    
    # Focus on entry and bind Enter key
    entry.focus()
    entry.bind('<Return>', lambda e: on_submit())
    
    # Run the window
    root.mainloop()
    
    # Get the value and destroy the window
    value = band_var.get()
    root.destroy()
    return value

async def search_band(band_name: str):
    # MCP Puppeteer endpoint
    base_url = 'http://localhost:1124'
    max_retries = 3
    retry_delay = 2
    
    logging.info(f'Starting search for band: {band_name}')
    
    for attempt in range(max_retries):
        logging.debug(f'Attempt {attempt + 1} of {max_retries}')
        try:
            # Test connection first
            try:
                logging.debug('Testing connection to Puppeteer MCP server...')
                response = requests.get(f'{base_url}')
                logging.debug(f'Server response status: {response.status_code}')
            except requests.exceptions.ConnectionError as e:
                logging.error(f'Connection error: {str(e)}')
                messagebox.showerror('Error', 'Cannot connect to Puppeteer MCP server. Please make sure it is running.')
                return
            
            # Navigate to RateYourMusic
            response = requests.post(f'{base_url}/navigate', json={
                'url': 'https://rateyourmusic.com/'
            })
            response.raise_for_status()
            
            # Wait for a moment to ensure page loads
            await asyncio.sleep(2)
            
            # Find and click the search button
            response = requests.post(f'{base_url}/click', json={
                'selector': '.searchIcon'
            })
            response.raise_for_status()
            
            # Find and fill the search input
            response = requests.post(f'{base_url}/fill', json={
                'selector': '.searchInput',
                'value': band_name
            })
            response.raise_for_status()
            
            # Press Enter using JavaScript
            response = requests.post(f'{base_url}/evaluate', json={
                'script': '''
                    document.querySelector('.searchInput').dispatchEvent(
                        new KeyboardEvent('keypress', {'key': 'Enter'})
                    )
                '''
            })
            response.raise_for_status()
            
            # Wait for results to load
            await asyncio.sleep(2)
            
            # Take a screenshot of the results
            response = requests.post(f'{base_url}/screenshot', json={
                'name': 'search_results',
                'selector': '.searchResults',
                'width': 1200,
                'height': 800
            })
            response.raise_for_status()
            
            messagebox.showinfo('Success', 'Search completed successfully!')
            break  # Break the retry loop on success
            
        except requests.exceptions.RequestException as e:
            logging.error(f'Request failed: {str(e)}')
            if attempt < max_retries - 1:
                msg = f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds..."
                logging.info(msg)
                messagebox.showinfo('Retrying', msg)
                time.sleep(retry_delay)
            else:
                error_msg = f'Failed to complete the search after {max_retries} attempts: {str(e)}'
                logging.error(error_msg)
                messagebox.showerror('Error', error_msg)

# Run the search
if __name__ == '__main__':
    band_name = get_band_name()
    if band_name.strip():  # Only search if band name is not empty
        asyncio.run(search_band(band_name))
    else:
        messagebox.showwarning('Warning', 'Please enter a band name to search.')
