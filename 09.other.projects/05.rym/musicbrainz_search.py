import tkinter as tk
from tkinter import messagebox, ttk
import requests
import json
from datetime import datetime
import webbrowser
import traceback
import sys
from lastfm_config import LASTFM_API_KEY

class MainWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('MusicBrainz Search')
        
        # Center the window
        window_width = 300
        window_height = 150
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Create search frame
        self.band_var = tk.StringVar()
        
        tk.Label(self.window, text='Enter band name:', pady=10).pack()
        self.entry = tk.Entry(self.window, textvariable=self.band_var, width=30)
        self.entry.pack(pady=5)
        
        search_button = tk.Button(self.window, text='Search', command=self.search)
        search_button.pack(pady=10)
        
        # Bind enter key
        self.entry.bind('<Return>', lambda e: self.search())
        self.entry.focus()
    
    def search(self):
        band_name = self.band_var.get().strip()
        if band_name:
            print(f"Searching for: {band_name}")
            search_band(self.window, band_name)
            self.entry.delete(0, tk.END)  # Clear the entry
        else:
            messagebox.showwarning('Warning', 'Please enter a band name.')

class LastFMAPI:
    def __init__(self):
        self.api_key = LASTFM_API_KEY
        self.base_url = 'http://ws.audioscrobbler.com/2.0/'
    
    def search_artist(self, name):
        params = {
            'method': 'artist.search',
            'artist': name,
            'api_key': self.api_key,
            'format': 'json',
            'limit': 10
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_top_albums(self, artist_name):
        params = {
            'method': 'artist.gettopalbums',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json',
            'limit': 50
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

class MusicBrainzSearch:
    def __init__(self):
        self.headers = {
            'User-Agent': 'MusicSearchApp/1.0.0 (your@email.com)',
            'Accept': 'application/json'
        }
        self.base_url = 'https://musicbrainz.org/ws/2'
    
    def get_albums(self, artist_id):
        url = f'{self.base_url}/release-group'
        params = {
            'artist': artist_id,
            'fmt': 'json',
            'type': 'album|ep',
            'limit': 100
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

def create_results_window(parent, title):
    window = tk.Toplevel(parent)
    window.title(title)
    window.geometry('800x600')
    window.transient(parent)
    window.grab_set()
    
    # Create main frame
    main_frame = ttk.Frame(window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create treeview
    tree = ttk.Treeview(main_frame, show='headings')
    tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
    tree.configure(yscrollcommand=scrollbar.set)
    
    return window, tree

def search_band(parent, band_name):
    try:
        print(f"Searching for band: {band_name}")
        mb = MusicBrainzSearch()
        
        # Search for artists
        url = f'{mb.base_url}/artist'
        params = {
            'query': band_name,
            'fmt': 'json'
        }
        
        response = requests.get(url, headers=mb.headers, params=params)
        response.raise_for_status()
        artists_data = response.json()
        
        if not artists_data['artists']:
            messagebox.showinfo('No Results', f'No artists found for {band_name}')
            return
        
        # Create artists window
        window, tree = create_results_window(parent, f'Artists matching "{band_name}"')
        
        # Configure columns
        tree['columns'] = ('name', 'type', 'country', 'score')
        tree.heading('name', text='Name')
        tree.heading('type', text='Type')
        tree.heading('country', text='Country')
        tree.heading('score', text='Score')
        
        # Store artist IDs for later use
        artist_ids = {}
        
        # Add artists to treeview
        for artist in artists_data['artists']:
            artist_id = artist['id']
            artist_ids[artist['name']] = artist_id
            tree.insert('', tk.END, values=(
                artist.get('name', 'N/A'),
                artist.get('type', 'N/A'),
                artist.get('country', 'N/A'),
                artist.get('score', 'N/A')
            ))
        
        def on_artist_select(event):
            selection = tree.selection()
            if not selection:
                return
            
            item = tree.item(selection[0])
            artist_name = item['values'][0]
            artist_id = artist_ids[artist_name]
            
            try:
                # Get albums
                albums_data = mb.get_albums(artist_id)
                
                # Create albums window
                albums_window, albums_tree = create_results_window(parent, f'Albums by {artist_name}')
                
                # Configure columns
                albums_tree['columns'] = ('title', 'type', 'year', 'playcount', 'listeners')
                albums_tree.heading('title', text='Title')
                albums_tree.heading('type', text='Type')
                albums_tree.heading('year', text='Year')
                albums_tree.heading('playcount', text='Play Count')
                albums_tree.heading('listeners', text='Listeners')
                
                # Get Last.fm data
                try:
                    lastfm = LastFMAPI()
                    lastfm_albums = lastfm.get_top_albums(artist_name)
                    
                    # Create album name to stats mapping
                    album_stats = {}
                    for album in lastfm_albums['topalbums']['album']:
                        album_stats[album['name'].lower()] = {
                            'playcount': album.get('playcount', 'N/A'),
                            'listeners': album.get('listeners', 'N/A')
                        }
                except Exception as e:
                    print(f"Last.fm error: {e}")
                    album_stats = {}
                
                # Add albums to treeview
                for album in albums_data['release-groups']:
                    title = album.get('title', 'N/A')
                    first_release_date = album.get('first-release-date', 'N/A')[:4]
                    stats = album_stats.get(title.lower(), {'playcount': 'N/A', 'listeners': 'N/A'})
                    
                    albums_tree.insert('', tk.END, values=(
                        title,
                        album.get('primary-type', 'N/A'),
                        first_release_date,
                        f"{int(stats['playcount']):,}" if stats['playcount'] != 'N/A' else 'N/A',
                        f"{int(stats['listeners']):,}" if stats['listeners'] != 'N/A' else 'N/A'
                    ))
                
                # Save full data to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"albums_{artist_name.replace(' ', '_')}_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(albums_data, f, indent=2)
                
            except Exception as e:
                messagebox.showerror('Error', f'Failed to fetch albums: {str(e)}')
        
        # Bind double-click event
        tree.bind('<Double-1>', on_artist_select)
        
        # Add instructions label
        ttk.Label(window, text='Double-click an artist to view their albums').pack(pady=5)
        
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {str(e)}')

if __name__ == '__main__':
    try:
        app = MainWindow()
        app.window.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        messagebox.showerror('Error', f'An unexpected error occurred: {str(e)}')
