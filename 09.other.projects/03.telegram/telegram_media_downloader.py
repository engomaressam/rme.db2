import os
import json
import time
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError, TimeoutError as TelethonTimeoutError, FloodWaitError
from telethon.tl.types import Channel, Chat, MessageMediaPhoto, MessageMediaDocument
from telethon.tl.types import DocumentAttributeVideo, DocumentAttributeAudio
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font

# Set base directory to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
SESSION_FILE = os.path.join(BASE_DIR, 'gui_session')
DOWNLOAD_BASE = r'D:\d\Download\Tele.API'

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')

client = None
selected_group = None
selected_group_name = None

def retry_download(client, message, folder, max_retries=3, initial_delay=1.0):
    """Attempt to download media with exponential backoff retry logic."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return client.download_media(message, folder)
        except TelethonTimeoutError as e:
            if attempt < max_retries - 1:
                print(f"Timeout occurred (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to download after {max_retries} attempts (ID: {message.id}): {str(e)}")
                return None
        except FloodWaitError as e:
            wait_time = e.seconds
            print(f"Rate limited by Telegram. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"Unexpected error downloading (ID: {message.id}): {str(e)}")
            return None
    return None

def download_media_terminal(client, group, group_name):
    safe_group_name = ''.join(c for c in group_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    group_folder = os.path.join(DOWNLOAD_BASE, safe_group_name)
    os.makedirs(group_folder, exist_ok=True)
    progress_json = os.path.join(group_folder, 'progress.json')
    downloaded_ids = set()
    last_id = 0
    
    # Load progress if exists
    if os.path.exists(progress_json):
        try:
            with open(progress_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                downloaded_ids = set(data.get('downloaded_ids', []))
                last_id = data.get('last_id', 0)
        except Exception as e:
            print(f"Error loading progress file: {e}")
    print(f"Loaded {len(downloaded_ids)} downloaded IDs, last_id: {last_id}")  # Debug print
    
    processed = 0
    downloaded = 0
    skipped = 0
    errors = 0
    print('Fetching and downloading messages...')
    
    try:
        # Get the latest message ID
        latest_msg = next(client.iter_messages(group, limit=1), None)
        latest_id = latest_msg.id if latest_msg else 0
        
        # Use min_id=0 to check all messages, but always skip IDs in downloaded_ids
        for message in client.iter_messages(group, min_id=0, reverse=True):
            try:
                if message.id in downloaded_ids:
                    continue
                processed += 1
                if message.id > latest_id:
                    continue
                
                if message.media:
                    filename = None
                    # Handle photos
                    if isinstance(message.media, MessageMediaPhoto):
                        filename = retry_download(client, message, group_folder)
                    # Handle videos
                    elif isinstance(message.media, MessageMediaDocument):
                        attrs = message.media.document.attributes
                        is_video = any(isinstance(a, DocumentAttributeVideo) for a in attrs)
                        is_audio = any(isinstance(a, DocumentAttributeAudio) for a in attrs)
                        if is_video and not is_audio:
                            filename = retry_download(client, message, group_folder)
                    
                    # Process download result
                    if filename and os.path.exists(filename):
                        if os.path.getsize(filename) > 0:
                            downloaded_ids.add(message.id)
                            downloaded += 1
                            last_id = max(last_id, message.id)
                            # Save progress after each successful download
                            try:
                                with open(progress_json, 'w', encoding='utf-8') as f:
                                    json.dump({
                                        'downloaded_ids': list(downloaded_ids),
                                        'last_id': last_id
                                    }, f)
                            except Exception as e:
                                print(f"Warning: Could not save progress: {e}")
                        else:
                            os.remove(filename)  # Remove 0-byte file
                            errors += 1
                    else:
                        errors += 1
                else:
                    skipped += 1
                
                # Print progress
                print(f"Progress: Downloaded {downloaded}, Errors {errors}, Skipped {skipped}, "
                      f"Processed {processed}/{latest_id} (ID: {message.id})")
                
            except Exception as e:
                print(f"Error processing message {message.id}: {str(e)}")
                errors += 1
                continue
                
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        # Save final progress
        try:
            with open(progress_json, 'w', encoding='utf-8') as f:
                json.dump({
                    'downloaded_ids': list(downloaded_ids),
                    'last_id': last_id
                }, f)
        except Exception as e:
            print(f"Warning: Could not save final progress: {e}")
        
        print(f'\nDownload summary:')
        print(f'Successfully downloaded: {downloaded} files')
        print(f'Errors encountered: {errors} files')
        print(f'Skipped (no media): {skipped} files')
        print(f'Total processed: {processed} messages')
        print(f'Download location: {group_folder}')

class TelegramDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Telegram Media Downloader')
        self.root.geometry('800x500')

        self.phone_var = tk.StringVar()
        self.groups = []
        self.group_status = []
        self.selected_index = None

        self.create_widgets()
        self.try_auto_login()

    def create_widgets(self):
        big_font = font.Font(size=14)
        label_font = font.Font(size=16, weight='bold')

        self.phone_label = tk.Label(self.root, text='Enter your phone number (with country code):', font=label_font)
        self.phone_label.pack(pady=10)
        self.phone_entry = tk.Entry(self.root, textvariable=self.phone_var, font=big_font, width=30)
        self.phone_entry.pack(pady=5)
        self.login_btn = tk.Button(self.root, text='Login & List Groups', font=big_font, command=self.login_and_list_groups)
        self.login_btn.pack(pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=10, fill='both', expand=True)

        self.group_listbox = tk.Listbox(frame, font=big_font, width=60, height=15, selectmode=tk.SINGLE)
        self.group_listbox.pack(side='left', fill='both', expand=True)
        self.group_listbox.bind('<<ListboxSelect>>', self.on_group_select)

        self.scrollbar = tk.Scrollbar(frame, orient='vertical', command=self.group_listbox.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.group_listbox.config(yscrollcommand=self.scrollbar.set)

        self.download_btn = tk.Button(self.root, text='Download Selected Group', font=big_font, state='disabled', command=self.select_group_and_close)
        self.download_btn.pack(pady=20)

    def on_group_select(self, event):
        selection = self.group_listbox.curselection()
        if selection:
            self.selected_index = selection[0]
            self.download_btn.config(state='normal')
        else:
            self.selected_index = None
            self.download_btn.config(state='disabled')

    def try_auto_login(self):
        if os.path.exists(SESSION_FILE + '.session'):
            try:
                global client
                client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
                client.connect()
                if client.is_user_authorized():
                    self.phone_label.pack_forget()
                    self.phone_entry.pack_forget()
                    self.login_btn.pack_forget()
                    self.list_groups()
            except Exception as e:
                messagebox.showerror('Auto Login Error', str(e))

    def login_and_list_groups(self):
        global client
        phone = self.phone_var.get().strip()
        if not phone:
            messagebox.showerror('Error', 'Please enter your phone number.')
            return
        try:
            client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
            client.connect()
            if not client.is_user_authorized():
                code_callback = lambda: simpledialog.askstring('Telegram Code', 'Enter the code you received:')
                client.send_code_request(phone)
                code = code_callback()
                client.sign_in(phone, code)
                # Handle 2FA password if enabled
                if client.is_user_authorized() is False:
                    try:
                        password = simpledialog.askstring('Telegram 2FA', 'Enter your Telegram password:', show='*')
                        client.sign_in(password=password)
                    except SessionPasswordNeededError:
                        messagebox.showerror('2FA Error', '2FA password required but not provided.')
                        return
            self.phone_label.pack_forget()
            self.phone_entry.pack_forget()
            self.login_btn.pack_forget()
            self.list_groups()
        except Exception as e:
            messagebox.showerror('Login Error', str(e))

    def list_groups(self):
        try:
            dialogs = client.get_dialogs()
            self.groups = [d for d in dialogs if isinstance(d.entity, (Channel, Chat))]
            self.group_status = []
            group_names = []
            for g in self.groups:
                group_name = f"{g.name} (ID: {g.id})"
                safe_group_name = ''.join(c for c in g.name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                group_folder = os.path.join(DOWNLOAD_BASE, safe_group_name)
                if os.path.exists(group_folder) and os.path.isdir(group_folder):
                    status = "(downloaded)"
                else:
                    status = "(not downloaded)"
                group_names.append(f"{group_name} {status}")
                self.group_status.append(status)
            self.group_listbox.delete(0, tk.END)
            for name in group_names:
                self.group_listbox.insert(tk.END, name)
            if group_names:
                self.group_listbox.selection_set(0)
                self.download_btn.config(state='normal')
            else:
                messagebox.showinfo('Info', 'No groups or channels found.')
        except Exception as e:
            messagebox.showerror('Group Listing Error', str(e))

    def select_group_and_close(self):
        global selected_group, selected_group_name
        idx = self.selected_index
        if idx is None or idx < 0 or idx >= len(self.groups):
            messagebox.showerror('Error', 'Please select a group/channel.')
            return
        selected_group = self.groups[idx].entity
        selected_group_name = getattr(selected_group, 'title', getattr(selected_group, 'name', str(selected_group.id)))
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = TelegramDownloaderApp(root)
    root.mainloop()
    if selected_group is not None:
        download_media_terminal(client, selected_group, selected_group_name) 