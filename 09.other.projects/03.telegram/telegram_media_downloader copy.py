import os
import json
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import Channel, Chat, MessageMediaPhoto, MessageMediaDocument
from telethon.tl.types import DocumentAttributeVideo, DocumentAttributeAudio
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

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

def download_media_terminal(client, group, group_name):
    safe_group_name = ''.join(c for c in group_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    group_folder = os.path.join(DOWNLOAD_BASE, safe_group_name)
    os.makedirs(group_folder, exist_ok=True)
    progress_json = os.path.join(group_folder, 'progress.json')
    downloaded_ids = set()
    last_id = 0
    if os.path.exists(progress_json):
        try:
            with open(progress_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                downloaded_ids = set(data.get('downloaded_ids', []))
                last_id = data.get('last_id', 0)
        except Exception:
            pass
    all_media = []
    print('Fetching messages...')
    for message in client.iter_messages(group, reverse=True):
        if message.id in downloaded_ids:
            continue
        if message.media:
            # Photo
            if isinstance(message.media, MessageMediaPhoto):
                all_media.append(message)
            # Video (Document with video attribute)
            elif isinstance(message.media, MessageMediaDocument):
                attrs = message.media.document.attributes
                is_video = any(isinstance(a, DocumentAttributeVideo) for a in attrs)
                is_audio = any(isinstance(a, DocumentAttributeAudio) for a in attrs)
                if is_video and not is_audio:
                    all_media.append(message)
    total = len(all_media)
    print(f'Found {total} media files. Starting download...')
    count = 0
    for message in all_media:
        count += 1
        print(f'Downloading {count} of {total} (ID: {message.id})...')
        try:
            client.download_media(message, group_folder)
            downloaded_ids.add(message.id)
            last_id = max(last_id, message.id)
            with open(progress_json, 'w', encoding='utf-8') as f:
                json.dump({'downloaded_ids': list(downloaded_ids), 'last_id': last_id}, f)
        except Exception as e:
            print(f'Error downloading ID {message.id}: {e}')
    print(f'Finished! Downloaded {count} media files to {group_folder}')

class TelegramDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Telegram Media Downloader')
        self.root.geometry('600x300')

        self.phone_var = tk.StringVar()
        self.group_var = tk.StringVar()
        self.groups = []

        self.create_widgets()
        self.try_auto_login()

    def create_widgets(self):
        self.phone_label = tk.Label(self.root, text='Enter your phone number (with country code):')
        self.phone_label.pack(pady=10)
        self.phone_entry = tk.Entry(self.root, textvariable=self.phone_var)
        self.phone_entry.pack(pady=5)
        self.login_btn = tk.Button(self.root, text='Login & List Groups', command=self.login_and_list_groups)
        self.login_btn.pack(pady=10)

        self.group_combo = ttk.Combobox(self.root, textvariable=self.group_var, state='readonly', width=60)
        self.group_combo.pack(pady=10)
        self.group_combo['values'] = []

        self.select_btn = tk.Button(self.root, text='Select Group and Start Download', state='disabled', command=self.select_group_and_close)
        self.select_btn.pack(pady=20)

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
            group_names = [f"{g.name} (ID: {g.id})" for g in self.groups]
            self.group_combo['values'] = group_names
            if group_names:
                self.group_combo.current(0)
                self.select_btn.config(state='normal')
            else:
                messagebox.showinfo('Info', 'No groups or channels found.')
        except Exception as e:
            messagebox.showerror('Group Listing Error', str(e))

    def select_group_and_close(self):
        global selected_group, selected_group_name
        idx = self.group_combo.current()
        if idx < 0 or idx >= len(self.groups):
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