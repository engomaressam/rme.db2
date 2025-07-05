import os
import shutil
import tkinter as tk
from tkinter import messagebox, ttk

NETWORK_MODELS = r'H:\Projects Control (PC)\10 Backup\07 Ollama'
LOCAL_MODELS = r'D:\OEssam\models'

class ModelManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Ollama Model Manager')
        self.root.geometry('700x400')

        self.network_models = self.get_model_list(NETWORK_MODELS)
        self.local_models = self.get_model_list(LOCAL_MODELS)

        self.create_widgets()
        self.refresh_lists()

    def get_model_list(self, base_dir):
        if not os.path.exists(base_dir):
            return []
        # Only show folders with a modelfile or .bin file inside
        models = []
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                files = os.listdir(path)
                if any(f.endswith('.bin') or f.lower() == 'modelfile' for f in files):
                    models.append(name)
        return sorted(models)

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Network models
        tk.Label(frame, text='Network Models', font=('Arial', 12, 'bold')).grid(row=0, column=0)
        self.network_listbox = tk.Listbox(frame, selectmode=tk.SINGLE, width=40, height=15)
        self.network_listbox.grid(row=1, column=0, padx=5)

        # Local models
        tk.Label(frame, text='Local Models', font=('Arial', 12, 'bold')).grid(row=0, column=2)
        self.local_listbox = tk.Listbox(frame, selectmode=tk.SINGLE, width=40, height=15)
        self.local_listbox.grid(row=1, column=2, padx=5)

        # Add button
        self.add_button = tk.Button(frame, text='Add →', command=self.copy_selected_model, width=10)
        self.add_button.grid(row=1, column=1, pady=10)

        # Refresh button
        self.refresh_button = tk.Button(frame, text='Refresh', command=self.refresh_lists)
        self.refresh_button.grid(row=2, column=1)

    def refresh_lists(self):
        self.network_models = self.get_model_list(NETWORK_MODELS)
        self.local_models = self.get_model_list(LOCAL_MODELS)
        self.network_listbox.delete(0, tk.END)
        self.local_listbox.delete(0, tk.END)
        for m in self.network_models:
            self.network_listbox.insert(tk.END, m)
        for m in self.local_models:
            self.local_listbox.insert(tk.END, m)

    def copy_selected_model(self):
        sel = self.network_listbox.curselection()
        if not sel:
            messagebox.showinfo('No selection', 'Please select a model from the network list.')
            return
        model_name = self.network_models[sel[0]]
        if model_name in self.local_models:
            messagebox.showinfo('Already exists', f'Model "{model_name}" already exists locally.')
            return
        src = os.path.join(NETWORK_MODELS, model_name)
        dst = os.path.join(LOCAL_MODELS, model_name)
        try:
            shutil.copytree(src, dst)
            messagebox.showinfo('Success', f'Model "{model_name}" copied to local models.')
            self.refresh_lists()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to copy model: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    app = ModelManagerApp(root)
    root.mainloop()
