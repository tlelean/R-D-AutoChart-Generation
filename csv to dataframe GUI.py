import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pathlib import Path
import subprocess
import pandas as pd
import os

# Fallback for content sniffing if naming fails
def is_data_csv(path):
    try:
        with open(path, 'r') as f:
            first = f.readline().strip().split(',')[0]
        for fmt in ('%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S'):
            try:
                from datetime import datetime
                datetime.strptime(first, fmt)
                return True
            except ValueError:
                continue
    except Exception:
        pass
    return False


def load_details_csv(path):
    metadata_df = pd.read_csv(path, header=None, usecols=[0, 1], nrows=16, index_col=0).fillna('')
    channel_df = pd.read_csv(path, header=None, usecols=[0, 3], skiprows=16, nrows=21, index_col=0).fillna(False)
    breakout_or_key = pd.read_csv(path, header=None, usecols=[0], skiprows=37, nrows=1).iat[0, 0]
    breakout_df = None
    if breakout_or_key == 'Breakouts':
        breakout_df = pd.read_csv(path, header=None, usecols=[0, 1], skiprows=38).dropna(how='all').fillna('')
    return metadata_df, channel_df, breakout_df, breakout_or_key


def main_gui():
    root = tk.Tk()
    root.title("CSV Editor & PDF Generator")
    root.geometry('800x800')  # Taller window to avoid scrolling metadata
    root.resizable(True, True)
    root.configure(bg='#2e2e2e')

    style = ttk.Style(root)
    style.theme_use('clam')
    # Dark theme
    style.configure('TFrame', background='#2e2e2e')
    style.configure('TLabel', background='#2e2e2e', foreground='#f0f0f0', font=('Segoe UI', 12))
    style.configure('TButton', background='#444', foreground='#f0f0f0', font=('Segoe UI', 12))
    style.map('TButton', background=[('active', '#555')])
    style.configure('TEntry', fieldbackground='#3e3e3e', foreground='#f0f0f0', font=('Segoe UI', 12))
    style.configure('TCheckbutton', background='#2e2e2e', foreground='#f0f0f0', font=('Segoe UI', 12))
    # Highlight color for checkbuttons
    style.map('TCheckbutton',
        background=[('active', '#555555'), ('selected', '#333333')],
        indicatorbackground=[('selected', '#00ff00')],
        indicatorforeground=[('selected', '#00ff00')]
    )
    style.configure('TNotebook', background='#2e2e2e')
    style.configure('TNotebook.Tab', background='#444', foreground='#f0f0f0', font=('Segoe UI', 12, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', '#333')], foreground=[('selected', '#ffffff')])

    # State
    data_path = None
    details_path = None
    metadata_df = None
    channel_df = None
    breakout_df = None
    mode = None
    metadata_vars = {}
    channel_vars = {}
    breakout_vars = {}
    original_dt_str = ''

    # Helper functions
    def select_two():
        nonlocal data_path, details_path
        files = filedialog.askopenfilenames(title="Select two CSVs", filetypes=[("CSV","*.csv")])
        if len(files) != 2:
            messagebox.showerror("Error", "Select exactly two CSV files.")
            return
        f1, f2 = files
        name1, name2 = Path(f1).name.lower(), Path(f2).name.lower()
        if 'test_details' in name1:
            details_path, data_path = f1, f2
        elif 'test_details' in name2:
            details_path, data_path = f2, f1
        elif 'data_' in name1 or name1.startswith('data'):
            data_path, details_path = f1, f2
        elif 'data_' in name2 or name2.startswith('data'):
            data_path, details_path = f2, f1
        else:
            if is_data_csv(f1) and not is_data_csv(f2): data_path, details_path = f1, f2
            elif is_data_csv(f2) and not is_data_csv(f1): data_path, details_path = f2, f1
            else:
                messagebox.showerror("Error", "Could not identify files.")
                return
        # Show on two lines, wrapped
        sel_label.config(text=f"Data: {Path(data_path).name}\nDetails: {Path(details_path).name}")
        # Auto-load details
        load_details()

    def browse_dir():
        d = filedialog.askdirectory()
        if d:
            out_entry.delete(0, tk.END)
            out_entry.insert(0, d)

    def load_details():
        nonlocal metadata_df, channel_df, breakout_df, mode, original_dt_str
        if not details_path:
            return
        metadata_df, channel_df, breakout_df, mode = load_details_csv(details_path)
        original_dt_str = metadata_df.loc['Date Time'].iat[0] if 'Date Time' in metadata_df.index else ''
        build_metadata_tab()
        build_channels_tab()
        build_breakouts_tab()

    def build_metadata_tab():
        for w in meta_tab.winfo_children(): w.destroy()
        metadata_vars.clear()
        for row, label in enumerate(metadata_df.index):
            if label == 'Date Time': continue
            val = metadata_df.loc[label].iat[0]
            ttk.Label(meta_tab, text=f"{label}:").grid(row=row, column=0, sticky='e', padx=10, pady=5)
            var = tk.StringVar(value=str(val))
            entry = ttk.Entry(meta_tab, textvariable=var, width=50, font=('Segoe UI',12))
            entry.grid(row=row, column=1, sticky='w', pady=5)
            metadata_vars[label] = var

    def build_channels_tab():
        for w in chan_tab.winfo_children(): w.destroy()
        channel_vars.clear()
        cols = 3
        for idx, (ch, row) in enumerate(channel_df.iterrows()):
            var = tk.BooleanVar(value=bool(row.iat[0]))
            ttk.Checkbutton(chan_tab, text=ch, variable=var).grid(row=idx//cols, column=idx%cols, sticky='w', padx=10, pady=5)
            channel_vars[ch] = var

    def build_breakouts_tab():
        for w in br_tab.winfo_children(): w.destroy()
        breakout_vars.clear()
        if breakout_df is None:
            ttk.Label(br_tab, text="No breakouts available").grid(row=0, column=0, padx=10, pady=10)
            return
        for idx, row in breakout_df.iterrows():
            lbl = row.iat[0] or f"Breakout {idx+1}"
            var = tk.StringVar(value=str(row.iat[1]))
            ttk.Label(br_tab, text=f"{lbl}:").grid(row=idx, column=0, sticky='e', padx=10, pady=5)
            entry = ttk.Entry(br_tab, textvariable=var, width=50, font=('Segoe UI',12))
            entry.grid(row=idx, column=1, sticky='w', pady=5)
            breakout_vars[idx] = var

    def apply_and_run():
        if not (data_path and details_path and out_entry.get()):
            messagebox.showerror("Error", "Ensure all inputs are set.")
            return
        df = pd.read_csv(details_path, header=None, engine='python')
        for label, var in metadata_vars.items():
            rows = df.index[df.iloc[:,0]==label].tolist()
            if rows: df.iat[rows[0],1] = var.get()
        for ch, var in channel_vars.items():
            rows = df.index[df.iloc[:,0]==ch].tolist()
            if rows: df.iat[rows[0],3] = var.get()
        if breakout_df is not None:
            for idx, var in breakout_vars.items():
                row_i = 38 + idx
                if row_i < len(df): df.iat[row_i,1] = var.get()
        mod = Path(details_path).with_name(Path(details_path).stem + '_modified' + Path(details_path).suffix)
        df.to_csv(mod, index=False, header=False)
        cmd = ['python3', str(Path(__file__).parent / 'csv to dataframe.py'), data_path, str(mod), out_entry.get(), 'True']
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            section = metadata_vars['Test Section Number'].get()
            name = metadata_vars['Test Name'].get()
            pdf_name = f"{section}_{name}_{original_dt_str}.pdf"
            pdf_path = Path(out_entry.get()) / pdf_name
            messagebox.showinfo("Success", f"PDF created successfully:\n{pdf_path}")
            try: os.startfile(str(pdf_path))
            except: pass
        else:
            messagebox.showerror("Error", f"Script error:\n{res.stderr}")

    # Top frame
    top_frame = ttk.Frame(root, padding=15)
    top_frame.pack(fill='x')
    ttk.Button(top_frame, text="Select CSVs", command=select_two).grid(row=0, column=0, padx=5)
    sel_label = ttk.Label(top_frame, text="No files selected", wraplength=480, justify='left', font=('Segoe UI',12))
    sel_label.grid(row=0, column=1, sticky='w', padx=10)
    ttk.Label(top_frame, text="Output Directory:").grid(row=1, column=0, pady=5, sticky='e', padx=5)
    out_entry = ttk.Entry(top_frame, width=50, font=('Segoe UI',12))
    out_entry.grid(row=1, column=1, sticky='ew', padx=10)
    ttk.Button(top_frame, text="Browse", command=browse_dir).grid(row=1, column=2, padx=5)
    top_frame.columnconfigure(1, weight=1)

    # Notebook and tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    meta_tab = ttk.Frame(notebook)
    chan_tab = ttk.Frame(notebook)
    br_tab = ttk.Frame(notebook)
    notebook.add(meta_tab, text='Metadata')
    notebook.add(chan_tab, text='Channels')
    notebook.add(br_tab, text='Breakouts')

    # Bottom frame with only Apply & Run
    bottom_frame = ttk.Frame(root, padding=15)
    bottom_frame.pack(fill='x')
    ttk.Button(bottom_frame, text="Apply & Run", command=apply_and_run).pack(side='right', padx=5)

    root.mainloop()

if __name__ == '__main__':
    main_gui()