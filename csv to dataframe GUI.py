import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import subprocess

def run_existing_script(file_path1, file_path2, file_path3, is_gui):
    """
    Run the existing script with the selected file paths.
    """
    try:
        # Construct the command to call your script
        script_command = [
            "python3",  # Call Python interpreter
            "V:/Userdoc/Mechatronics/Applications/Python/R&D AutoChart Generation/csv to dataframe.py",
            file_path1,
            file_path2,
            file_path3,
            str(is_gui)
        ]

        # Run the script as a subprocess
        result = subprocess.run(script_command, capture_output=True, text=True)

        # Display the output or errors
        if result.returncode == 0:
            messagebox.showinfo("Success", f"Script executed successfully.\n\nOutput:\n{result.stdout}")
        else:
            messagebox.showerror("Error", f"Script execution failed:\n\nError:\n{result.stderr}\n\nOutput:\n{result.stdout}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")


def select_file(entry_field):
    """
    Open a file dialog to select a file and populate the entry field.
    """
    file_path = filedialog.askopenfilename()
    if file_path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, file_path)


def select_directory(entry_field):
    """
    Open a file dialog to select a directory and populate the entry field.
    """
    dir_path = filedialog.askdirectory()
    if dir_path:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, dir_path)


def main_gui():
    """
    Create the main GUI for selecting file paths and running the script.
    """
    # Initialize tkinter
    root = tk.Tk()
    root.title("File Path Selector for Script")

    # Create labels and entry fields for file paths
    labels = [
        "Select Test Data (.csv):",
        "Select Test Details (.csv):",
        "Select PDF Output Directory:"
    ]
    entry_fields = []

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
        entry = tk.Entry(root, width=50)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entry_fields.append(entry)
        if i < 2:  # For file selection
            tk.Button(
                root,
                text="Browse",
                command=lambda e=entry: select_file(e)
            ).grid(row=i, column=2, padx=5, pady=5)
        else:  # For directory selection
            tk.Button(
                root,
                text="Browse",
                command=lambda e=entry: select_directory(e)
            ).grid(row=i, column=2, padx=5, pady=5)

    # Run script button
    def run_script():
        file_path1 = entry_fields[0].get()
        file_path2 = entry_fields[1].get()
        file_path3 = entry_fields[2].get()

        if not file_path1 or not file_path2 or not file_path3:
            messagebox.showerror("Error", "Please provide all file paths.")
            return

        run_existing_script(file_path1, file_path2, file_path3, is_gui=True)

    tk.Button(root, text="Run Script", command=run_script).grid(
        row=3, column=1, pady=20
    )

    # Run the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main_gui()