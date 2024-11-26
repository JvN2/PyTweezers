import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for combobox
from tkinter import filedialog  # Import filedialog for file selection
import numpy as np
from pathlib import Path


class SettingsEditor(tk.Toplevel):
    def __init__(self, parent, settings, title=None):
        super().__init__(parent)
        self.settings = settings
        self.entries = {}
        self.sliders = {}
        if title:
            self.title(title)  # Set the title of the window
        self.create_widgets()

    def create_widgets(self):
        row = 0
        button_width = 10  # Set a fixed width for the buttons
        entry_width = 20  # Set a fixed width for the entries and comboboxes
        for key, values in self.settings.items():
            if isinstance(values[0], (int, float)):
                tk.Label(self, text=key).grid(
                    row=row, column=0, padx=5, pady=5)
                entry = tk.Entry(self, width=entry_width)
                entry.insert(0, str(values[0]))
                entry.grid(row=row, column=1, padx=5, pady=5)
                self.entries[key] = entry

                # Define min, max, default values, and resolution for the slider
                min_val, max_val, resolution, par_type = values[1:5]

                slider = tk.Scale(
                    self,
                    from_=min_val,
                    to=max_val,
                    orient=tk.HORIZONTAL,
                    resolution=resolution,
                    showvalue=0,
                )

                if par_type == "linear":
                    slider_value = values[0]
                elif par_type == "2log":
                    slider_value = np.log2(values[0])
                elif par_type == "10log":
                    slider_value = np.log10(values[0])
                slider.set(slider_value)
                slider.grid(row=row, column=2, padx=5, pady=5)
                self.sliders[key] = (slider, par_type, min_val, max_val)

                # Couple the entry and slider
                entry.bind(
                    "<KeyRelease>",
                    lambda event, k=key, e=entry, s=slider, par_type=par_type, min_val=min_val, max_val=max_val: self.update_slider_from_entry(
                        k, e, s, par_type, min_val, max_val
                    ),
                )
                slider.bind(
                    "<Motion>",
                    lambda event, k=key, e=entry, s=slider, par_type=par_type, min_val=min_val, max_val=max_val: self.update_entry_from_slider(
                        k, e, s, par_type, min_val, max_val
                    ),
                )
                row += 1
            elif isinstance(values[0], str):
                tk.Label(self, text=key).grid(
                    row=row, column=0, padx=5, pady=5)
                var = tk.StringVar(self)
                var.set(values[0])
                combobox = ttk.Combobox(
                    self, textvariable=var, values=values, width=entry_width - 3
                )  # Adjust width to match entry
                combobox.grid(
                    row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w"
                )
                self.entries[key] = var
                row += 1
            elif isinstance(values[0], Path):
                tk.Label(self, text=key).grid(
                    row=row, column=0, padx=5, pady=5)
                var = tk.StringVar(self)
                var.set(values[0])
                combobox = ttk.Combobox(
                    self, textvariable=var, values=values[1:], width=entry_width - 3
                )  # Adjust width to match entry
                combobox.grid(
                    row=row, column=1, columnspan=2, padx=5, pady=5, sticky="we"
                )
                combobox.bind("<<ComboboxSelected>>", self.open_file_dialog)
                self.entries[key] = var
                row += 1
                print(values[0])

        tk.Button(self, text="OK", command=self.on_ok, width=button_width).grid(
            row=row, column=2, padx=15, pady=5
        )
        tk.Button(self, text="Cancel", command=self.on_cancel, width=button_width).grid(
            row=row, column=0, padx=15, pady=5
        )

    def open_file_dialog(self, event):
        selected_extension = event.widget.get()
        file_path = filedialog.askopenfilename(
            filetypes=[(selected_extension, selected_extension)]
        )
        if file_path:
            event.widget.set(file_path)

    def to_slider_value(self, value, min_val, max_val, par_type):
        if par_type == "2log":
            return np.log2(value)
        elif par_type == "10log":
            return np.log10(value)
        return value

    def from_slider_value(self, slider_value, min_val, max_val, par_type):
        if par_type == "2log":
            return 2**slider_value
        elif par_type == "10log":
            return 10**slider_value
        return slider_value

    def update_slider_from_entry(self, key, entry, slider, par_type, min_val, max_val):
        try:
            value = float(entry.get())
            slider.set(self.to_slider_value(value, min_val, max_val, par_type))
        except ValueError:
            pass

    def update_entry_from_slider(self, key, entry, slider, par_type, min_val, max_val):
        value = self.from_slider_value(
            slider.get(), min_val, max_val, par_type)
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def on_ok(self):
        for key, entry in self.entries.items():
            if isinstance(entry, tk.StringVar):
                self.settings[key] = entry.get()
            else:
                value = entry.get()
                if value.isdigit():
                    self.settings[key] = int(value)
                else:
                    try:
                        self.settings[key] = float(value)
                    except ValueError:
                        messagebox.showerror(
                            "Invalid input", f"Invalid value for {key}: {value}"
                        )
                        return

                # Update the entry with the slider value
                slider, par_type, min_val, max_val = self.sliders[key]
                slider_value = self.from_slider_value(
                    slider.get(), min_val, max_val, par_type
                )
                entry.delete(0, tk.END)
                entry.insert(0, str(slider_value))
                self.settings[key] = slider_value

        self.master.settings = (
            self.settings
        )  # Return the new settings to the main function
        self.destroy()

    def on_cancel(self):
        self.destroy()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.settings = {}  # Initialize root.settings
    settings = {
        "roi_size (pix)": (64, 2, 8, 1, "2log"),
        "frames": (100, 0, 4, 1, "10log"),
        "window (pix)": (1024, 7, 10, 1, "2log"),
        "fov_size (pix)": (50, 0, 100, 10, "Linear"),
        "some_float": (50, 0, 3, 1, "10log"),
        "some string": ("hello", "10log", "else"),
        "File": (Path("c:/tmp/image.bin"), "*.bin", "*.hdf"),
    }

    settings = {
        "axis": ("X (mm)", "Y (mm)", "Z (mm)"),
        "target": (1, 0, 10, 0.1, "linear"),
    }
    print(settings)

    title = "Adjust settings ..."
    editor = SettingsEditor(root, settings, title)
    root.wait_window(editor)  # Wait for the settings editor to close
    for key, value in root.settings.items():
        print(f"{key}: {value}")  # Print the updated settings
    root.mainloop()
    # for key, value in root.settings.items():
    #     print(f"{key}: {value}")  # Print the updated settings
