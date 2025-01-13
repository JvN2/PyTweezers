import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for combobox
from tkinter import filedialog  # Import filedialog for file selection
import numpy as np
from pathlib import Path
from icecream import ic


class SettingsEditor(tk.Toplevel):
    def __init__(self, parent, parameters, title=None, settings=None):
        super().__init__(parent)
        self.parameters = parameters
        self.start_paramters = parameters.copy()
        if settings is not None:
            self.old_settings = settings.to_dict()
        self.settings = settings
        self.entries = {}
        self.sliders = {}

        self.title(title)
        self.transient(parent)  # Set to be on of the main window
        self.grab_set()  # Make the window modal
        self.create_widgets()
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        row = 0
        button_width = 10  # Set a fixed width for the buttons
        entry_width = 20  # Set a fixed width for the entries and comboboxes
        for key, values in self.parameters.items():
            if isinstance(values[0], (int, float)):
                tk.Label(self, text=key).grid(row=row, column=0, padx=5, pady=5)
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
                entry.bind(
                    "<FocusOut>",
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
                tk.Label(self, text=key).grid(row=row, column=0, padx=5, pady=5)
                var = tk.StringVar(self)
                var.set(values[0])
                combobox = ttk.Combobox(
                    self, textvariable=var, values=values[1:], width=entry_width - 3
                )  # Adjust width to match entry
                combobox.grid(
                    row=row, column=1, columnspan=2, padx=5, pady=5, sticky="w"
                )
                self.entries[key] = var
                row += 1
                combobox.bind("<ButtonRelease>", self.update_settings())
            elif isinstance(values[0], Path):
                tk.Label(self, text=key).grid(row=row, column=0, padx=5, pady=5)
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

        tk.Button(self, text="OK", command=self.on_ok, width=button_width).grid(
            row=row, column=2, padx=15, pady=5
        )
        tk.Button(self, text="Cancel", command=self.on_cancel, width=button_width).grid(
            row=row, column=0, padx=15, pady=5
        )

    def update_parameters(self):
        for key, entry in self.entries.items():
            if isinstance(entry, tk.StringVar):
                self.parameters[key] = entry.get()
            else:
                value = entry.get()
                if value.isdigit():
                    self.parameters[key] = int(value)
                else:
                    try:
                        self.parameters[key] = float(value)
                    except ValueError:
                        messagebox.showerror(
                            "Invalid input", f"Invalid value for {key}: {value}"
                        )
                        self.parameters[key] = self.start_paramters[key]

                # Update the entry with the slider value
                slider, par_type, min_val, max_val = self.sliders[key]
                slider_value = self.from_slider_value(
                    slider.get(), min_val, max_val, par_type
                )
                entry.delete(0, tk.END)
                entry.insert(0, str(slider_value))
                self.parameters[key] = slider_value

            if self.settings and "channel" in self.parameters.keys():
                self.settings._plot_channel = self.parameters["channel"]
            if self.settings and "range" in self.parameters.keys():
                self.settings._plot_range = self.parameters["range"]
            if self.settings and "offset" in self.parameters.keys():
                self.settings._plot_offset = self.parameters["offset"]
            if self.settings and "subtract mean" in self.parameters.keys():
                self.settings._plot_subtract_mean = self.parameters["subtract mean"]

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
        return np.clip(value, min_val, max_val)

    def from_slider_value(self, slider_value, min_val, max_val, par_type):
        def round_to_significant_digits(value, digits):
            return float(f"{value:.{digits}g}")

        if par_type == "2log":
            return round_to_significant_digits(2**slider_value, 3)
        elif par_type == "10log":
            return round_to_significant_digits(10**slider_value, 3)
        return slider_value

    def update_slider_from_entry(self, key, entry, slider, par_type, min_val, max_val):
        try:
            value = float(entry.get())
            slider.set(self.to_slider_value(value, min_val, max_val, par_type))
        except ValueError:
            pass
        self.update_parameters()
        self.update_settings()

    def update_entry_from_slider(self, key, entry, slider, par_type, min_val, max_val):
        value = self.from_slider_value(slider.get(), min_val, max_val, par_type)
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        self.update_parameters()
        self.update_settings()

    def update_settings(self):
        if self.settings:
            for key, value in self.parameters.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

    def on_ok(self):
        # Return the new settings to the main function
        self.update_parameters()
        self.update_settings()
        self.destroy()

    def on_cancel(self):
        self.destroy()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    settings = Settings()  # Assuming you have a Settings class
    app = SettingsEditor(master=root, parameters=settings.to_dict(), settings=settings)
    app.mainloop()
