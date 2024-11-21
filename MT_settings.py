import tkinter as tk
from tkinter import messagebox
import numpy as np


class SettingsEditor(tk.Toplevel):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.entries = {}
        self.sliders = {}
        self.create_widgets()

    def create_widgets(self):
        row = 0
        button_width = 10  # Set a fixed width for the buttons
        for key, value in self.settings.items():
            if isinstance(value, (int, float)):
                tk.Label(self, text=key).grid(row=row, column=0, padx=5, pady=5)
                entry = tk.Entry(self)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=5)
                self.entries[key] = entry

                # Define min, max, default values, and resolution for the slider
                min_val, max_val, default_val, resolution, is_log = (
                    self.get_slider_values(key, value)
                )
                slider = tk.Scale(
                    self,
                    from_=min_val,
                    to=max_val,
                    orient=tk.HORIZONTAL,
                    resolution=resolution,
                    showvalue=0,
                )
                slider.set(self.to_slider_value(default_val, min_val, max_val, is_log))
                slider.grid(row=row, column=2, padx=5, pady=5)
                self.sliders[key] = (slider, is_log, min_val, max_val)

                # Couple the entry and slider
                entry.bind(
                    "<KeyRelease>",
                    lambda event, k=key, e=entry, s=slider, is_log=is_log, min_val=min_val, max_val=max_val: self.update_slider_from_entry(
                        k, e, s, is_log, min_val, max_val
                    ),
                )
                slider.bind(
                    "<Motion>",
                    lambda event, k=key, e=entry, s=slider, is_log=is_log, min_val=min_val, max_val=max_val: self.update_entry_from_slider(
                        k, e, s, is_log, min_val, max_val
                    ),
                )

                row += 1

        tk.Button(self, text="OK", command=self.on_ok, width=button_width).grid(
            row=row, column=2, padx=5, pady=5
        )
        tk.Button(self, text="Cancel", command=self.on_cancel, width=button_width).grid(
            row=row, column=0, padx=5, pady=5
        )

    def get_slider_values(self, key, value):
        # Define min, max, default values, resolution, and whether to use logarithmic scale
        if key == "roi_size (pix)":
            return 10, 100, value, 1, False
        elif key == "frames":
            return 1, 1000, value, 1, False
        elif key == "window (pix)":
            return 100, 2000, value, 10, False
        elif key == "fov_size (pix)":
            return 100, 2000, value, 10, False
        elif key == "some_float":
            return 0.1, 100.0, value, 0.01, True  # Example of a logarithmic scale
        else:
            return 0, 1000, value, 1, False

    def to_slider_value(self, value, min_val, max_val, is_log):
        if is_log:
            return (
                np.log10(value / min_val)
                / np.log10(max_val / min_val)
                * (max_val - min_val)
                + min_val
            )
        return value

    def from_slider_value(self, slider_value, min_val, max_val, is_log):
        if is_log:
            return min_val * 10 ** (
                (slider_value - min_val)
                / (max_val - min_val)
                * np.log10(max_val / min_val)
            )
        return slider_value

    def update_slider_from_entry(self, key, entry, slider, is_log, min_val, max_val):
        try:
            value = float(entry.get())
            slider.set(self.to_slider_value(value, min_val, max_val, is_log))
        except ValueError:
            pass

    def update_entry_from_slider(self, key, entry, slider, is_log, min_val, max_val):
        value = self.from_slider_value(slider.get(), min_val, max_val, is_log)
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def on_ok(self):
        for key, entry in self.entries.items():
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
            slider, is_log, min_val, max_val = self.sliders[key]
            slider_value = self.from_slider_value(
                slider.get(), min_val, max_val, is_log
            )
            entry.delete(0, tk.END)
            entry.insert(0, str(slider_value))
            self.settings[key] = slider_value

        self.destroy()

    def on_cancel(self):
        self.destroy()


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    settings = {
        "roi_size (pix)": 50,
        "frames": 200,
        "window (pix)": 800,
        "fov_size (pix)": 400,
        "some_float": 1.23,
    }
    editor = SettingsEditor(root, settings)
    root.mainloop()
