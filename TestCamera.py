import tkinter as tk
from tkinter import messagebox
import threading
from multithreading_opencv import CameraApplication

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Control")
        
        self.camera_app = CameraApplication()
        self.camera_thread = None
        
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        file_menu.add_command(label="Start", command=self.start_camera)
        file_menu.add_command(label="Stop", command=self.stop_camera)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_application)
        
    def start_camera(self):
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.camera_thread = threading.Thread(target=self.run_camera)
            self.camera_thread.start()
            # messagebox.showinfo("Info", "Camera started.")
        else:
            print("Camera is already running.")
            messagebox.showwarning("Warning", "Camera is already running.")
    
    def stop_camera(self):
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_app.stop()  # Call the stop method of the Application class
            self.camera_thread.join()
            self.camera_thread = None
            # messagebox.showinfo("Info", "Camera stopped.")


        else:
            print("Camera is not running.")
            messagebox.showwarning("Warning", "Camera is not running.")
    
    def exit_application(self):
        if self.camera_thread and self.camera_thread.is_alive():
            print("Exiting application...")
            self.camera_app.stop()
            self.camera_thread.join()
        self.root.quit()
    
    def run_camera(self):
        self.camera_app.run()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()