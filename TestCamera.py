import tkinter as tk
from tkinter import messagebox
from AlliedVision import CameraApplication
import DummyCamera
import threading
from time import sleep
import cv2


class MainApp:
    def __init__(self, root):

        self.settings = {"roi_size": 50, "rois": [(50, 50), (100, 200), (198, 350)]}
        self.root = root
        self.root.title("Camera Control") 

        self.dummy_camera = DummyCamera.CameraApplication(dummy=True)

        menubar = tk.Menu(root)
        root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Start", command=self.start_camera)
        file_menu.add_command(label="Stop", command=self.stop_camera)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_application)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Test", command=self.test)


    def start_camera(self):
        # self.camera.start(self.settings)
        self.dummy_camera.start(self.settings)

    def stop_camera(self):
        self.dummy_camera.stop()
        self.camera.stop()

    def exit_application(self):
        self.stop_camera()
        self.dummy_camera.stop()
        self.root.quit()

    def show_about(self):
        messagebox.showinfo("About", "Camera Control Application")

    def test(self):
        self.camera = CameraApplication()
        threading.Thread(target=self.camera.run).start()
        # sleep(2)
        # self.camera.stop()




if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
