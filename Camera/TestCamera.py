import vmbpy
import queue
import numpy as np
import time

import multithreading_opencv as cam


def list_cameras():
    vmb = vmbpy.VmbSystem.get_instance()
    print("Available cameras:")
    with vmb:
        cams = vmb.get_all_cameras()
        for cam in cams:
            print("+", cam)


if __name__ == "__main__":
    list_cameras()
    camera = cam.MainThread(filename="test.avi")
    camera.start()
    time.sleep(5)
    camera.stop()
    camera.join()
