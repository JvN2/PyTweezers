from time import time
from functools import wraps
import cv2


def color_text(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def time_it(f):
    @wraps(f)
    def wrap(*args, **kw):
        color = [0, 100, 0]
        print(color_text(*color, f'Running: {f.__name__:25s}'), end='')
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(color_text(*color, f'-> {te - ts:9.3f} s'))
        return result

    return wrap


def show_image(image, title='Image'):
    if not cv2.getWindowProperty():
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    # if cv2.waitKey(5) & 0xFF == ord('q'):
    #     break
    return
