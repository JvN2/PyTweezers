# Inspired by SuperFastPython.com -> example of one producer and multiple consumers with threads
# and https://www.askpython.com/python/producer-consumer-problem for buffering
from pathlib import Path
from queue import Queue
from threading import Thread
from time import sleep
from time import time

import numpy as np
import pandas as pd
from cv2 import imread
from pylablib.devices import IMAQ
from tqdm import tqdm

from modules.TweezerImageProcessing import get_roi, get_xyza, Beads, Traces
from modules.TweezersSupport import time_it, color_text

ref_filename = Path(r'data\data_024.tdms')
tracker = Beads(ref_filename)

print(tracker.get_processing_settings())