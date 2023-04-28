from modules.TweezerImageProcessing import Beads, Traces
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def process_movie(filename, tracker):
    movie = tracker.read_avi(filename.with_suffix('.avi'))
    data = Traces(filename)

    if 'x0 (pix)' not in data.pars.columns:
        coords = tracker.pick_beads(movie[0], filename.with_suffix('.jpg'))
        for i, c in enumerate(coords):
            data.set_par('x0 (pix)', c[0], i)
            data.set_par('y0 (pix)', c[1], i)
        data.to_file()

    for i in tqdm(range(len(movie)), desc='Tracking'):
        beads = tracker.process_image(movie[i], data.pars[['x0 (pix)', 'y0 (pix)']].values)
        data.add_frame_coords(i, beads)

    for key, val in tracker.get_settings().items():
        data.set_glob(key, val, 'LUT')
    data.to_file()


if __name__ == '__main__':
    filename = Path(r'C:\tmp\TeezersOSU\20230408\shift.xlsx')
    tracker = Beads(filename.with_name('focus.xlsx'))

    process_movie(filename, tracker)

    data = Traces()
    data.from_file(filename)

    z_cols = [c for c in data.traces.columns if 'z (um)' in c]
    for z in z_cols:
        plt.plot(data.traces[z] - data.traces['1: z (um)'], label=z[0])

    plt.legend()
    plt.show()
