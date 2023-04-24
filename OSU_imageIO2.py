from modules.TweezerImageProcessing import Beads, Traces
from tqdm import tqdm

new_files = True
if new_files:

    data = Traces()  # create Traces object

    files = {
        'directory': r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408',
        'piezo': r'LUT Tiffs - 10-40 um\piezo.xlsx',
        'lut': r'LUT Tiffs - 10-40 um\lut.avi',
        # 'movie': r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\4-8-23_FEC Trial_Selected AVIs - 1-14.5 mm'
        'movie': r'combined.avi'
    }  # define  data files/folders
    for keys, values in files.items():
        data.set_glob(keys, values, 'files')

    settings = {'pix_um': 0.225, 'n': 1.33, 'roi_pix': 128}  # define imaging settings
    for keys, values in settings.items():
        data.set_glob(keys, values, 'settings')

    beads = Beads(data.get_glob('piezo', 'files'))  # create Beads object and select beads
    movie = beads.read_avi(data.get_glob('movie', 'files'), 2)
    coords = beads.pick_beads(movie[0],data.get_glob('movie', 'files').replace('.avi', '.jpg'))
    for i, c in enumerate(coords):
        data.set_par(f'x0 (pix)', c[0], i)
        data.set_par(f'y0 (pix)', c[1], i)

    data.to_file(data.get_glob('directory') + r'\\test2.xlsx')

data = Traces(r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\test2.xlsx')

print(data.get_glob('movie'))

tracker = Beads(data.get_glob('piezo', 'files'))

movie = tracker.read_avi(data.get_glob('movie', 'files'))

# tracker.save_avi(movie, data.get_glob('movie')+r'\combined.avi')

coords = [(data.get_par(f'x0 (pix)', i), data.get_par(f'y0 (pix)', i)) for i in range(len(data.pars))]
for i in tqdm(range(len(movie)), desc='Tracking'):
    beads = tracker.process_image(movie[i], coords)
    data.add_frame_coords(i, beads)

data.add_from_excel(data.get_glob('directory') + r'\shift.xlsx', 'Shift (mm)')

print(data.traces.head(4))

data.to_file()
