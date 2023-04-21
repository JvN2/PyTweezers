from modules.TweezerImageProcessing import Beads, Traces, get_roi, get_xyza

data = Traces()

files = {
         'piezo': r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\LUT Tiffs - 10-40 um\piezo.xlsx',
         'lut': r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\LUT Tiffs - 10-40 um\lut.avi',
         'data': r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\4-8-23_FEC Trial_Selected AVIs - 1-14.5 mm'
         }
for keys, values in files.items():
    data.set_glob(keys, values, 'files')

settings = {'pix_um': 0.225, 'n': 1.33, 'roi_pix' : 128}
for keys, values in settings.items():
    data.set_glob(keys, values, 'settings')

beads= Beads(data.get_glob('piezo'))

movie = beads.read_avi(data.get_glob('data') + r'\*.avi', 2)
coords = beads.pick_beads(movie[0])

for i, c in enumerate(coords):
    data.set_par(f'x0 (pix)', c[0], i)
    data.set_par(f'y0 (pix)', c[1], i)

data.to_file(r'\\data03\pi-vannoort\Noort\Data\TweezerOSU\20230408\test.xlsx')

