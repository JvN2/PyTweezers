import pandas as pd
from pathlib import Path
from modules.TweezerImageProcessing import Traces
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def convert_pix_um(data, pix_um):
    columns = list(data.traces.columns)
    for i, c in enumerate(columns):
        if 'pix' in c:
            data.traces[c] *= pix_um
            columns[i] = c.replace('pix', 'um')
    data.traces.columns = columns
    data.set_glob('pix_um', pix_um, 'calibration')
    return data

def fit_exp_decay(x,y):
    def func(x, a, b):
        return a * np.exp(- x/b)

    popt, pcov = curve_fit(func, x, y)

    return (func(x, *popt)), popt


def compute_force(data, ref_bead, beads=None, show=False):
    kT = 0.0041  # pN um
    shifts = set(data.traces['Shift (mm)'])
    if beads is None:
        beads = set(data.pars.index)
        beads.remove(ref_bead)

    p_mean = np.zeros(2)
    for axis in ['x']:
        for b in beads:
            forces = np.empty_like(data.traces.index)
            for s in shifts:
                selection = data.traces[data.traces['Shift (mm)'] == s]
                mean_z = np.mean(selection[f'{b}: z (um)'] - selection[f'{ref_bead}: z (um)'])
                if axis == 'y':
                    mean_z += 1.4
                variance_x = np.var(selection[f'{b}: {axis} (um)'])
                forces[data.traces['Shift (mm)'] == s] = kT * mean_z / variance_x

            data.traces[f'{b}: F_{axis} (pN)'] = forces

            fit, p = fit_exp_decay(data.traces['Shift (mm)'], forces)
            p_mean += p

            if show:
                plt.scatter(data.traces['Shift (mm)'], forces, label=f'{b}: F(pN) ={p[0]:.0f} exp(-Shift (mm) / {p[1]:.2f} )')
                plt.plot(data.traces['Shift (mm)'], fit)
    p_mean /= len(beads)
    data.traces[f'F (pN)'] =  p_mean[0]* np.exp(-data.traces['Shift (mm)']/ p_mean[1])
    data.set_glob('f0 (pN)', p_mean[0], 'calibration')
    data.set_glob('shift0 (mm)', p_mean[1], 'calibration')

    if show:
        x = np.linspace(0, 15, 100)
        y = p_mean[0] * np.exp(-x / p_mean[1])
        plt.plot(x, y, color='k', label=f'mean: F(pN) ={p_mean[0]:.0f} exp(-Shift (mm) / {p_mean[1]:.2f}')

        plt.legend()
        plt.semilogy()
        plt.xlabel('Shift (mm)')
        plt.ylabel('F (pN)')
        plt.ylim((0.1, 150))
        plt.xlim((0, 15))
        plt.title(r'$F = \frac{k_BT<z>}{<x^2>}$')
        plt.show()

    return data

def wlc(f, L, p = 50):
    kT = 4.1 # pN nm
    return 0.001 * 0.34 * L *(1 - np.sqrt(kT/(4 * f* p)) )

if __name__ == '__main__':
    filename = Path(r'C:\tmp\TeezersOSU\20230408\shift.xlsx')

    pix_um = 4.64 / 40
    L = 6600 # bp

    beads = [0, 2, 3]
    ref_bead = 1

    data = Traces()
    data.from_file(filename)
    data = convert_pix_um(data, pix_um)
    data = compute_force(data, ref_bead = ref_bead, beads = beads, show = True)

    for b in beads:
        z = data.traces[f'{b}: z (um)']-data.traces[f'{ref_bead}: z (um)']
        # z/= np.max(z)
        plt.scatter(z , data.traces['F (pN)'], label = b)
    f = np.logspace(-2,2, 100)
    z_wlc = wlc(f, L)
    # z_wlc /= np.max(z_wlc)
    plt.plot(z_wlc, f, color = 'k', label = 'WLC')
    plt.legend()
    plt.xlabel('z (um)')
    plt.ylabel('F (pN)')
    plt.xlim(0, 1.1* np.max(z_wlc))
    # plt.semilogy()
    plt.show()

    data.to_file()


