import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    filename = Path(r'C:\tmp\TeezersOSU\20230408\shift.xlsx')

    df = pd.read_excel(filename, sheet_name='traces')

    ref_z = df['1: z (um)'].values
    z_cols = [c for c in df.columns if 'z (um)' in c]

    for z in z_cols:
        plt.plot(df[z], label=z[0])

    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('z (um)')
    plt.show()