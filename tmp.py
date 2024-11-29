import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_to_section(x0, v0, xe, ve, v, t0, dt, a=10):
    dt1 = (v - v0) / a
    dx1 = v0 * dt1 + 0.5 * a * dt1 ** 2

    dt3 = abs((ve - v) / a)
    dx3 = ve * dt3 - 0.5 * a * dt3 ** 2

    dx2 = (xe - x0) - (dx1 - dx3)
    dt2 = abs(dx2 / v)

    df = pd.DataFrame(columns=['t', 'x'])
    df.loc[0] = [t0, x0]
    while df.iloc[-1]['t'] < t0 + dt1 + dt2 + dt3: # + dt2 + dt3 or df.iloc[-1]['x'] < xe:
        t_new = df.iloc[-1]['t'] + dt
        if df.iloc[-1]['t'] < dt1 + t0:
            df.loc[len(df)] = [
                t_new,
                x0 +
                v0 * t_new +
                0.5 * a * t_new ** 2
            ]
        elif df.iloc[-1]['t'] < dt1 + dt2 + t0:
            df.loc[len(df)] = [
                t_new,
                x0 + dx1 + v * (t_new - dt1)
            ]
        else:
            df.loc[len(df)] = [
                t_new,
                x0 + dx1 + dx2 + v * (t_new - dt1 - dt2) -
                0.5 * a * (t_new - dt1 -dt2) ** 2
            ]
    df.set_index('t', inplace=True)
    print(df)
    return df


def convert_to_profile(gcodes, a=8, axes=['X', 'Y', 'Z', 'A', 'B']):

    df = pd.DataFrame(columns=['t'] + axes + [f'v_{axis}' for axis in axes])
    dt = None

    for i, line in enumerate(gcodes):
        gcode = line.split()
        if gcode[0] == 'G93':
            df.loc[0] = np.zeros(len(df.columns))
            dt = float(gcode[1][1:])
        # elif gcode[0] == 'G1' and dt:
        #     axis = gcode[1][0]
        #     position = float(gcode[1][1:])
        #     velocity = float(gcode[2][1:])
        #     df.loc[len(df), axis] = position
        #     df.loc[len(df)-1, 'v_' + axis] = velocity
        elif gcode[0] == 'G4' and dt:
            values = df.iloc[-1].values.copy()
            if gcode[1][0] == 'S':
                values[0] += float(gcode[1][1:])
            elif gcode[1][0] == 'M':
                values[0] += float(gcode[1][1:]) / 1000
            df.loc[len(df)] = values

    # Replace NaN values in columns that start with 'v_' with 0
    df.loc[:, df.columns.str.startswith(
        'v_')] = df.loc[:, df.columns.str.startswith('v_')].fillna(0)

    df.set_index('t', inplace=True)
    print(df)
    return df


if __name__ == "__main__":
    gcodes = [
        "G93 S0.01",
        "G4 M500",
        "G1 Y0.1 F30",
        "G1 X0.25 F30",
        "G4 S1",
        # "G1 X0 F10",
        # "G4 S0.4",
        # "G1 Y0.0 F10",
        # "G4 S0.4",
        "G94"
    ]

    # convert_to_profile(gcodes)
    plt.plot(convert_to_section(0, 0, 2, 0, 120/60, 0, 0.01))
    plt.show()
