import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


def convert_to_section(x0, xe, v=30, t0=0, v0=0, ve=0, dt=0.01, a=10, vmax=30):

    v = min(v, vmax)
    v0 = min(v0, vmax)
    ve = min(ve, vmax)

    if np.sign(v0) != np.sign(ve):
        ve = 0
    v = np.sign(xe - x0) * abs(v)

    dt1 = abs((v - v0) / a)
    dx1 = v0 * dt1 + np.sign(v - v0) * 0.5 * a * dt1**2

    dt3 = abs((ve - v) / a)
    dx3 = ve * dt3 - np.sign(ve - v) * 0.5 * a * dt3**2

    dx2 = (xe - x0) - dx1 - dx3
    dt2 = abs(dx2 / v)

    if np.sign(dx2) != np.sign(xe - x0):
        # not enough time to get to full speed
        v = np.sqrt(a * np.abs(xe - x0) + 0.5 * (v0**2 + ve**2)) * np.sign(xe - x0)
        dt1 = abs((v - v0) / a)
        dx1 = v0 * dt1 + np.sign(v - v0) * 0.5 * a * dt1**2

        dt3 = abs((v - ve) / a)
        dx3 = v * dt3 - np.sign(ve - v) * 0.5 * a * dt3**2

        dx2 = 0
        dt2 = 0

    t1 = np.linspace(0, dt1, int(dt1 / dt) + 1)
    x1 = x0 + v0 * t1 + np.sign(v - v0) * 0.5 * a * t1**2
    t1 = t1 + t0

    t3 = np.linspace(0, dt3, int(dt3 / dt) + 1)
    x3 = x0 + dx1 + dx2 + v * t3 + np.sign(ve - v) * 0.5 * a * t3**2
    t3 = t3 + t0 + dt1 + dt2

    t = np.append(t1[1:], t3)
    x = np.append(x1[1:], x3)

    df = pd.DataFrame(np.asarray([t, x]).T, columns=["t", "x"])
    df.set_index("t", inplace=True)
    return df


def convert_to_profile(gcodes, a=8, axes=["X", "Y", "Z", "A", "B"]):

    df = pd.DataFrame(columns=["t"] + axes + [f"v_{axis}" for axis in axes])
    dt = None

    for i, line in enumerate(gcodes):
        gcode = line.split()
        if gcode[0] == "G93":
            df.loc[0] = np.zeros(len(df.columns))
            dt = float(gcode[1][1:])
        elif gcode[0] == "G1" and dt:
            axis = gcode[1][0]
            position = float(gcode[1][1:])
            df.loc[len(df), axis] = position
            try:
                velocity = float(gcode[2][1:])
            except IndexError:
                velocity = 1000
            df.loc[len(df) - 1, "v_" + axis] = velocity
        elif gcode[0] == "G4" and dt:
            values = df.iloc[-1].values.copy()
            values[-len(axes) // 2 :] = np.nan
            if gcode[1][0] == "S":
                values[0] += float(gcode[1][1:])
            elif gcode[1][0] == "M":
                values[0] += float(gcode[1][1:]) / 1000
            df.loc[len(df)] = values

    # Replace NaN values in columns that start with 'v_' with 0
    df.loc[:, df.columns.str.startswith("v_")] = df.loc[
        :, df.columns.str.startswith("v_")
    ].fillna(0)

    # df.set_index("t", inplace=True)
    print(df)
    return df


if __name__ == "__main__":
    gcodes = [
        "G93 S0.01",
        "G4 M500",
        "G1 Y0.1 F30",
        "G1 X0.25",
        "G4 S1",
        # "G1 X0 F10",
        # "G4 S0.4",
        # "G1 Y0.0 F10",
        # "G4 S0.4",
        "G94",
    ]

    convert_to_profile(gcodes)
    # plt.plot(
    #     convert_to_section(20, -40, 10, t0=100),
    #     marker="o",
    #     fillstyle="none",
    # )
    # plt.show()
