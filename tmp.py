import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from MT_steppers import to_profile


if __name__ == "__main__":
    gcodes = [
        "G93 S0.01",
        "G4 M500",
        "G1 Y10 F600",
        "G1 X-0.25",
        "G4 S10",
        "G1 X0 F10",
        "G4 S0.4",
        "G1 Y2.0 F600",
        "G4 S0.4",
        "G93",
    ]

    df = to_profile(gcodes)
    plt.plot(df)
    plt.show()

    # df2 = convert_to_section(df.loc[len(df)], -40, 10, t0=df1.index[-1])
    # plt.plot(
    #     df["x"],
    #     marker="o",
    #     fillstyle="none",
    # )
    # plt.show()
