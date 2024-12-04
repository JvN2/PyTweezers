import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.optimize import curve_fit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.signal import medfilt
import glob
import os
import itertools as it
import pandas as pd
from pathlib import Path

import TraceIO as tio

# specify constants
kT = 4.114  # Thermal energy [pN*nm]
nm_bp = 0.34  # length dsDNA [nm/bp]
nm_base = 0.58  # length ssDNA = [nm/base];
omega0 = 2 * np.pi / 3.6  # inverted pitch of dsDNA [nm-1]

L_bp = 4753
Lds_bp = L_bp
bound_fit = []
slope = 0.0
double = False
overstretch_params = 0
shift_max = 15.0


# functions


def bilinear_interpolation(x, y, points):
    """Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)  # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError("points do not form a rectangle")
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError("(x, y) not within the rectangle")

    return (
        q11 * (x2 - x) * (y2 - y)
        + q21 * (x - x1) * (y2 - y)
        + q12 * (x2 - x) * (y - y1)
        + q22 * (x - x1) * (y - y1)
    ) / ((x2 - x1) * (y2 - y1) + 0.0)


def intersection(curve1, curve2, xrange=[-np.inf, np.inf], show=False):
    """Find the intercept of two curves, given by the different x data"""
    curve1 = np.sort(curve1)
    curve2 = np.sort(curve2)
    xrange = [
        np.max([np.min(curve1[0]), np.min(curve2[0]), xrange[0]]),
        np.min([np.max(curve1[0]), np.max(curve2[0]), xrange[1]]),
    ]
    x = np.linspace(xrange[0], xrange[1], 100)
    ny1 = np.interp(x, curve1[0], curve1[1])
    ny2 = np.interp(x, curve2[0], curve2[1])
    diff = ny2 - ny1
    if diff[0] > 0:  # Make sure diff is rising
        diff *= -1
    i = np.argmax(diff > 0)
    if diff[i] > 0:  # Make sure that the first point is below 0
        i -= 1
    xi = x[i] - diff[i] * ((x[i + 1] - x[i]) / (diff[i + 1] - diff[i]))
    yi = np.interp(xi, x, ny1)

    if show:
        plt.plot(curve1[0], curve1[1])
        plt.plot(curve2[0], curve2[1])
        plt.plot(x, ny1)
        plt.plot(x, ny2)
        plt.plot([xi], [yi], "o")
        plt.show()
    return xi, yi


def median_filter(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def gap_filter(x_array, dt_min):
    start = 0
    for i, x in enumerate(x_array):
        if x > 0:
            if i - start <= dt_min:
                x_array[start:i] = 1
            start = i
    return x_array


def get_ruptures(data, median_filter_width=15, dL_min=200, dt_min=5):
    L = tio.h5_read_trace(data, "L (bp)")
    f = tio.h5_read_trace(data, "F (pN)")
    z = tio.h5_read_trace(data, "Z (um)")
    z -= tio.h5_read_par(data, "Zbead (um)")[0]
    z -= tio.h5_read_par(data, "Z0 (um)")[0]
    x = tio.h5_read_trace(data, "X (um)")
    y = tio.h5_read_trace(data, "Y (um)")

    selection = tio.h5_read_trace(data, "Selection").astype(bool)

    dZ = np.diff(z, prepend=z[0])
    dL = np.diff(median_filter(L, median_filter_width), prepend=L[0])
    dR = np.diff(median_filter(x, median_filter_width), prepend=x[0]) ** 2
    dR += np.diff(median_filter(y, median_filter_width), prepend=y[0]) ** 2
    dR = np.sqrt(dR)

    ruptures = gap_filter(dL > dL_min, dt_min) * selection
    pos = np.cumsum((np.diff(ruptures, prepend=1) == 1))
    pos *= ruptures
    tio.h5_write_trace(ruptures, data, "Selection")

    result = []
    for i in np.unique(pos)[1:]:
        sel = pos == i
        f_rupt = np.sum(sel * f) / np.sum(sel)
        dR_rupt = np.sum(sel * dR)
        dL_rupt = np.sum(sel * dL)
        dZ_rupt = np.sum(sel * dZ)

        path = Path(filename)
        name = rf"{path.parts[-2]}/{path.parts[-1]}"
        name = name.split(".")[0].replace(" ", "_")

        result.append([name, label, f_rupt, dR_rupt, dZ_rupt, dL_rupt])
    df_result = pd.DataFrame(
        result, columns=["filename", "label", "F (pN)", "dR (um)", "dZ (um)", "dL (bp)"]
    )
    return df_result


def MagnetForce(x, L1=1.4, L2=0.8, f0=0.02, fmax=85.0, alpha=0.7, dt=0):
    """
    Calculate the force from the magnet position.
    :param x: Shift (mm) #
    :param L1: L1 (mm) #
    :param L2: L2 (mm) #
    :param f0: Fmin (pN) #
    :param fmax: Fmax (pN) #
    :param alpha: alpha (-) #
    :param dt: offset (pts) #
    :return: F (pN) #
    """
    force_pN = fmax * (alpha * np.exp(-x / L1) + (1 - alpha) * np.exp(-x / L2)) + f0

    # resample force to correct for time shift
    if dt != 0:
        force_pN = np.interp(
            np.arange(len(force_pN)) - dt, np.arange(len(force_pN)), force_pN
        )

    return force_pN


def zMagnetForce(x, L1=1.4, L2=0.8, f0=0.02, fmax=85.0, alpha=0.7, L=1, dt=0, z_bead=0):
    """
    Calculate the force from the magnet position.
    :param x: Shift (mm) #
    :param L1: L1 (mm) #
    :param L2: L2 (mm) #
    :param f0: Fmin (pN) #
    :param fmax: Fmax (pN) #
    :param alpha: alpha (-) #
    :param L: L (bp) #
    :param dt: offset (pts) #
    :param z_bead: Zbead (um) #
    :return: Z (um) #
    """

    force_pN = MagnetForce(x, L1, L2, f0, fmax, alpha, dt)
    z = zWLC(force_pN, L=L, z_bead=z_bead)
    return z


def correct_phase_jump(x, y, z, xy_margin=np.inf, z_margin=5):
    """
    Correct the z-jumps caused by limited range of phase calibration

    :param x: x timetrace
    :param y: y timetrace
    :param z: z timetrace
    :param xy_margin: maximal x/y jump (um)
    :param z_margin: minimal z jump (um)
    :return: correced z timetrace
    """

    jump = np.abs(np.diff(x, prepend=x[0])) < xy_margin
    jump *= np.abs(np.diff(y, prepend=y[0])) < xy_margin
    jump *= np.abs(np.diff(z, prepend=z[0])) > z_margin

    if np.sum(jump) > 0:
        jump_size = np.sum(np.abs(np.diff(z, prepend=z[0])) * jump) / np.sum(jump)
        correction = np.cumsum(jump_size * jump * np.sign(np.diff(z, prepend=z[-1])))
        return z - correction
    else:
        return z


def coth(x):
    return np.cosh(x) / np.sinh(x)


def Langevin(x):
    return coth(x) - 1.0 / x


def zFJC_old(
    f_pN,
    k_pN__nm=0.3,
    z0_nm=0.0,
    L_nm=15,
    nm_base=0.58,
    b_nm=None,
    sigma=None,
    L_base=1,
    S_pN=630,
    EFJC=False,
):
    if b_nm is None and L_nm is not None:
        b_nm = 3 * kT / (L_nm * k_pN__nm)
    if L_bp is not None:
        L_nm = L_base * nm_base
    if EFJC:
        b_nm = 1.55
        S_pN = 630
        f = b_nm * f_pN / kT
        f_S = f_pN / S_pN
        z_nm = L_nm * (
            Langevin(f)
            + (f_S) * (1 + (1 - Langevin(f) * coth(f)) / (1 + f_S * coth(f)))
        )
    else:
        z_nm = (
            L_nm * (coth(b_nm * f_pN / kT) - (kT / (b_nm * f_pN)) + f_pN / S_pN) + z0_nm
        )
    if sigma is not None:
        z_nm = np.ones_like(sigma) * z_nm
    return z_nm


def zFJC(f, k=0.3, z0=0.0, L=25.0, b=None):
    """
    Compute force induced extension of a Freely Jointed Chain

    :param f: force (pN)
    :param k: stiffness (pN/nm), computed from b if not supplied
    :param z0: rest extension (nm)
    :param L: contour length (nm)
    :param b: Kuhn length (nm)
    :return: extension (nm)
    """
    k = np.max([k, 1e-3])
    if b is None:
        b = 3 * kT / (L * k)
    z_nm = L * Langevin(f * b / kT)
    return z_nm + z0


def gFJC(f, k=0.3, L=25.0, b=None):
    k = np.max([k, 1e-3])
    if b is None:
        b = 3 * kT / (L * k)

    g_pNnm = -(kT * L / b) * (np.log((b * f) / (kT)) - np.log(np.sinh((b * f) / (kT))))

    # Remove offset at f = 0
    f = 1e-9
    g_pNnm -= -(kT * L / b) * (np.log((b * f) / (kT)) - np.log(np.sinh((b * f) / (kT))))

    return g_pNnm / kT


def gEFJC(f, k=0.3, L=25.0, b=1.55, sigma=None, C=28, L_bp=1, p=16, S=630, EFJC=False):
    if b is None:
        b = 3 * kT / (L * k)
    if L_bp is not None:
        L = L_bp * nm_base
    if EFJC:
        b = 1.55
        S = 630
        f = b * f / kT
        f_S = f / S
        g_pNnm = (
            -1
            * kT
            * L_bp
            * (
                np.log(np.sinh(f))
                - np.log(f)
                + (b * f**2) / (kT * 2 * S)
                + np.log(1 + f_S * coth(f))
            )
        )
        # Remove offset at f = 0
        f = 1e-9
        g_pNnm -= (
            -1
            * kT
            * L_bp
            * (
                np.log(np.sinh(f))
                - np.log(f)
                + (b * f**2) / (kT * 2 * S)
                + np.log(1 + f_S * coth(f))
            )
        )
    else:
        g_pNnm = -(kT * L / b) * (
            np.log((b * f) / (kT)) - np.log(np.sinh((b * f) / (kT)))
        ) + L * f**2 / (2 * S)

        # Remove offset at f = 0
        f = 1e-9
        g_pNnm -= -(kT * L / b) * (
            np.log((b * f) / (kT)) - np.log(np.sinh((b * f) / (kT)))
        ) + L * f**2 / (2 * S)
    if sigma is not None:
        # See Meng et al. 2014 Biophysical Journal:
        # https://www.cell.com/cms/10.1016/j.bpj.2014.01.017/attachment/7b4445f3-d296-43b2-8dc2-d98e13f5f00f/mmc1.pdf

        c = kT * (omega0**-2) * C
        if p != 0:
            g_pNnm += L_bp * 0.5 * c * ((sigma + 1)) ** p
        g_pNnm += L_bp * 0.5 * c * (sigma + 1) ** 2

    return g_pNnm / kT


def zHooke(f_pN, k_pN__nm=0.3, L0_nm=0):
    return f_pN / k_pN__nm + L0_nm


def gHooke(f_pN, k_pN__nm=0.3, L0_nm=0):
    g_pNnm = 0.5 * f_pN**2 / k_pN__nm - f_pN * L0_nm
    return g_pNnm / kT


def zWLC(x, L=1, A=50, S=1000, z_bead=0):
    """
    extendable Worm-Like Chain model
    :param x: F (pN) # Force
    :param A: A (nm) # persistence length
    :param L: L (bp) # Contour length of DNA
    :param S: S (pN) # Stretch modulus
    :paran z_bead: Zbead (um) # Height offset
    :param tw: twist (turns) # twist
    :return: Z (um) # extension of DNA
    """
    z_nm = nm_bp * L * (1 - 0.5 * np.sqrt(kT / (x * A)) + x / S)
    # if tw is not None:
    #     z_nm = np.ones_like(tw) * z_nm
    return z_nm * 0.001 + z_bead


def gWLC(f_pN, L=1, A=50, S=1000, sigma=None, C_nm=100):
    # definition: g = integral(z(f), df)
    g_pNnm = nm_bp * L * (f_pN - np.sqrt(f_pN * kT / A) + f_pN**2 / (2 * S))
    if sigma is not None:
        # See Meng et al. 2014 Biophysical Journal:
        # https://www.cell.com/cms/10.1016/j.bpj.2014.01.017/attachment/7b4445f3-d296-43b2-8dc2-d98e13f5f00f/mmc1.pdf
        # force dependence of twist persistence length
        C_nm = C_nm * (1 - (C_nm / (4 * A)) * (kT / (A * f_pN)))
        c = kT * (omega0**2) * C_nm
        g_pNnm += L * 0.5 * c * sigma**2
    return g_pNnm / kT


def calc_free_DNA(z, f, L0, A, S):
    L = L0 * z / zWLC(f, L0, A, S)
    return L


def zDNA(f_pN, L_bp=1, tw_turns=None, b_nm=1.55, Ebp_kT=1.57, n_strands=2):
    zds_nm = zWLC(f_pN)
    zss_nm = zFJC(f_pN / n_strands, b=b_nm)
    if tw_turns is None:
        # Torsionally relaxed
        gds = gWLC(f_pN, L=1) - f_pN * zds_nm / kT
        gss = (
            n_strands * gEFJC(f_pN / n_strands, b=1.5, L_bp=1)
            + Ebp_kT
            - f_pN * zss_nm / kT
        )

        # tweak factor: limiting cooperativity, avoiding overflow
        c = np.min([100, L_bp])
        Pds = np.exp(-gds * c) / (np.exp(-gds * c) + np.exp(-gss * c))
        Z_nm = L_bp * (zds_nm * Pds + zss_nm * (1 - Pds))
    else:  # Torsionally constraint
        if np.isscalar(tw_turns):
            tw_turns = np.ones_like(f_pN) * tw_turns

        # Range to compute
        Nds_range = np.linspace(0, L_bp, 100)
        # change of twist in dsDNA to evaluate each force step
        TW_range = np.linspace(-3, 3, 50)

        # Matrices with all distributions of conformations and twist
        Nds = np.outer(Nds_range, np.ones_like(TW_range))
        Nss = L_bp - Nds
        deltaTWds = np.outer(np.ones_like(Nds_range), TW_range)
        TWds_last = tw_turns[0]  # start with all twist in dsDNA

        Z_nm = []
        for f, tw, zds, zss in zip(f_pN, tw_turns, zds_nm, zss_nm):
            TWds = TWds_last + deltaTWds
            TWss = tw - TWds - Nss / 10.4
            sigma_ds = TWds / (omega0 * Nds * nm_bp)
            Gds = gWLC(f, L=Nds, sigma=sigma_ds) - f * zds * Nds / kT
            sigma_ss = TWss / (omega0 * Nss * nm_bp)
            Gss = (
                2 * gEFJC(f / 2, b=1.5, L_bp=Nss, sigma=sigma_ss)
                - (f * zss * Nss / kT)
                + Ebp_kT * Nss
            )
            G = Gds + Gss

            selected = np.isfinite(G)
            G -= np.min(G[selected])  # Avoid large numbers in exponents
            P = np.exp(-G[selected])
            P /= np.sum(P)  # Normalize P
            Z = P * (Nds[selected] * zds + Nss[selected] * zss)
            Z_nm.append(np.sum(Z))
            TWds_last = np.sum(P * TWds[selected])
    return Z_nm


def zDNA_unzip(f, L=1, Lds_bp=1, Lss0_bp=0):
    """
    two state unzipping model

    :param f: force (pN)
    :param L: contourlength (bp)
    :param Lds_bp:
    :param Lss0_bp:
    :return: DNA extension (nm)
    """
    zss_nm = zFJC(f)
    gss = gEFJC(f)
    # gds = gWLC(f_pN)
    zds_nm = zWLC(f, L=Lds_bp) + Lss0_bp * zss_nm
    Lss_bp = L - Lds_bp - Lss0_bp
    G = np.empty((2, len(f)))
    for k in range(0, 2):
        G[k] = -1 * k * Lss_bp * (2 * gss + Ebp_kT - 2 * f * zss_nm / kT)
    log_Z = np.empty(len(f))
    for l in range(len(f)):
        log_Z[l] = logsumexp(G[:, l])
    log_p = np.empty((2, len(f)))
    for k in range(0, 2):
        for l in range(len(f)):
            log_p[k][l] = G[k][l] - log_Z[l]
    p = np.exp(log_p)
    p[0] = 1 - p[1]
    N_unzip = np.zeros(len(f))
    for k in range(0, 2):
        for l in range(len(f)):
            N_unzip[l] += k * Lss_bp * p[k][l]
    z_nm = zds_nm + 2 * N_unzip * zss_nm
    return z_nm


def zFiber_old(
    x,
    L=6000,
    A=50,
    S=1000,
    n4=0,
    n=0,
    k=0.1,
    NLD=1.2,
    dg1=17,
    dg2=5,
    dg3=50,
    NRL=197,
    L1=89,
    L2=80,
    d=1,
    z_bead=0,
):
    """
    Compute force-dependent extension of chromatin fiber according to Meng2019NAR

    :param x: F (pN) # Force
    :param L: L (bp) #contour length (bp)
    :param A: A (nm) # persistence length
    :param S: S (pN) # Stretch modulus
    :param n4: n4 # number of tetrasomes
    :param n: n # number of nucleosomes (incl. tetrasomes)
    :param k: k (pN_nm) # stiffness (pN/nm)
    :param NLD: NLD (nm) #Nucleosome Line Density (nm)
    :param dg1: G1 (kT) #s tacking energy (kT)
    :param dg2: G2 (kT) # partial unwrapping energy (kT) of (l2-L1) basepairs
    :param dg3: G3 (kT) # full unwrapping energy (kT) of last L2 basepairs
    :param NRL: NRL (bp) # Nucleosome Repeat Length (bp)
    :param L1: L1 (bp) # wrapped amount of DNA (bp) in unstacked conformation
    :param L2: L2 (bp) # wrapped amount of DNA (bp) in singly wrapped conformation
    :param d: degeneracy # degeneracy (0 = cooperative, 1 = degenerate)
    :param z_bead: Zbead (um) # Height offset
    :return: Z (um) # fiber extension
    """

    # Compute all possible distributions between states and accompanying degeneracy

    # trick to avoid long computation times for large number of nucleosomes
    n_max = 25
    if n > n_max:
        multiplier = n / int(n / (n // n_max))
    else:
        multiplier = 1

    L /= multiplier
    n = int(n / multiplier)
    n4 = int(n4 / multiplier)
    n8 = n - n4

    states = np.array(
        list(it.combinations_with_replacement([0, 1, 2, 3], n))
    )  # all possible states
    # remove states with more than n nucleo-/tetrasomes
    states = np.array([s for s in states if np.sum(s <= 1) <= n])
    state = np.array(
        [
            [np.sum(s == 0), np.sum(s == 1), np.sum(s == 2), np.sum(s == 3)]
            for s in states
            if np.sum(s <= 1) <= n8
        ]
    )  # count each conformation

    degeneracy = np.array(
        [
            binom(n, s[1]) * binom(n - s[1], s[2]) * binom(n - s[1] - s[2], s[3])
            for s in state
        ]
    )
    degeneracy = d * (degeneracy - 1) + 1

    # Compute extension and energy per stacked nucleosome and per basepair
    z0 = zFJC(x, k=k, z0=NLD, L=25)
    # g0 = gFJC(f, k_pN__nm=k) / kT

    # z0 = zHooke(f, k, NLD)
    g0 = gHooke(x, k, NLD)
    # g0[f > 10] *= 100

    z_bp = zWLC(x, 1, S=S, A=A) * 1000
    g_bp = gWLC(x, 1, S=S, A=A) / kT

    # Compute Boltzmann weighted average extension of all states for each force
    z_fiber = np.zeros_like(x)
    for i, f_i in enumerate(x):
        z_state = state * np.array(
            [z0[i], z_bp[i] * (NRL - L1), z_bp[i] * (NRL - L2), z_bp[i] * NRL]
        )

        g_state = state * np.array(
            [g0[i], g_bp[i] * (NRL - L1), g_bp[i] * (NRL - L2), g_bp[i] * NRL]
        )
        g_state -= f_i * z_state / kT
        # Rupture energies
        g_state += state * np.array([0, dg1, dg1 + dg2, dg1 + dg2 + dg3])
        g_state = np.sum(g_state, axis=1)
        g_state -= np.min(g_state)  # Avoid numerical overflow

        z_state = np.sum(z_state, axis=1)
        P_state = degeneracy * np.exp(-g_state)
        P_state /= np.sum(P_state)

        z_fiber[i] = np.sum(P_state * z_state)

    z_fiber += z_bp * (L - n * NRL)  # add DNA handles
    return z_fiber * 0.001 * multiplier + z_bead


def zFiber(
    x,
    L=6000,
    A=50,
    S=1000,
    n4=0,
    n=0,
    k=0.1,
    NLD=1.2,
    dg1=17,
    dg2=5,
    dg3=50,
    NRL=197,
    L1=89,
    L2=80,
    d=1,
    z_bead=0,
):
    """
    Compute force-dependent extension of chromatin fiber according to Meng2019NAR

    :param x: F (pN) # Force
    :param L: L (bp) #contour length (bp)
    :param A: A (nm) # persistence length
    :param S: S (pN) # Stretch modulus
    :param n4: n4 # number of tetrasomes
    :param n: n # number of nucleosomes (incl. tetrasomes)
    :param k: k (pN_nm) # stiffness (pN/nm)
    :param NLD: NLD (nm) #Nucleosome Line Density (nm)
    :param dg1: G1 (kT) #s tacking energy (kT)
    :param dg2: G2 (kT) # partial unwrapping energy (kT) of (l2-L1) basepairs
    :param dg3: G3 (kT) # full unwrapping energy (kT) of last L2 basepairs
    :param NRL: NRL (bp) # Nucleosome Repeat Length (bp)
    :param L1: L1 (bp) # wrapped amount of DNA (bp) in unstacked conformation
    :param L2: L2 (bp) # wrapped amount of DNA (bp) in singly wrapped conformation
    :param d: degeneracy # degeneracy (0 = cooperative, 1 = degenerate)
    :param z_bead: Zbead (um) # Height offset
    :return: Z (um) # fiber extension
    """

    x = np.asarray(x)
    # Compute extension and energy per stacked nucleosome and per basepair
    z0 = zFJC(x, k=k, z0=NLD, L=25)
    g0 = gHooke(x, k, NLD)

    # Compute the extension and energy of free DNA
    z_bp = zWLC(x, 1, S=S, A=A) * 1000
    g_bp = gWLC(x, 1, S=S, A=A) / kT

    # Combine extension and energy of each states for all forces
    z_state = np.array([z0, z_bp * (NRL - L1), z_bp * (NRL - L2), z_bp * NRL])
    g_state = np.array(
        [
            g0,
            g_bp * (NRL - L1) + dg1,
            g_bp * (NRL - L2) + dg1 + dg2,
            g_bp * NRL + dg1 + dg2 + dg3,
        ]
    )

    # subtract work done by force
    g_state = g_state - x[np.newaxis, :] * z_state / kT
    g_min = np.min(g_state, axis=0)  # Avoid numerical overflow
    g_state -= g_min[np.newaxis, :]

    # compute Boltzmann weights
    P_state = np.exp(-g_state)
    sum = np.sum(P_state, axis=0)
    P_state /= sum[np.newaxis, :]

    if d < 1:  # adjust for cooperativity of first transition
        g_state = (1 - d) * n * g_state[:2]
        g_min = np.min(g_state, axis=0)  # Avoid numerical overflow
        g_state -= g_min[np.newaxis, :]
        P_state_coop = np.exp(-g_state)
        sum = np.sum(P_state_coop, axis=0)
        P_state_coop /= sum[np.newaxis, :]
        P_state[0] = P_state_coop[0]
        P_state[1] *= 0
        P_state[1] = 1 - np.sum(P_state, axis=0)

    # compute average extension
    z_fiber = np.sum(P_state * z_state, axis=0)
    z_fiber *= n - n4

    if n4 > 0:  # add tetrasomes
        p4 = np.exp(-g_state[2:])
        sum = np.sum(p4, axis=0)
        p4 /= sum[np.newaxis, :]
        z_fiber += np.sum(p4 * z_state[2:], axis=0) * n4

    z_fiber += z_bp * (L - n * NRL)  # add DNA handles

    return z_fiber * 0.001 + z_bead


def calculate_A_Kulic(L_bp, A=50, N=1, alpha=0):
    #   Persistence length correction according to Kulic2005PRE
    C = 8 * (1 - np.cos(alpha * (np.pi / 180) / 4.0))
    rho = N / (L_bp * 0.34)
    return A / ((1 + A * N * C * rho) ** 2)


def zHMf(
    x,
    L=3646,
    A=50,
    S=1000,
    k=0.5,
    z_dimer=4,
    L_dimer=30,
    angle=10,
    n_dimer=100,
    n_gap=0,
    g1=1.9,
    g2=4.11,
    Z0=0,
):
    """
    Calculate extension of HMf fiber according to Henneman2020NAR
    :param x: F (pN) # Force
    :param L: L (bp) # [1, inf]  contour length
    :param A: A (nm) # [1, 500] persistence length
    :param S: S (pN) # [1, 5000]Stretch modulus
    :param k: k (pN_nm) [0.01, inf] # Stiffness of stacked fiber
    :param z_dimer: Zdimer (nm) # [0.01, 10] maximal extension of FJC of stacked fiber
    :param L_dimer: Ldimer (bp) # [1, 50] Footprint of dimer
    :param angle: angle (deg) # [0, 90]  deflection angle of DNA induced by dimer
    :param n_dimer: Ndimer (-) # [0, 300] number of bound proteins
    :param n_gap: Ngap (-) # [0, 100] Number of defects in hypernucleosome
    :param g1: G1 (kT) # [0, 10] Stacking energy
    :param g2: G2 (kT) # [0, 10] Wrapping energy
    :param Z0: Zbead (um) # [-2, 0] Offset due to bead atachement
    :return: Z (um) # extension of the fiber
    """
    states, degeneracies = calc_degeneracy([n_dimer, n_gap, 0], degeneracy=0)
    if g1 <= 0:
        states = np.asarray([[n_dimer, 0, 0]])
        degeneracies = np.ones_like(states)
    if g2 <= 0:
        states = np.asarray([[0, n_dimer, 0]])
        degeneracies = np.ones_like(states)

    z_max = z_dimer * 1
    z0 = zFJC(x, k, L=z_max)
    g0 = gFJC(x, k, L=z_max)
    z_states = np.zeros((len(states), len(x)))
    g_states = np.zeros_like(z_states)
    for i, s in enumerate(states):
        Ldna = L - L_dimer * s[0]
        A_app = calculate_A_Kulic(Ldna, alpha=angle, N=s[1], A=A)
        z_states[i] = z0 * s[0] + zWLC(x, Ldna, S=S, A=A_app) * 1000
        g_states[i] = (
            g0 * s[0]
            + (gWLC(x, Ldna, S=S, A=A_app) - x * z_states[i]) / kT
            - (g1 + g2) * s[0]
            - g2 * s[1]
        )
    z_fiber = np.zeros_like(x)
    for i, (z, g) in enumerate(zip(z_states.T, g_states.T)):
        g -= np.min(g)
        P = degeneracies * np.exp(-g)
        P /= np.sum(P)
        z_fiber[i] = np.sum(P * z)
    return z_fiber * 0.001 + Z0


def zHMf_new(
    x,
    L=3646,
    A=50,
    S=1000,
    k=0.5,
    z_dimer=4,
    L_dimer=30,
    angle=10,
    n_dimer=100,
    n_gap=0,
    g1=1.9,
    g2=4.11,
    Z0=0,
):
    """
    Calculate extension of HMf fiber according to Henneman2020NAR
    :param x: F (pN) # Force
    :param L: L (bp) # [1, inf]  contour length
    :param A: A (nm) # [1, 500] persistence length
    :param S: S (pN) # [1, 5000]Stretch modulus
    :param k: k (pN_nm) [0.01, inf] # Stiffness of stacked fiber
    :param z_dimer: Zdimer (nm) # [0.01, 10] maximal extension of FJC of stacked fiber
    :param L_dimer: Ldimer (bp) # [1, 50] Footprint of dimer
    :param angle: angle (deg) # [0, 90]  deflection angle of DNA induced by dimer
    :param n_dimer: Ndimer (-) # [0, 300] number of bound proteins
    :param n_gap: Ngap (-) # [0, 100] Number of defects in hypernucleosome
    :param g1: G1 (kT) # [0, 10] Stacking energy
    :param g2: G2 (kT) # [0, 10] Wrapping energy
    :param Z0: Zbead (um) # [-2, 0] Offset due to bead atachement
    :return: Z (um) # extension of the fiber
    """

    x = np.asarray(x)
    # Compute extension and energy per stacked stacked dimer
    z_max = z_dimer * 1
    z0 = zFJC(x, k, L=z_max)
    g0 = gFJC(x, k, L=z_max)

    # Compute the extension and energy of wrapped DNA per dimer
    A_wrapped = calculate_A_Kulic(L_dimer * n_dimer, alpha=angle, N=n_dimer, A=A)
    z_wrapped = zWLC(x, L_dimer, S=S, A=A_wrapped) * 1000
    g_wrapped = gWLC(x, L_dimer, S=S, A=A_wrapped) / kT

    # Compute the extension and energy of unwrapped DNA per dimer
    z_unwrapped = zWLC(x, L_dimer, S=S, A=A) * 1000
    g_unwrapped = gWLC(x, L_dimer, S=S, A=A) / kT

    # Combine extension and energy of each states for all forces per dimer
    z_state = np.array([z0, z_wrapped, z_unwrapped])
    g_state = np.array([g0, g_wrapped + g1, g_unwrapped + g1 + g2])

    # subtract work done by force
    g_state = g_state - x[np.newaxis, :] * z_state / kT
    g_min = np.min(g_state, axis=0)  # Avoid numerical overflow
    g_state -= g_min[np.newaxis, :]

    # compute Boltzmann weights
    # P_state = (np.exp(-g_state* np.log(n_dimer)))
    P_state = np.exp(-g_state) ** (n_dimer / n_gap)
    sum = np.sum(P_state, axis=0)
    P_state /= sum[np.newaxis, :]

    # plt.close('all')
    # plt.plot(x, P_state.T)
    # plt.show()

    # compute average extension
    z_fiber = np.sum(P_state * z_state, axis=0)
    z_fiber *= n_dimer
    z_fiber += (z_unwrapped / L_dimer) * (L - n_dimer * L_dimer)  # add unbound DNA
    return z_fiber * 0.001 + Z0


def zHMf2(
    x,
    L=3646,
    A=50,
    S=1000,
    k=0.5,
    z_dimer=4,
    L_dimer=30,
    angle=10,
    n_dimer=100,
    n_gap=0,
    g1=1.9,
    g2=4.11,
    Z0=0,
):
    """
    Calculate extension of HMf fiber according to Henneman2020NAR.
    This function is a wrapper to allow for non-integer n_dimer and n_gap values.

    :param x: F (pN) # Force
    :param L: L (bp) # [1, inf]  contour length
    :param A: A (nm) # [1, 500] persistence length
    :param S: S (pN) # [1, 5000] Stretch modulus
    :param k: k (pN_nm) # [0.01, inf] Stiffness of stacked fiber
    :param z_dimer: Zdimer (nm) # [0.01, 10] maximal extension of FJC of stacked fiber
    :param L_dimer: Ldimer (bp) # [1, 50] Footprint of dimer
    :param angle: angle (deg) # [0, 90]  deflection angle of DNA induced by dimer
    :param n_dimer: Ndimer (-) # [0, 300] number of bound proteins
    :param n_gap: Ngap (-) # [0, 100] Number of defects in hypernucleosome
    :param g1: G1 (kT) # [0, 10] Stacking energy
    :param g2: G2 (kT) # [0, 10] Wrapping energy
    :param Z0: Zbead (um) # [-2, 0] Offset due to bead attachment
    :return: Z (um) # extension of the fiber
    """
    pars = locals()
    extension_curves = []
    for dimers in [np.floor(n_dimer), np.floor(n_dimer) + 1]:
        for gaps in [np.floor(n_gap), np.floor(n_gap) + 1]:
            pars["n_dimer"] = dimers
            pars["n_gap"] = gaps
            z = zHMf(**pars)
            extension_curves.append((dimers, gaps, z))
    z_fiber = bilinear_interpolation(n_dimer, n_gap, extension_curves)
    return z_fiber


def zMedianFilter(x, w=1.0, t=None):
    """
    Calculate median filter.

    :param x: z (um) # Heigth of tether
    :param w: w (s) # Width of median filter
    :param t: t (s) # Time trace to determine filter window
    :return: Z (um) # Median filtered height of tether
    """
    if t is None:
        window = w
    else:
        dt = t[1] - t[0]
        window = round(w / dt)
        window = (
            window
            if window % 2 == 1
            else (window + 1 if window < w / dt else window - 1)
        )
    return medfilt(x, window)


def calc_degeneracy(states_start, degeneracy=1):
    shift = np.zeros_like(states_start)
    shift[0] = -1
    shift[1] = 1

    states = [[int(s) for s in states_start]]
    i = 0
    while i < len(states_start) - 1:
        states.append(states[-1] + shift)
        if states[-1][i] == 0:
            i += 1
            shift = np.roll(shift, 1)

    d = np.ones(len(states))
    if degeneracy > 0:
        for i, _ in enumerate(states_start[:-1]):
            d *= [binom(np.sum(s[i:]), s[i]) for s in states]
    if degeneracy < 1:
        d = degeneracy * (d - 1) + 1
    return states, d


def drift(time_s, dz_dt_nm__s=0, z0_nm=0):
    return (time_s * dz_dt_nm__s) + z0_nm


def ExponentialDrift(x, Z0=0, dZ_dt=0, tau=np.inf):
    """
    Calculate drift, assuming an exponential decay
    :param x: Time (s)  # Time
    :param Z0: Z0 (um) # offset
    :param dZ_dt: dZ_dt (nm_s) # drift
    :param tau: tau (s) # decay time
    :return: Z (um) #
    """
    time = x
    if dZ_dt == 0:
        return np.zeros_like(time) + Z0
    if tau == np.inf:
        tau = 1e9
    a = -dZ_dt / 1000.0 * tau
    b = tau
    c = Z0 - a
    return a * np.exp(-time / b) + c


def exp_func(x, a, b, c):
    return a * np.exp(-x / b) + c


def get_free_DNA(z, f, L, A, S):
    wlc = zWLC(f, L, A, S)
    Lfree = L * z / wlc
    Lfree[Lfree > L * 1.05] = np.nan
    return Lfree


def fit_zbead(f, z, L=1, A=50, S=1000, selection=None):
    from scipy.special import erf

    def skewed_gaussian(x, a, x0, sigma, alpha):
        return (
            a
            * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
            * (1 + erf(alpha * (x - x0) / (sigma * np.sqrt(2))))
        )

    bins = np.linspace(-1.5, 0.1, 300)
    wlc = zWLC(f, L, A, S)

    if selection is None:
        selection = z > 0.25  # discard data points close to surface
    else:
        selection = selection & (z > 0.25)

    # plt.close('all')
    # plt.plot(z[selection]-wlc[selection])
    # plt.show()

    shift, bins = np.histogram(z[selection] - wlc[selection], bins=bins)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2

    initial_guess = [max(shift), bins[np.argmax(shift)], 0.05, 1]
    try:
        sigma = [np.sqrt(1 / s) if s > 0 else np.inf for s in shift]
        popt, cov = curve_fit(
            skewed_gaussian, bins, shift, p0=initial_guess, sigma=sigma
        )

        # plt.close('all')
        # plt.plot(bins, shift, 'o')
        # plt.plot(bins, skewed_gaussian(bins, *popt))
        # plt.show()
        return popt[1], np.sqrt(cov[1, 1])
    except:
        return np.nan, np.nan


def plot_from_h5(filename, label):
    with tio.Traces(filename) as data:
        z_traces = tio.filter_traces(data, channel="Z (um)")
        for i in z_traces:
            print(i)


def initial_analysis(files):
    """
    do initial analysis of *.dat file, including conversion to h5, fit exponential drift
    and produce z(t) and F(z) plots in subdirectory

    :param files: list of files to process, use wildcards to select multiple files
    """
    filenames = glob.glob(files)

    shift = np.linspace(0, 12)
    force = MagnetForce(shift, fmax=100)

    L_bp = 4817
    F2 = np.logspace(-2, 2, 100)
    z_DNA1 = zDNA(F2, L_bp) / 1000
    z_DNA2 = zDNA(F2, (L_bp - 16 * 80)) / 1000

    for filename in filenames:
        try:
            with tio.Traces(filename) as data:
                t = data.read_trace(
                    tio.filter_traces(data.contents(), channel="Time (s)")
                )
                shift = data.read_trace(
                    tio.filter_traces(data.contents(), channel="Stepper shift (mm)")
                )
                force = np.asarray(MagnetForce(shift, fmax=100))
                z_DNA = zDNA(force, L_bp) / 1000
                z_traces = tio.filter_traces(data.contents(), channel="Z (um)")
                a_traces = tio.filter_traces(data.contents(), channel="Amp")

                for i, z_trace in enumerate(z_traces):
                    label = int(z_trace.split(">")[-1])
                    title = data.plot_title(z_trace, dir_depth=-2)
                    print(title)
                    z = data.read_trace(z_trace)
                    amplitude = data.read_trace(a_traces[i])

                    selection = (shift > 6.5) & (
                        amplitude > np.mean(amplitude[0:25]) / 10
                    )
                    sigma = 1 / amplitude

                    try:
                        p, cov = curve_fit(
                            exp_func,
                            t[selection],
                            z[selection],
                            sigma=sigma[selection],
                            maxfev=5000,
                        )
                        drift = exp_func(np.asarray(t), *p)
                        for i, par in enumerate(
                            ["Adrift (um)", "Tdrift (s)", "Z0 (um)"]
                        ):
                            data.write_par(label, par, p[i], np.sqrt(cov[i, i]))
                    except:
                        drift = np.mean(z)
                    z -= drift

                    fileout = tio.change_extension(
                        filename, "jpg", label=label, dir="z_t"
                    )
                    plt.scatter(t, z, s=20, facecolors="none", edgecolors="lightgrey")
                    plt.scatter(
                        t[selection],
                        z[selection],
                        s=20,
                        facecolors="none",
                        edgecolors="red",
                    )
                    plt.plot(t, z_DNA, color="black")
                    tio.format_plot(
                        "t (s)", "z (um)", yrange=[-0.2, 3.2], title=title, save=fileout
                    )

                    selection = force > 30
                    dz = np.mean(
                        (z_DNA - z)[selection] * amplitude[selection]
                    ) / np.mean(amplitude[selection])

                    selection = amplitude > np.mean(amplitude[:50]) / 50
                    # selection = (amplitude >1e7)

                    fileout = tio.change_extension(
                        filename, "jpg", label=label, dir="Fz"
                    )
                    plt.scatter(
                        z + dz, force, s=20, facecolors="none", edgecolors="lightgrey"
                    )
                    plt.scatter(
                        z[selection] + dz,
                        force[selection],
                        s=20,
                        facecolors="none",
                        edgecolors="red",
                    )
                    plt.plot(z_DNA1, F2, color="black")
                    plt.plot(z_DNA2, F2, color="black", linestyle="dashed")
                    tio.format_plot(
                        "z (um)",
                        "F (pN)",
                        title=title,
                        xrange=[-0.2, 3.2],
                        yrange=[-5, 50],
                        save=fileout,
                    )
                print(filename)
        except:
            print(f"Error in computing F-z traces in file {filename}")


def fit_zbead_from_rotation(data, show=False, bead_radius=1.4):
    def fit_circle(params, x, y):
        radius, cx, cy = params
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return np.sum((distances - radius) ** 2)

    x = data.traces["X (um)"].values
    y = data.traces["Y (um)"].values

    initial_guess = [
        1.0,
        data.get_parameter("X0 (um)", default=0),
        data.get_parameter("Y0 (um)", default=0),
    ]
    result = minimize(fit_circle, initial_guess, args=(x, y), method="Nelder-Mead")

    num_bootstraps = 100
    fitted_parameters = []
    for _ in range(num_bootstraps):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_bootstrap = x[indices]
        y_bootstrap = y[indices]
        result = minimize(
            fit_circle,
            initial_guess,
            args=(x_bootstrap, y_bootstrap),
            method="Nelder-Mead",
        )
        fitted_parameters.append(result.x)
    parameter_errors = np.std(np.array(fitted_parameters), axis=0)

    z_bead = np.sqrt(bead_radius**2 - result.x[0] ** 2) - bead_radius
    error_z_bead = np.sqrt(
        ((result.x[0] / np.sqrt(z_bead**2 + result.x[0])) ** 2)
        * parameter_errors[0] ** 2
    )

    data.set_parameter("Zbead (um)", z_bead, error_z_bead)
    data.set_parameter("Rbead (um)", bead_radius)
    for i, p in enumerate(["Radius (um)", "X0 (um)", "Y0 (um)"]):
        data.set_parameter(p, result.x[i], parameter_errors[i])

    if show:
        print(f"z_bead = {z_bead:.3f} +/- {error_z_bead:.3f}um")
        plt.close("all")
        plt.scatter(x, y, s=20, facecolors="none", edgecolors="red")
        phi = np.linspace(0, 2 * np.pi, 100)
        plt.plot(
            result.x[0] * np.cos(phi) + result.x[1],
            result.x[0] * np.sin(phi) + result.x[2],
            color="black",
        )
        plt.show()
    return data


if __name__ == "__main__":
    if False:  # generate overstretching FD-curve
        force = np.logspace(np.log10(0.05), np.log10(65), 100)
        z = zWLC(force, 3000)

        plt.plot(
            z,
            force,
            color="blue",
            marker="o",
            markerfacecolor="none",
        )
        plt.xlabel("z (um)")
        plt.ylabel("F (pN)")
        plt.show()

    if True:  # test hdf file with sttings
        data = tio.hdf_data(f"c:\tmp\test.h5")
        print(data.filename)
