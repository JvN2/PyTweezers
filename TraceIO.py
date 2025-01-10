import inspect
import traceback
import pandas as pd
import os
import re
import time
import tables
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from lmfit import Model
from mpl_toolkits.axes_grid1 import Divider, Size
from nptdms import TdmsFile
import warnings
from tables import NaturalNameWarning
import seaborn as sns
import tkinter as tk
import lmfit
from icecream import ic
from ast import literal_eval
from datetime import datetime
import threading
from icecream import ic

import ForceSpectroscopy as fs

warnings.simplefilter("ignore", NaturalNameWarning)

# Process the up and down arrow keys to the tk.entry widgets


def increment_filename(old_filename=None, base_dir=None):
    if old_filename:
        current_file_nr = int(old_filename[-7:-4]) + 1
    else:
        current_file_nr = 0

    if not current_file_nr:
        current_file_nr = 0

    if not base_dir:
        if os.path.exists("d:/"):
            disk = "d"
        else:
            disk = "c"
        date = datetime.now().strftime("%Y%m%d")
        base_dir = Path(f"{disk}:/users/{os.getlogin()}/data/{date}")
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = base_dir / f"data_{current_file_nr:03d}.hdf"

    while filename.exists():
        current_file_nr += 1
        filename = base_dir / f"data_{current_file_nr:03d}.hdf"
    return filename


def adjust_value(
    tk_entry, direction, event=None, update_func=None, update_args_func=None
):
    try:
        value = float(tk_entry.get())
        cursor_position = tk_entry.index(tk.INSERT)
        text = tk_entry.get()
        if "." in text:
            decimal_position = text.index(".")
            if cursor_position <= decimal_position:
                factor = 10 ** (decimal_position - cursor_position)
            else:
                factor = 10 ** (decimal_position - cursor_position + 1)
        else:
            factor = 10 ** (len(text) - cursor_position)

        # Modify factor by 10 if Ctrl/Shift key is pressed
        if event is not None:
            if event.state & 0x0004:  # Check if Ctrl key is pressed
                factor /= 10
            if event.state & 0x0001:  # Check if Shift key is pressed
                factor *= 10

        new_value = value + direction * factor
        tk_entry.delete(0, tk.END)
        tk_entry.insert(0, f"{new_value:.6f}".rstrip("0").rstrip("."))
        tk_entry.icursor(cursor_position)
        if update_args_func is not None:
            update_func(update_args_func())
        else:
            update_func()
    except ValueError:
        pass
    finally:
        tk_entry.focus_force()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(f"{time.asctime()} @timeit: <{method.__name__}> took {te - ts:.3f} s")
        return result

    return timed


def debug(*args, msg=None):
    """Prints the name and value of variables in a Python function, with the calling function name.

    Args:
      *args: The variables to print.
    """

    calling_function = inspect.stack()[1].function
    print("\n>>>>>>>>>>debugging<<<<<<<<<<<")
    print(f"Calling function: {calling_function}")
    if msg:
        print(msg)
    for a in args:
        if hasattr(a, "__name__"):
            print(f"{a.__name__}: {a}")
        else:
            if isinstance(a, list) or isinstance(a, tuple) or isinstance(a, set):
                for a_i in a:
                    print(f" > {a_i} <")
            else:
                print(f"{a}")
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<<")


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """

    def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def round_significance(x, stderr=None, par_name=None, delimiter=" ± "):
    """
    Compute proper significant digits. Standard error has 1 significant digit, unless
    that digit equals 1. In that cas it has 2 significant digits. Accuracy of x value
    matches accuracy of error.
    :param x: value
    :param stderr: standard error or standard deviation
    :param par_name: name of variable, preferably with units in brackets: 'x (a.u.)'
    :return: If par_name is provided a string will result: 'x (a.u.) = value +/- err'.
             Otherwise the rounded values of x and stderr.
    """
    if pd.isna(x):
        str_value = "-"
    elif pd.isna(stderr) or stderr <= 0:
        str_value = f"{x:g}"
    else:
        try:
            position = -int(np.floor(np.log10(abs(stderr))))
            first_digit = int(stderr * 10**position)
            if first_digit == 1:
                position += 1
            if position < 0:
                position = 0
            x_str = "{:.{}f}".format(x, position)
            stderr_str = "{:.{}f}".format(stderr, position)
            str_value = f"{x_str}{delimiter}{stderr_str}"
        except (ValueError, OverflowError):
            str_value = "-"
    if par_name is not None:
        str_value = f"{par_name} = " + str_value

    return str_value


def prepare_fit(func, initial_parameters=None):
    """
    Prepare all settings for fitting or evaluating a function.
    Make sure the that the function:

    - has a fully filled out doc string
    - the independent variable is called 'x' and comes first
    - each variable, including the independent variable and the return entry, has a proper, readable name.
    - the readable name includes units in brackets
    - the readable name is terminated by a hash and optionally supplemented by a description.
      (i.e. a valid entry would be:  ':param x: F (pN) # The force that was applied.')

    :param func: a function that is implemented in an imported module
    :return: an LMfit model, the corresponding LMfit parameters, and a dictionary of the names
             that describe the used function variables.
    """
    try:
        model = Model(getattr(fs, func))
    except AttributeError:
        print(f"Function {func} not found in ForceSpectroscopy.py")
        return None, None, None

    params = model.make_params()
    for p in params:
        params[p].type = "Global"
        params[p].error = None

    doc = getattr(fs, func).__doc__

    names = {}
    names["x"] = doc.split("x:")[1].split("#")[0].strip()
    names["y"] = doc.split("return:")[1].split("#")[0].strip()

    for key, item in params.items():
        names[key] = doc.split(f"{key}:")[1].split("#")[0].strip()
        params[key].vary = False
        try:
            range = (
                doc.split(f"{key}:")[-1]
                .split("\n")[0]
                .split("]")[0]
                .split("[")[-1]
                .strip()
                .split(", ")
            )
            range = np.asarray(range).astype(float)
            params[key].set(min=range[0], max=range[1])
        except ValueError:
            pass
    if initial_parameters is not None:
        params.update(initial_parameters)

    return model, params, names


def apply_fit(func, x, y, s=None, initial_parameters=None):
    def calculate_r_squared(y_obs, y_fit):
        residuals = y_obs - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    model, params, names = prepare_fit(func, initial_parameters=initial_parameters)

    if s is not None:
        x = x[s == 1]
        y = y[s == 1]

    # Check if all parameters are fixed
    all_fixed = all(not param.vary for param in params.values())

    if all_fixed:
        # Evaluate the model with fixed parameters
        y_fit = model.eval(params=params, x=x)
        r2 = calculate_r_squared(y, y_fit)
        # Create a mock result object
        result = lmfit.model.ModelResult(model, params, method="leastsq")
        result.y_fit = y_fit
        result.r2 = r2
        result.success = True
        result.message = "All parameters are fixed; no fitting performed."
    else:
        # Perform the fit
        result = model.fit(y, params, x=x)
        y_fit = result.eval(x=x)
        r2 = calculate_r_squared(y, y_fit)
        result.r2 = r2

    return result


def fix_axes(axew=7, axeh=7):
    # axew = axew / 2.54
    # axeh = axeh / 2.54

    topmargin = 0.5

    # lets use the tight layout function to get a good padding size for our axes labels.
    fig = plt.gcf()
    fig.set_size_inches(axew + 1.5, axeh + topmargin)

    ax = plt.gca()
    fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top + topmargin * 1.5
    b = ax.figure.subplotpars.bottom + topmargin * 1.5

    # work out what the new  ratio values for padding are, and the new fig size.
    neww = axew + oldw * (1 - r + l)
    newh = axeh + oldh * (1 - t + b)
    newr = r * oldw / neww
    newl = l * oldw / neww
    newt = t * oldh / newh
    newb = b * oldh / newh

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 0.95, 0.95), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure


def key_press_action(event):
    print(event.key)
    fig = plt.gcf()
    if event.key == "z":
        print(fig.texts[0].get_text())
    return


def format_plot(
    xtitle="x (a.u.)",
    ytitle="y (a.u.)",
    title="",
    xrange=None,
    yrange=None,
    ylog=False,
    xlog=False,
    scale_page=1.0,
    aspect=0.5,
    save=None,
    boxed=True,
    GUI=False,
    ref="",
    legend=None,
    fig=None,
    ax=None,
    txt=None,
    empty_canvas=False,
):
    # adjust the format to nice looks
    from matplotlib.ticker import AutoMinorLocator
    import os
    import subprocess

    page_width = 7  # inches ( = A4 width - 1 inch margin)
    margins = (0.55, 0.45, 0.2, 0.2)  # inches
    fontsize = 14

    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()

    if empty_canvas:
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    # Set up figure
    fig_width = page_width * scale_page * 0.95
    fig_height = (
        (fig_width - (margins[0] + margins[2])) * aspect + margins[1] + margins[3]
    )

    if txt:
        txt_width = 2
    else:
        txt_width = 0

    fig.set_size_inches(fig_width + txt_width, fig_height)

    # Set up axes
    ax_rect = [margins[0] / fig_width]
    ax_rect.append(margins[1] / fig_height)
    ax_rect.append((fig_width - margins[0] - margins[2]) / (fig_width + txt_width))
    ax_rect.append((fig_height - margins[1] - margins[3]) / fig_height)

    # ax_rect = [margins[0] / fig_width,
    #            margins[1] / fig_height,
    #            (fig_width - margins[2] - margins[0]) / fig_width,
    #            (fig_height - margins[3] - margins[1]) / fig_height
    #            ]

    ax.set_position(ax_rect)

    # Add axis titles and frame label; use absolute locations, rather then leaving it up to Matplotlib
    if ref is not None:
        plt.text(
            ax_rect[1] * 0.15,
            ax_rect[-1] + ax_rect[1],
            ref,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=fontsize * 1.2,
            transform=fig.transFigure,
        )
    plt.text(
        ax_rect[0] + 0.5 * ax_rect[2],
        0,
        xtitle,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=fontsize,
        transform=fig.transFigure,
    )
    plt.text(
        ax_rect[1] * 0.005,
        ax_rect[1] + 0.5 * ax_rect[3],
        ytitle,
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=fontsize,
        transform=fig.transFigure,
        rotation=90,
    )

    if legend is not None:
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.edgecolor"] = "none"
        plt.rcParams["legend.labelspacing"] = 0.25
        plt.rcParams["legend.handlelength"] = 1
        plt.rcParams["legend.handletextpad"] = 0.25
        plt.legend(
            legend,
            prop={"size": fontsize * 0.8},
        )

    # fig.canvas.mpl_connect("key_press_event", key_press_action)

    if not boxed:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        which="both",
        axis="both",
        bottom=True,
        top=boxed,
        left=True,
        right=boxed,
        direction="in",
    )
    ax.tick_params(which="major", length=4, labelsize=fontsize * 0.8, width=1)
    ax.tick_params(which="minor", length=2, width=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1)

    if xlog:
        ax.semilogx()
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    if ylog:
        ax.semilogy()
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    try:
        ax.set_xlim([float(x) if x else None for x in xrange])
    except (ValueError, TypeError):
        pass
    try:
        ax.set_ylim([float(x) if x else None for x in yrange])
    except (ValueError, TypeError):
        pass

    if txt:
        pos = (ax.get_xlim()[1] * 1.025, ax.get_ylim()[1])
        if ylog:
            ypos = np.logspace(
                np.log10(ax.get_ylim()[1]), np.log10(ax.get_ylim()[0]), 25
            )
        else:
            ypos = np.linspace(ax.get_ylim()[1], ax.get_ylim()[0], 25)
        for t, y in zip(txt, ypos):
            plt.text(pos[0], y, t, size=fontsize / 1.5)

    if not GUI and save is None:
        plt.show()

    if save is not None:
        if not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))
        base, ext = os.path.splitext(save)
        if ext == ".emf":
            save = base + ".pdf"
        fig.savefig(save, dpi=600, transparent=True)
        if ext == ".emf":
            try:
                subprocess.call(
                    [
                        r"C:\Program Files\Inkscape\inkscape.exe",
                        "--file",
                        save,
                        "--export-emf",
                        base + ".emf",
                    ]
                )
                os.remove(save)
            except:
                print(
                    "Install Inkscape for conversion to emf.\nPlot saved as pdf in stead."
                )

    return fig, ax


def plot_heatmap(window, plot, xdata, ydata, size=100):
    ax = plt.gca()
    plt.hexbin(xdata, ydata, cmap=plt.cm.Blues, vmax=3, vmin=0)
    return


class hdf_data(object):
    def __init__(self, filename, label=None):
        self.open_files = []
        self.filename = Path(filename).with_suffix(".hdf")
        self.label = None
        self.shared_tracenames = []
        self.parameters = pd.DataFrame()
        self.traces = pd.DataFrame()
        self.settings = {}

        try:
            with tables.open_file(filename, mode="r") as hdf5_file:
                labels = [
                    l
                    for l in hdf5_file.root._v_children.keys()
                    if l not in [" parameters", " shared"]
                ]
                labels = sorted(labels, key=natural_keys)
                label = labels[0]
        except (FileNotFoundError, IndexError) as e:
            return

        if filename is not None:
            if label is None:
                label = "0"
            self.read(filename, label)

    def read(self, filename, label, save=False):
        filename = Path(filename).with_suffix(".hdf")
        if filename == self.filename and label == self.label:
            return False

        if save:
            self.save()

        with tables.open_file(filename, mode="r") as hdf5_file:
            labels = [
                l
                for l in hdf5_file.root._v_children.keys()
                if l not in [" parameters", " shared"]
            ]
            labels = sorted(labels, key=natural_keys)
            if isinstance(label, int):
                self.label = labels[label]
            elif isinstance(label, str):
                self.label = label
            try:
                self.traces = pd.DataFrame(hdf5_file.get_node(rf"/{self.label}")[:])
                shared = pd.DataFrame(hdf5_file.get_node("/ shared")[:])
                self.shared_tracenames = shared.columns
                self.traces = self.traces.combine_first(shared)
            except tables.NoSuchNodeError:
                return False

            try:
                self.parameters = pd.DataFrame(hdf5_file.get_node(rf"/ parameters")[:])
                self.parameters.index = self.parameters[" label"].str.decode("utf-8")
                self.parameters = self.parameters.drop(" label", axis=1)

                # Separate columns with and without "e:" and sort them
                columns = sorted(
                    [c for c in self.parameters.columns if "e:" not in c]
                ) + sorted([c for c in self.parameters.columns if "e:" in c])

                self.parameters = self.parameters[columns]
            except tables.NoSuchNodeError:
                pass

            try:
                settings = pd.DataFrame(hdf5_file.get_node(rf"/ settings")[:])
                for setting in settings.values:
                    try:
                        self.settings[setting[0].decode("utf-8")] = literal_eval(
                            setting[1].decode("utf-8")
                        )
                    except SyntaxError:
                        self.settings[setting[0].decode("utf-8")] = literal_eval(
                            setting[1].decode("utf-8").replace(" ", ",")
                        )
            except tables.NoSuchNodeError:
                pass

            self.filename = filename
            if self.open_files == []:
                self.open_files = [filename]
            else:
                if filename not in self.open_files:
                    self.open_files.append(filename)
                    self.open_files = sorted(self.open_files)

        return True

    def save(self, traces=True, parameters=True, settings=False):

        if self.filename is None:
            print("No filename provided.")
            return None

        with threading.Lock():
            with tables.open_file(self.filename, mode="a") as hdf5_file:

                if settings and self.settings:
                    try:
                        hdf5_file.remove_node(rf"/ settings", recursive=True)
                    except tables.NoSuchNodeError:
                        pass
                    df = pd.DataFrame(
                        [
                            [str(key), str(value)]
                            for key, value in self.settings.items()
                            if key[0] != "_"
                        ],
                        columns=["setting", "value"],
                    )

                    df.sort_values("setting", inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    df["setting"] = df["setting"].astype("S")
                    df["value"] = df["value"].astype("S")

                    hdf5_file.create_table(
                        r"/", " settings", obj=df.to_records(index=False)
                    )

                if traces and not self.traces.empty:

                    shared_tracenames = [
                        channel
                        for channel in self.shared_tracenames
                        if channel in self.traces.columns
                    ]

                    if shared_tracenames:
                        try:
                            hdf5_file.remove_node(rf"/ shared", recursive=True)
                        except tables.NoSuchNodeError:
                            pass

                        hdf5_file.create_table(
                            rf"/",
                            " shared",
                            obj=self.traces[shared_tracenames].to_records(index=False),
                        )

                    try:
                        hdf5_file.remove_node(rf"/{self.label}", recursive=True)
                    except tables.NoSuchNodeError:
                        pass

                    labeled_traces = sorted(
                        [
                            col
                            for col in self.traces.columns
                            if col not in shared_tracenames
                        ]
                    )

                    if labeled_traces:
                        hdf5_file.create_table(
                            rf"/",
                            self.label,
                            obj=self.traces[labeled_traces].to_records(index=False),
                        )

                if parameters and not self.parameters.empty:
                    df = self.parameters.copy().astype(float)
                    df[" label"] = df.index.values.astype("bytes")
                    df = df.reindex(
                        columns=sorted(df.columns.tolist(), key=natural_keys)
                    )
                    try:
                        hdf5_file.remove_node(rf"/ parameters", recursive=True)
                    except tables.NoSuchNodeError:
                        pass
                    hdf5_file.create_table(
                        r"/", " parameters", obj=df.to_records(index=False)
                    )
        return self.filename

    def export(self, filename=None, label=None):
        if label is None:
            label = self.label
        if filename is None:
            filename = self.filename.with_suffix(".xlsx")
        else:
            self.read(filename, label)
            filename = Path(filename).with_suffix(".xlsx")

        filename = str(filename).split(".")[0] + f"_{label}.xlsx"

        with pd.ExcelWriter(filename) as writer:
            if self.settings:
                df = pd.DataFrame(
                    [[str(key), str(value)] for key, value in self.settings.items()],
                    columns=["setting", "value"],
                )
                df.to_excel(writer, sheet_name="settings", index=False)

            pars = self.parameters.loc[["global", self.label]].copy().T
            pars.columns = ["global", "local"]
            error_rows = pars[pars.index.str.startswith("e:")]
            error_values = error_rows["local"]
            for idx in error_values.index:
                original_idx = idx[2:]  # Remove 'e:' prefix
                if original_idx in pars.index:
                    pars.at[original_idx, "error"] = error_values[idx]
            pars = pars[~pars.index.str.startswith("e:")]
            pars.to_excel(writer, sheet_name=f"parameters", index=True)

            self.traces.to_excel(writer, sheet_name="traces", index=False)

            try:
                models = self.traces[["Time (s)", "F (pN)"]].copy()
                models["Z_corrected (um)"] = (
                    self.traces["Z (um)"].to_numpy()
                    - self.calculate_drift()
                    - self.get_parameter("Zbead (um)", 0)
                )
                for modelname in [
                    "zWLC",
                    "zFiber",
                ]:
                    fit_model, params, names = prepare_fit(modelname)
                    for p in params:
                        if names[p] in self.list_parameters(format="names"):
                            params[p].value = self.get_parameter(names[p])
                        if p == "z_bead":
                            params[p].value = 0
                    models[modelname] = fit_model.eval(
                        x=self.traces[names["x"]], params=params
                    )
                models.to_excel(writer, sheet_name="models", index=False)
            except:
                pass

            print(f"Data exported to {filename}")
        return filename

    def close(self):
        self.save()
        self.open_files = [f for f in self.open_files if f != self.filename]
        if len(self.open_files) == 0:
            self.open_files = []
            self.filename = None
            self.label = None
            self.shared_tracenames = []
            self.parameters = pd.DataFrame()
            self.traces = pd.DataFrame()
        else:
            self.filename = self.open_files[0]
            self.label = self.list_labels()[0]
            self.read(self.filename, self.label)
        return

    def read_next_trace(self, selected_only=True, count_up=True):
        label_list = self.list_labels(selected_only=selected_only)
        if self.label in label_list:
            label_index = label_list.index(self.label)
        else:
            label_index = 0
        file_index = self.open_files.index(self.filename)

        label_index = label_index + 1 if count_up else label_index - 1
        if label_index < 0:
            file_index -= 1
            if file_index < 0:
                return False
        elif label_index >= len(label_list):
            file_index += 1
            if file_index >= len(self.open_files):
                return False
            label_index = 0
        return self.read(self.open_files[file_index], label_list[label_index])

    def set_parameter(self, parameter_name, value, error=None, type="local"):
        if self.label:
            if type == "global":
                self.parameters.loc["global", parameter_name] = value
                self.parameters.loc[self.label, parameter_name] = np.nan
                if "e:" + parameter_name in self.parameters.columns:
                    self.parameters.loc[self.label, "e:" + parameter_name] = np.nan
            else:
                self.parameters.loc[self.label, parameter_name] = float(value)
                if type == "local":
                    self.parameters.loc[self.label, "e:" + parameter_name] = np.nan
                if error is not None:
                    self.parameters.loc[self.label, "e:" + parameter_name] = float(
                        error
                    )
        else:
            print("No label selected.")

    def get_parameter(self, parameter_name, default=np.nan):
        try:
            value = self.parameters.loc[self.label, parameter_name]
            if np.isnan(value):
                value = self.parameters.loc["global", parameter_name]
            if np.isnan(value):
                value = self.parameters[parameter_name].mean()
            return value
        except KeyError:
            return default

    def list_parameters(self, selection=None, format="strings"):
        if selection is None:
            selection = [c for c in self.parameters.columns if "e:" not in c]
        else:
            selection = [c for c in selection if c in self.parameters.columns]

        selection = sorted(selection, key=natural_keys)
        if format in ["dict", "dictionary"]:
            parameters = {}
        else:
            parameters = []

        for col in selection:
            value = self.parameters.loc[self.label, col]
            type = "local"
            if np.isnan(value):
                value = self.parameters.loc["global", col]
                type = "global"
            if "e:" + col in self.parameters.columns:
                error = self.parameters.loc[self.label, "e:" + col]
                type = "fit"
            else:
                error = np.nan

            if format in ["strings", "string"]:
                parameters.append(round_significance(value, error, col))
                parameters = [p for p in parameters if p[-1] != "-"]
            elif format in ["names"]:
                parameters.append(col)
            elif format in ["dict", "dictionary"]:
                parameters[col] = value
            elif format in ["fit_list"]:
                parameters.append([col, round_significance(value, error), type])
            else:
                parameters.append([col, round_significance(value, error)])
        return parameters

    def list_labels(self, selected_only=False):
        if self.open_files == []:
            return []
        if self.parameters.empty:
            with tables.open_file(self.filename, mode="r") as hdf5_file:
                labels = list(hdf5_file.root._v_children.keys())
        else:
            labels = list(self.parameters.index)
        labels = [l for l in labels if l not in ["parameters", "shared", "global"]]
        if selected_only:
            labels = [l for l in labels if self.parameters.loc[l, " selected"]]
        return sorted(labels, key=natural_keys)

    def list_channels(self):
        channels = list(self.traces.columns)
        channels = [t for t in channels if t not in ["Selection", "_Z (um)"]]
        return sorted(channels, key=natural_keys)

    def clean_up_parameters(self):
        df_copy = self.parameters.copy().T
        if "global" in df_copy.index:
            df_global = df_copy.loc["global"].copy()
            df_copy.drop("global", inplace=True)
        else:
            df_global = pd.DataFrame(columns=df_copy.columns, index=["global"])
            df_global["global"] = np.nan
        for col in df_copy.columns:
            if len(df_copy[col].unique()) == 1:
                df_global.loc["global", col] = df_copy[col].mean()
                df_copy[col] = np.nan
        self.parameters = pd.concat([df_global, df_copy], ignore_index=False).T.drop(
            "global"
        )
        return

    def calculate_drift(self):
        model, params, names = prepare_fit("ExponentialDrift")
        for p in params:
            if names[p] in self.list_parameters(format="names"):
                params[p].value = self.get_parameter(names[p])
        drift = model.eval(x=self.traces["Time (s)"].copy(), params=params)
        return drift

    def calc_free_DNA(self):
        f = self.traces["F (pN)"].copy().values
        z = self.traces["Z (um)"].copy().values
        z -= self.calculate_drift() + self.get_parameter("Zbead (um)", 0)
        L0 = self.get_parameter("L (bp)", 1)
        A = self.get_parameter("A (nm)", 50)
        S = self.get_parameter("S (pN)", 1000)
        self.traces["L (bp)"] = fs.calc_free_DNA(z, f, L0, A, S) / (f > 0.5)

    def select_points(self, criteria):
        # criteria = [{"Parameter": "Z (um)", "min": "", "max": "", "invert": False}]
        try:
            selection = np.ones_like(self.traces["Selection"])
        except KeyError:
            selection = np.ones_like(self.traces[self.list_channels()[0]])

        for c in criteria:
            if c["min"] != "" or c["max"] != "":
                c["min"] = -np.inf if c["min"] == "" else float(c["min"])
                c["max"] = np.inf if c["max"] == "" else float(c["max"])
                trace = self.traces[c["channel"]].copy().values
                if c["channel"] == "Z (um)":
                    trace -= self.calculate_drift() + self.get_parameter(
                        "Zbead (um)", default=0
                    )

                if c["derivative"]:
                    trace = np.diff(trace, prepend=np.nan)
                selection *= (trace > c["min"]) & (trace < c["max"])
                if c["invert"]:
                    selection = 1 - selection

        self.traces["Selection"] = selection
        self.save(parameters=False)

    def select_trace(self, criteria):
        total_selected = 1
        for c in criteria:
            selected = 1
            par = self.parameters.loc[self.label, c["Parameter"]]
            try:
                selected *= par > float(c["min"])
            except ValueError:
                pass
            try:
                selected *= par < float(c["max"])
            except ValueError:
                pass
            if c["invert"] == True:
                selected = 1 - selected
            total_selected *= selected
        self.set_parameter(" selected", total_selected)
        self.save(traces=False)
        return total_selected

    def fit(self, task, parameters=None):
        model, params, names = prepare_fit(task, parameters)

        x_data = self.traces[names["x"]].copy()
        y_data = self.traces[names["y"]].copy()

        if task == "ExponentialDrift":
            # only lowest forces and sufficient amplitude
            selection = self.traces["Shift (mm)"].values >= np.percentile(
                self.traces["Shift (mm)"], 90
            )
            selection *= (
                self.traces["A (a.u.)"]
                >= np.percentile(self.traces["A (a.u.)"][: len(selection) // 10], 95)
                / 10
            )
            self.traces["Selection"] = selection
            self.save(parameters=False)
        elif task == "zMagnetForce":
            # only the largest extension and sufficient amplitude
            y_data -= self.calculate_drift()
            selection = y_data > self.get_parameter("dZ (um)") * 0.98
            selection *= (
                self.traces["A (a.u.)"]
                >= np.percentile(self.traces["A (a.u.)"][: len(selection) // 10], 90)
                / 10
            )
            self.traces["Selection"] = selection
            self.save(parameters=False)
        else:
            selection = self.traces["Selection"]
            if names["y"] == "Z (um)":
                y_data -= self.calculate_drift() + self.get_parameter("Zbead (um)", 0)

        # Ensure data arrays are not empty and have matching lengths
        if len(x_data) == 0 or len(y_data) == 0 or len(selection) == 0:
            raise ValueError("Data arrays must not be empty")

        result = apply_fit(
            task,
            x_data,
            y_data,
            selection,
            params,
        )

        if result.success:
            if task == "ExponentialDrift":
                # Calculate bead centers X0 and Y0
                for trace in ["X (um)", "Y (um)"]:
                    self.set_parameter(
                        trace[0] + "0 (um)",
                        np.mean(self.traces[trace][selection]),
                        np.std(self.traces[trace][selection]),
                    )

                # take bottom 5% of the Z data as reference Z0
                corrected_z = self.traces["Z (um)"].copy() - result.eval(x=x_data)
                offset = np.percentile(corrected_z[selection], 5)
                result.params["Z0"].value += offset
                corrected_z -= offset

                # take top 10% of the Z data as extension dZ
                selection = self.traces["Shift (mm)"].values <= np.percentile(
                    self.traces["Shift (mm)"], 10
                )
                selection *= (
                    self.traces["A (a.u.)"]
                    >= np.percentile(
                        self.traces["A (a.u.)"][: len(selection) // 10], 90
                    )
                    / 10
                )
                self.set_parameter(
                    "dZ (um)",
                    np.mean(corrected_z[selection]),
                    np.std(corrected_z[selection]),
                )

            for p in result.params:
                if params[p].vary:
                    fit_type = "fit"
                elif params[p].brute_step > 0.5:
                    fit_type = "global"
                else:
                    fit_type = "local"
                self.set_parameter(
                    names[p],
                    result.params[p].value,
                    result.params[p].stderr,
                    type=fit_type,
                )
            self.save(traces=False)
        return result

    def read_parameter_table(self):
        parameters = pd.DataFrame()
        for filename in self.open_files:
            with tables.open_file(filename, mode="r") as hdf5_file:
                df = pd.DataFrame(hdf5_file.get_node(rf"/ parameters")[:])
                df[" label"] = df[" label"].str.decode("utf-8")
                df["filename"] = filename
                df = df[df[" label"] != "global"]
                cols = ["filename", " label", " selected"]
                cols = cols + [c for c in df.columns if c not in cols]
                df = df[cols]
                parameters = pd.concat([parameters, df], ignore_index=True)
        return parameters

    def set_settings(self, settings_new):
        rois = settings_new.rois
        self.settings = settings_new.to_dict()
        if rois is not None:
            for label, roi in enumerate(rois):
                self.label = str(label)
                self.set_parameter("X0 (pix)", roi[0], type="local")
                self.set_parameter("Y0 (pix)", roi[0], type="local")
        return


def import_tdms(filename, data=None):
    filename = Path(filename).with_suffix(".tdms")
    tdms_file = TdmsFile(filename)
    old_traces = tdms_file.as_dataframe()
    labels = sorted(
        list(
            set(
                [
                    re.findall(r"\d+", trace[0])[0]
                    for trace in old_traces.items()
                    if re.findall(r"\d+", trace[0])
                ]
            )
        ),
        key=natural_keys,
    )
    new_filename = Path(filename).with_suffix(".hdf")
    if data is None:
        data = hdf_data()
    else:
        data.traces = pd.DataFrame()
        data.parameters = pd.DataFrame()

    data.open_files.append(new_filename)
    data.filename = Path(filename).with_suffix(".hdf")

    data.parameters = pd.DataFrame(
        index=["global"] + labels, data={" selected": [1] * (1 + len(labels))}
    )
    data.save(traces=False)

    data.shared_tracenames = ["Time (s)", "Shift (mm)", "F (pN)", "Rotation (turns)"]
    for channel in data.shared_tracenames:
        old_channel = rf"/'Tracking data'/'{channel}'"
        if old_channel in old_traces.columns:
            data.traces[channel] = old_traces[old_channel].values
    data.traces["Selection"] = 1.0

    tracenames = ["X# (um)", "Y# (um)", "Z# (um)", "A# (a.u.)"]
    for label in tqdm(labels):
        for trace in tracenames:
            old_channel = rf"/'Tracking data'/'{trace.replace('#', label)}'"
            if old_channel in old_traces.columns:
                data.traces[trace.replace("#", "")] = old_traces[old_channel].values
        data.traces = data.traces[sorted(data.traces.columns, key=natural_keys)]
        data.label = label
        data.save(parameters=False)
    return data


def create_hdf(settings, stepper_df, tracker_df):
    data = hdf_data(Path(settings._filename).with_suffix(".hdf"))
    data.set_settings(settings)
    data.save(settings=True)

    if tracker_df is not None:
        time = tracker_df["Frame"].values
        time -= time[0]
        time *= settings.exposure_time__us * 1e-6
        data.traces["Time (s)"] = time

        data.shared_tracenames = ["Time (s)"]

        stepper_df = stepper_df.loc[:, stepper_df.nunique() > 1]
        for col in stepper_df.columns[1:]:
            data.traces[col] = np.interp(
                time, stepper_df["Time (s)"], stepper_df[col].values
            )
            data.shared_tracenames.append(col)

        data.save(settings=True)

        tracker_df.drop("Frame", axis=1, inplace=True)
        labels = set([trace.split(" ")[0][1:] for trace in tracker_df.columns])
        channels = set([re.sub(r"\d+", "", trace) for trace in tracker_df.columns])

        for label in sorted(labels):
            data.label = label
            for channel in sorted(channels):
                data.traces[channel] = tracker_df[
                    channel[0] + label + channel[1:]
                ].values
            data.save()

    return data.filename


if __name__ == "__main__":
    # filename = rf"data\data_031.hdf"
    # label = "0"
    # data = hdf_data()
    # data.read(filename, label)
    if False:
        df = data.read_parameter_table()
        sns.pairplot(df, hue="filename", vars=[" selected", "L (bp)", "S (pN)"])
        plt.show()

    if False:  # test convert
        new_file = import_tdms(Path(r"data\data_042.tdms"))
        print(new_file.list_parameters)

        breakpoint
        # hdf_data.convert(filename, None)
        # data = hdf_data(filename)
        # data.read(filename, label)
        # print(data.list_labels(selected_only=False))
    if False:  # test read_next_trace
        filenames = [
            rf"data\data_014.hdf",
            rf"data\data_013.hdf",
            rf"data\data_013.hdf",
        ]
        data = hdf_data()
        for f in filenames:
            data.read(f, "49")

        for i in range(5):
            data.read_next_trace(selected_only=False, count_up=True)
            # data.read(f, str(i))
            print(data.filename, data.label)
    if False:  # test save
        data = hdf_data(filename, label)
        data.parameters[" selected"] = 1
        data.save()
    if False:  # test model
        data = hdf_data(filename, label)
        model, params, names = prepare_fit("ExponentialDrift")
        # print(model.param_names)
        # print(params)
        # print(names)
        for p in params:
            if names[p] in data.list_parameters(format="names"):
                params[p].value = data.get_parameter(names[p])
        params.pretty_print()
        model = model.eval(x=data.traces[names["x"]], params=params)
        plt.plot(data.traces["Time (s)"], data.traces["Z (um)"], "o", fillstyle="none")
        plt.plot(data.traces["Time (s)"], model, "-", color="k")
        plt.show()
        debug(list(data.traces.columns))
    if False:  # test free DNA
        from scipy.optimize import curve_fit
        from scipy.special import erf
        from scipy.ndimage import median_filter

        data = hdf_data(filename, label)
        t = data.traces["Time (s)"]
        z = data.traces["Z (um)"] - data.calculate_drift()
        f = data.traces["F (pN)"]
        amp = data.traces["A (a.u.)"]

        selection = np.diff(f, prepend=0) < 0
        selection *= f > 12

        data.set_parameter("L (bp)", 5500, type="global")
        L = data.get_parameter("L (bp)")
        A = data.get_parameter("A (pN)", 50)
        S = data.get_parameter("S (pN)", 1000)
        z_bead, _ = fs.fit_zbead(f, z, L, A, S)
        z -= z_bead

        plt.plot(z, f, "o", fillstyle="none", color="lightgrey")
        plt.plot(z[selection], f[selection], "o", fillstyle="none", color="blue")
        plt.plot(fs.zWLC(f, L, A, S, 0), f, "-", color="k")
        plt.show()

    if False:
        selection *= amp > np.percentile(amp, 90) / 5
        wlc = fs.zWLC(f, L, A, S, 0)

        plt.scatter(z, f, c=selection, cmap="viridis")
        plt.plot(wlc, f, "-", color="k")
        plt.show()

        bins = np.linspace(-1.4, 0, 275)
        shift, bins = np.histogram(z[selection] - wlc[selection], bins=bins)
        bins = bins[:-1] + (bins[1] - bins[0]) / 2

        def skewed_gaussian(x, a, x0, sigma, alpha):
            return (
                a
                * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
                * (1 + erf(alpha * (x - x0) / (sigma * np.sqrt(2))))
            )

        initial_guess = [max(shift), bins[np.argmax(shift)], 0.05, 1]
        popt, cov = curve_fit(
            skewed_gaussian, bins, shift, p0=initial_guess, sigma=np.sqrt(1 / shift)
        )
        data.set_parameter("Zbead (um)", popt[1], np.sqrt(cov[1, 1]), type="fit")
        data.set_parameter("L (bp)", L, type="global")
        data.set_parameter("A (nm)", A, type="global")
        data.set_parameter("S (pN)", S, type="global")

        # The refined peak location is the mean of the Gaussian function
        y_fit = skewed_gaussian(bins, *popt)
        for i, p in enumerate(popt):
            print(round_significance(p, np.sqrt(cov[i, i])))
        plt.plot(bins, shift, "o", color="blue", fillstyle="none")
        plt.plot(bins + 0.01, y_fit, "-", color="k")
        plt.show()

        L2 = 80
        n4 = 50
        # z = median_filter(z, 15)
        selection *= (
            np.diff(
                f,
                prepend=0,
            )
            > 0
        )

        z -= data.get_parameter("Zbead (um)")

        # plt.plot(z, f, 'o', fillstyle='none', color='lightgrey')
        # plt.plot(z[selection], f[selection],
        #          'o', fillstyle='none', color='blue')
        # for i in range(n4):
        #     plt.plot(wlc*(L-i*L2)/L, f, '-', color='k')
        plt.plot(t, z, "o", fillstyle="none", color="lightgrey")
        plt.plot(t[selection], z[selection], "o", fillstyle="none", color="blue")
        for i in range(n4):
            plt.plot(t, wlc * (L - i * L2) / L, "-", color="k")
        plt.show()

        Lfree = L * z / wlc
        # Lfree = median_filter(Lfree, 30)

        # plt.plot(t, Lfree, 'o', fillstyle='none', color='lightgrey')
        # plt.plot(t[selection], Lfree[selection],
        #          'o', fillstyle='none', color='blue')
        # plt.show()

        # plt.plot(t, Lfree, 'o', fillstyle='none', color='lightgrey')
        # plt.plot(t[selection], Lfree[selection],
        #          'o', fillstyle='none', color='blue')
        # plt.show()
        L2 /= 2

        bins = np.linspace(L - 2 * n4 * L2, L, 2 * n4) + L2 / 2
        Lfree, bins = np.histogram(Lfree[selection], bins=bins)
        bins = bins[:-1] + (bins[1] - bins[0]) / 2

        n4_guess = np.argmax(Lfree)
        print(n4_guess, bins[n4_guess], Lfree[n4_guess])

        plt.plot(bins, Lfree, "-o", fillstyle="none", color="blue")
        plt.show()

    if False:
        filename = rf"data\data_031.hdf"
        label = "14"

        data = hdf_data(filename, label)
        label = np.random.choice(data.list_labels())

        if True:
            import_tdms(filename)
            data = hdf_data(filename, label)
            model, params, names = prepare_fit("ExponentialDrift")
            params["Z0"].vary = True
            params["dZ_dt"].vary = True
            fit = data.fit("ExponentialDrift", params)

        model, params, names = prepare_fit("zMagnetForce")
        params["L"].value = 5000
        params["z_bead"].vary = True
        params["dt"].vary = True

        init = model.eval(params, x=data.traces["Shift (mm)"])
        fit = data.fit("zMagnetForce", params)

        for p in fit.result.params:
            print(
                names[p],
                round_significance(
                    fit.result.params[p].value, fit.result.params[p].stderr
                ),
            )

        selection = data.traces["Selection"]
        t = data.traces["Time (s)"].values
        z = data.traces["Z (um)"].copy().values
        z -= data.calculate_drift()

        plt.plot(
            t,
            z,
            color="lightgrey",
            linestyle="none",
            marker="o",
            markerfacecolor="none",
        )
        plt.plot(
            t,
            z / selection,
            color="blue",
            linestyle="none",
            marker="o",
            markerfacecolor="none",
        )

        plt.plot(t, init, linestyle="dotted", color="k")
        plt.plot(t, fit.eval(x=data.traces["Shift (mm)"]), color="k")
        plt.title(f"{data.filename.stem}:{data.label}")
        plt.show()

    if True:
        filename = rf"data_001.bin"
        filename = r"d:\users\Administrator\data\20250108\data_065.hdf"
        data = hdf_data(filename)
        # ic(data.traces, data.list_labels(), data.list_channels())
        if data.list_channels():
            data.read(filename, "0")
            t = data.traces["Time (s)"]
            z = data.traces["Z (um)"]
        plt.plot(t, z, "o", fillstyle="none")
        plt.show()

        # print(create_hdf(None, pd.DataFrame(), pd.DataFrame()))

        # data.read(filename, "0")
        # ic(data.list_channels())
        # ic(data.settings["roi_size (pix)"])

        # # ic(data.settings)
        # # ic(data.traces)
        # # ic(data.parameters)

        # # label = np.random.choice(data.list_labels())
        # # data.read(filename, label)
        # data.export(filename, 0)
