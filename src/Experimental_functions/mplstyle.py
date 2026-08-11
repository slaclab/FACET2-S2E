import matplotlib as mpl
from cycler import cycler

mpl.rcParams["font.size"] = 8
mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=["#0077BB", "#CC3311", "#009988", "#EE7733", "#33BBEE", "#EE3377", "#BBBBBB"]
)
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelpad"] = 2.0
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.bottom"] = True
mpl.rcParams["xtick.labeltop"] = False
mpl.rcParams["xtick.labelbottom"] = True
mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["xtick.major.width"] = 0.5
mpl.rcParams["xtick.major.pad"] = 2
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["xtick.labelsize"] = 6
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.major.top"] = True
mpl.rcParams["xtick.major.bottom"] = True
mpl.rcParams["ytick.left"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.labelleft"] = True
mpl.rcParams["ytick.labelright"] = False
mpl.rcParams["ytick.major.size"] = 2
mpl.rcParams["ytick.major.width"] = 0.5
mpl.rcParams["ytick.major.pad"] = 2
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["ytick.labelsize"] = 6
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.major.left"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["errorbar.capsize"] = 1.0
mpl.rcParams["lines.markeredgewidth"] = 0.5
mpl.rcParams["legend.frameon"] = False
