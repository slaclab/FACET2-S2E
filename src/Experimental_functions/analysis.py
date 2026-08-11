import numpy as np
from scipy import signal
from scipy import interpolate
from scipy import optimize


def fwhm(
    y: np.ndarray, x: np.ndarray = None, dx: float = None
) -> tuple[float, float, float, float]:
    """Finds the full width at half max of a signal with a single peak.

    If the signal has multiple peaks, this will return the FWHM of the tallest peak.
    By default the function returns the width in the number of samples.
    Passing x will return the width in the same units as x, can handle signals with non-uniform spacing.
    Passing dx will return the units in the same units as dx if the samples are uniformly spaced.

    Args:
        y: The representing the signal with the peak.
        x: Array representing the x axis (time, wavelength, etc.) of the signal.
        dx: spacing between the sample points of the signal.

    Returns:
        fwhm: The full width at half max of the signal.
        height: The half height of the peak.
        start: The position where the left side of the peak crosses the halfway point.
        end: The position where the right side of the peak crosses the halfway point.
    """
    peaks = signal.find_peaks(y, height=0.9999 * np.max(y))[0]
    fwhms, heights, lefts, rights = signal.peak_widths(y, peaks, rel_height=0.5)
    fwhm = fwhms[0]
    height = heights[0]
    start = lefts[0]
    end = rights[0]
    if x is not None:
        N = len(x)
        fx = interpolate.interp1d(np.arange(0, N, 1), x)
        start = fx(start)
        end = fx(end)
        fwhm = end - start
    elif dx is not None:
        fwhm *= dx
        start *= dx
        end *= dx
    return fwhm, height, start, end


def fw_e2(
    y: np.ndarray, x: np.ndarray = None, dx: float = None
) -> tuple[float, float, float, float]:
    """Finds the 1/e^2 width of a signal with a single peak.

    If the signal has multiple peaks, this will return the 1/e^2 of the tallest peak.
    By default the function returns the width in the number of samples.
    Passing x will return the width in the same units as x, can handle signals with non-uniform spacing.
    Passing dx will return the units in the same units as dx if the samples are uniformly spaced.

    Args:
        y: The representing the signal with the peak.
        x: Array representing the x axis (time, wavelength, etc.) of the signal.
        dx: spacing between the sample points of the signal.

    Returns:
        width: The 1/e^2 of the signal.
        height: The 1/e^2 height of the peak.
        start: The position where the left side of the peak crosses the 1/e^2 point.
        end: The position where the right side of the peak crosses the 1/e^2 point.
    """
    peaks = signal.find_peaks(y, height=0.9999 * np.max(y))[0]
    widths, heights, lefts, rights = signal.peak_widths(
        y, peaks, rel_height=1 - 1 / np.e**2
    )
    width = widths[0]
    height = heights[0]
    start = lefts[0]
    end = rights[0]
    if x is not None:
        N = len(x)
        fx = interpolate.interp1d(np.arange(0, N, 1), x)
        start = fx(start)
        end = fx(end)
        width = end - start
    elif dx is not None:
        width *= dx
        start *= dx
        end *= dx
    return width, height, start, end


def fit_gaussian(x, y, p0=(0.0, 1.0, 1.0, 0.0), sigma=None, maxfev=1000):
    """Fits a Gaussian to a set of data.

    Args:
        x: Array representing the x axis (time, wavelength, etc.) of the signal.
        y: The representing the signal with the peak.
        p0: Initial guess for the fit (x_0, I_0, sigma, y_0).
        sigma: Error of each point.

    Returns:
        f: Function that was fit to the signal, call as f(x, *popt).
        popt: Fit parameters (x_0, I_0, sigma, y_0).
    """

    def fg(x, x_0, I_0, sigma, y_0):
        return I_0 * np.exp(-0.5 * ((x - x_0) / sigma) ** 2) + y_0

    px, pcov = optimize.curve_fit(fg, x, y, p0=p0, sigma=sigma, maxfev=maxfev)
    return fg, px


def fit_elliptical_gaussian(image, p0=(1500.0, 1500.0, 1.0, 2000.0, 2000.0, 0.0)):
    """Fits an elliptical Gaussian to an image like object.

    Is designed for fitting to Gaussian beams, returns the Gaussian beam spot size (also called beam radius).

    Args:
        image: Image like object, must have an atribute data that is a 2D array, and attributes x and y which are 1D arrays.
        p0: Initial guess for the fit (x_0, y_0, I_0, w_0x, w_0y).

    Returns:
        f: Function that was fit to the image, call as f([x, y], *popt).
        popt: Fit parameters (x_0, y_0, I_0, w_0x, w_0y).
    """

    def f2eg(xy, x_0, y_0, I_0, w_0x, w_0y, theta_0):
        x = xy[0]
        y = xy[1]
        xp = (x - x_0) * np.cos(theta_0) + (y - y_0) * np.sin(theta_0)
        yp = -(x - x_0) * np.sin(theta_0) + (y - y_0) * np.cos(theta_0)
        return I_0 * np.exp(-2 * ((xp) / w_0x) ** 2 - 2 * ((yp) / w_0y) ** 2)

    X, Y = np.meshgrid(image.x, image.y)
    Z = image.data
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = Z.ravel()
    valid = ~np.isnan(ydata)
    popt, pcov = optimize.curve_fit(f2eg, xdata[:, valid], ydata[valid], p0=p0)
    return f2eg, popt


def fit_superGaussian(x, y, p0=(60, 2000, 20, 4)):
    """Fits an super Gaussian to an image like object.

    Is designed for fitting to lineouts of laser beams, returns the Gaussian beam spot size (also called beam radius).

    Args:
        x: x values of the data to be fit.
        y: y values of the data to be fit.
        p0: Initial guess for the fit (x_0, I_0, w_0, m).

    Returns:
        f: Function that was fit to the image, call as f([x, y], *popt).
        popt: Fit parameters (x_0, I_0, w_0, m).
    """

    def fsg(x, x_0, I_0, w_0, m):
        return I_0 * np.exp(-2 * (((x - x_0) / w_0) ** 2) ** m)

    px, pcov = optimize.curve_fit(fsg, x, y, p0=p0)
    return fsg, px


def fit_2D_superGaussian(image, p0=(60, 35, 2200, 15, 4)):
    """Fits a 2D super-Gaussian to an image like object.

    Is designed for fitting to Gaussian beams, returns the Gaussian beam spot size (also called beam radius).

    Args:
        image: Image like object, must have an atribute data that is a 2D array, and attributes x and y which are 1D arrays.
        p0: Initial guess for the fit (x_0, y_0, I_0, w_0, m).

    Returns:
        f: Function that was fit to the image, call as f([x, y], *popt).
        popt: Fit parameters (x_0, y_0, I_0, w_0, m).
    """

    def f2sg(xy, x_0, y_0, I_0, w_0, m):
        x = xy[0]
        y = xy[1]
        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        return I_0 * np.exp(-2 * ((r / w_0) ** 2) ** m)

    X, Y = np.meshgrid(image.x, image.y)
    Z = image.data
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = Z.ravel()
    popt, pcov = optimize.curve_fit(f2sg, xdata, ydata, p0=p0)
    return f2sg, popt


def find_roots(x, y):
    inds = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0] + 1
    N = len(inds)
    if N == 0:
        raise RuntimeError("No roots found.")
    roots = np.zeros(N)
    for i in range(N):
        ind = inds[i]
        x1 = x[ind - 1]
        x2 = x[ind]
        y1 = y[ind - 1]
        y2 = y[ind]
        roots[i] = -y1 * ((x2 - x1) / (y2 - y1)) + x1
    return roots


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
