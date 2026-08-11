import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.io
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit

from . import analysis as an

cal = {}
__ElogBaseURL = "http://mccas0.slac.stanford.edu/u1/facet/matlab/data/"


def set_calibration(cam_name: str, calibration: float):
    """Adds/updates the calibration list entry for the given camera.

    Args:
        cam_name: Name of the camera, must match the name in the metadata.
        calibration: Pixel calibration in mm/px.
    """
    cal[str(cam_name)] = calibration


def orientImage(
    data: np.ndarray, XOrient: str, YOrient: str, isRotated: bool = None
) -> np.ndarray:
    """Flips/rotates a raw image to match the orientation shown by the profMon.

    Args:
        data: Array with raw image data.
        XOrient: Flips the image in the x-direction, from the PV. Either "Positive" or "Negative".
        YOrient: Flips the image in the y-direction, from the PV. Either "Positive" or "Negative".
        isRotated: Rotates the image by transposing.

    Returns:
        Array with image data after orienting.
    """
    newData = data
    if XOrient is not None and YOrient is not None:
        if XOrient == "Positive" and YOrient == "Positive":
            newData = data
        elif XOrient == "Negative" and YOrient == "Negative":
            newData = np.rot90(data, k=2)
        elif XOrient == "Negative" and YOrient == "Positive":
            newData = np.flip(data, axis=1)
        elif XOrient == "Positive" and YOrient == "Negative":
            newData = np.flip(data, axis=0)
        else:
            raise RuntimeError("Image orientations does not match expected values")

    if isRotated == 1:
        newData = np.transpose(newData)
    return newData


def specialFlips(camera: str, data: np.ndarray, isRotated=None) -> np.ndarray:
    """Rotates a raw image to match the profMon orientation if the isRotated PV is not present.

    At one point in time, the rotation of certain cameras as hardcoded in to the profMon GUI.
    Back in these dark days, the isRotated PV did not exist and cameras had to be rotated based
    on the names.

    Args:
        camera: Name of the camera, must match the name in the metadata.
        data: Array with raw image data.
        isRotated: Rotates the image by transposing.

    Returns:
        Array with image data after orienting.
    """
    if isRotated is not None:
        return data
    if camera == "LFOV" or camera == "GAMMA1" or camera == "EDC_SCREEN":
        data = np.transpose(data)
    return data


def ElogImage(path: str | os.PathLike, cam: str, date: str, time: str):
    """Loads the image matching the given name from the Elog.

    Args:
        path: Path describing where image should be downloaded to.
        cam: Camera designator to load data, e.g. 'LI20_111'.
        date: Date string for when the image was taken in YYYY-MM-DD: '2022-10-27'.
        time: Time stamp for the requested image in HHMMSS: '175546'.

    Raises:
        RuntimeError: An error occurs while trying to download the image, most likely VPN problems.
    """
    # Check if the image already exists
    filename = "ProfMon-CAMR_{}-{}-{}.mat".format(cam, date, time)
    filepath = os.path.join(path, filename)
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(filepath):
        return

    # Try and connect to the SLAC servers, throw an error if they aren't available
    year, month, day = parseDate(date)
    url = __ElogBaseURL + "{}/{}-{}/{}-{}-{}/{}".format(
        year, year, month, year, month, day, filename
    )
    try:
        response = requests.get(url)
        open(filepath, "wb").write(response.content)
    except:
        raise RuntimeError(
            "The file was not already downloaded and something went wrong trying to download it, are you on the VPN?"
        )


def parseDate(date: str) -> tuple[str, str, str]:
    """Returns the year, month, and day from the passed date string.

    Args:
        date: Date string, format: '2022-10-27'.

    Returns:
        year: The year from the date string, format YYYY.
        month: The month from the date string, format MM.
        day: The day from the date string, format DD.
    """
    year, month, day = date.split("-")
    return year, month, day


class IMAGE:
    """An image taken by one of the FACET-II camera.

    This class supports simple image processing and center finding in images acquired
    from cameras at FACET. Simple plotting operations are provided for quick visualizations
    of the data.

    Attributes:
        camera: Name of the camera.
        meta: Metadata associated with the image.
        image: Pillow image object for the image.
        data: Image data as a numpy array.
        cal: Calibration factor (in mm/px).
        width: Width of the image in pixels.
        height: Height of the image in pixels.
        xp: Array of x pixel coordinates.
        yp: Array of y pixel coordinates.
        x: Array of x coordinates in millimeters.
        y: Array of y coordinates in millimeters.
        center: Center coordinates of the image (x, y) in pixels. Will be None if center
            finding has not been run.
        cropBox: Coordinates of the cropping box (left, upper, right, lower). Will be None
            if the image has not been cropped.
        fit: Parameters of the fitted function if center finding has been accomplished using
            fitting, None otherwise.
    """

    # Initialization functions ------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, camera: str, **kwargs):
        self.camera = str(camera)
        self.meta = self._get_image_meta()
        self.image = self._load_image()
        self.data = np.array(self.image, dtype="float")
        if self.camera not in cal:
            set_calibration(self.camera, 1)
            print(
                "Warning, calibration was not defined for camera {}, defaulting to 1.".format(
                    self.camera
                )
            )
        self.cal = cal[self.camera]  # In mm/px
        self.width = self.image.width
        self.height = self.image.height
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp
        self.center = None
        self.cropBox = None
        self._check_image()

    def _load_image(self) -> object:
        """Loads an image from a given data set.

        Overwrite to define how to load an image.

        Returns:
            image: Pillow image object for the image.
        """
        image = Image.new("I", (1292, 964))
        return image

    def _get_image_meta(self):
        """Returns the meta data dictionary for the image object.

        Overwrite to define how to load the metadata for an image.
        """
        meta = {}
        return meta

    def _check_image(self):
        """Verifys meta data for the image is consistent with image data."""
        # if "Width" in self.meta and self.meta["Width"] != self.width:
        #     print(
        #         "Image meta data width {:d} does not match image width {:d}".format(
        #             self.meta["Width"], self.width
        #         )
        #     )
        # if "Height" in self.meta and self.meta["Height"] != self.height:
        #     print(
        #         "Image meta data height {:d} does not match image height {:d}".format(
        #             self.meta["Height"], self.height
        #         )
        #     )

    def refresh_calibration(self):
        """Updates the camera calibration if it has been changed."""
        self.cal = cal[self.camera]
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp

    # Modification functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def rotate(self, angle: float):
        """Rotates the image.

        Args:
            angle: Angle to rotate the image by, in deg.
        """
        self.image = self.image.rotate(angle)
        self.data = np.array(self.image, dtype="float")

    def crop(self, box: tuple):
        """Crops the image.

        Args:
            box: The crop rectangle, as a (left, upper, right, lower) tuple.
        """
        self.image = self.image.crop(box)
        self.data = np.array(self.image, dtype="float")
        self.width = self.image.width
        self.height = self.image.height
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp
        self.center = None
        self.cropBox = box

    def center_image(self, strategy: str, o: int, **kwargs):
        """Centers the image by non-uniformaly padding it. Meta will no longer match class attributes.

        Args:
            strategy: Strategy for centering the image (e.g., 'cm', 'max', 'mask', 'fit', 'projFit', 'external').
            o: Padding on each side of the returned array, in pixels.
            **kwargs: Additional arguments for specific centering strategies, see calculate_center.
        """
        cen_image, center = self.get_center_image(strategy, o, **kwargs)
        self.data = cen_image
        self.image = Image.fromarray(cen_image)
        self.width = self.image.width
        self.height = self.image.height
        self.center = (self.width / 2, self.height / 2)
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal * self.xp
        self.y = self.cal * self.yp

    def subtract_background_noise(self, threshold: float):
        """Subtracts background noise from the image data.

        Background noise is determined by averaging the value of all pixels below the given threshold.

        Args:
            threshold: Threshold value for determining background pixels.
        """
        data = self.data
        sel = data < threshold
        background = data[sel]
        avg_background = np.average(background)
        self.data = data - avg_background
        self.image = Image.fromarray(self.data)

    def subtract_background(self, background: np.ndarray):
        """Subtracts a specified background from the image data.

        Args:
            background: Background image data to subtract.
        """
        data = self.data
        self.data = data - background
        self.image = Image.fromarray(self.data)

    def median_filter(self, size: int = 3):
        """Applies a median filter to the image data.

        Args:
            size: Size of the median filter.
        """
        self.data = ndimage.median_filter(self.data, size)
        self.image = Image.fromarray(self.data)

    # Calculation functions ---------------------------------------------------
    # --------------------------------------------------------------------------
    def calculate_center(
        self,
        strategy: str = "cm",
        size: int = 3,
        threshold: int = 12,
        f=None,
        p0: tuple = None,
        center: np.ndarray = None,
        maxfev: int = 1000,
    ):
        """Calculates the pixel location of the center of the image.

        Args:
            strategy: Select the technique to use to find the center of the image.
                'cm' - the image center is found by taking the cm of the image.
                'max' - the image center is found by median filtering then taking the max.
                'mask' - a mask is formed from all pixels with values greather than a threshold.
                    The image center is the centroid of the mask.
                'fit' - fit a function to the image to find the center.
                'projFit' - Project then fit a Guassian.
                'external' - pass in the location of the mask center.
            size: When strategy='max', size of the median filter
            threshold: When strategy='mask', threshold is used to create the mask.
            f: When strategy='f', function to fit to the data. The first two free parameters
                should be the x and y positions of the center. It should accept as the first
                arguments (x, y).
            p0: When strategy='f', initial guesses for model parameters.
            center: When strategy='external', location of the image center.
        """
        if strategy == "cm":
            self.center = self.center_cm()
        if strategy == "max":
            self.center = self.center_max(size)
        if strategy == "mask":
            self.center = self.center_mask(threshold)
        if strategy == "fit":
            self.fit = self.center_fit(f, p0)
            self.f = f
            self.center = np.array([self.fit[0], self.fit[1]])
        if strategy == "external":
            self.center = center
        if strategy == "projFit":
            self.center = self.center_projFit(maxfev=maxfev)

    def center_cm(self):
        """Returns the center of mass of the image."""
        return np.flip(ndimage.center_of_mass(self.data))

    def center_max(self, size: int):
        """Returns the max location of the maximum value after median filtering."""
        data = ndimage.median_filter(self.data, size)
        center = np.argmax(data)
        center_y = center / self.width
        center_x = center % self.width
        return np.array([center_x, center_y])

    def center_mask(self, threshold: int):
        """Returns the centroid of a mask of the image.

        The mask is found by setting all pixels greater than threshold to 1 and all
        pixels less than threshold to 0.
        """
        mask = self.data > threshold
        return np.flip(ndimage.center_of_mass(mask))

    def center_fit(self, f, p0):
        """Returns the center of the image from fitting a function to the data."""
        X, Y = np.meshgrid(self.xp, self.yp)
        Z = self.data
        xdata = np.vstack((X.ravel(), Y.ravel()))
        ydata = Z.ravel()
        popt, pcov = curve_fit(f, xdata, ydata, p0=p0)
        return popt

    def center_projFit(self, maxfev: int):
        """Returns the center found by projecting then fitting a Gaussian."""
        xProj = np.sum(self.data, axis=0)
        yProj = np.sum(self.data, axis=1)
        A0x = np.max(xProj)
        A0y = np.max(yProj)
        x0 = self.xp[np.argmax(xProj)]
        y0 = self.yp[np.argmax(yProj)]
        fg, fitx = an.fit_gaussian(self.xp, xProj, (x0, A0x, 50, 0.0), maxfev=maxfev)
        fg, fity = an.fit_gaussian(self.yp, yProj, (y0, A0y, 50, 0.0), maxfev=maxfev)
        return np.array([fitx[0], fity[0]])

    def calculate_sum(self) -> float:
        """Returns the total sum of all the pixels."""
        return np.sum(self.data)

    def get_center_image(self, strategy: str, o: int, **kwargs):
        """Centers the image by non-uniformly padding it. Meta will no longer match class attributes.

        Args:
            strategy: Strategy for centering the image (e.g., 'cm', 'max', 'mask', 'fit', 'projFit', 'external').
            o: Padding on each side of the returned array, in pixels.
            **kwargs: Additional arguments for specific centering strategies, see calculate_center.

        Returns:
            A tuple(cen_image, center) where cen_image is the padded image with the original
            image shifted to be centered and center is a tuple(cen_x, cen_y) with the location
            of the image center in pixel coordinates.
        """
        self.calculate_center(strategy, **kwargs)
        cen = np.array([self.width / 2 + o, self.height / 2 + o], dtype="int")
        center = self.center
        if center is None:
            center = (self.width / 2, self.height / 2)
        shift = np.array(np.rint(cen - center), dtype="int")
        cen_image = np.zeros((self.height + 2 * o, self.width + 2 * o))
        start_y = shift[1]
        end_y = start_y + self.height
        start_x = shift[0]
        end_x = start_x + self.width
        cen_image[start_y:end_y, start_x:end_x] = self.data
        return cen_image, center

    # Visualization functions --------------------------------------------------
    # --------------------------------------------------------------------------
    def get_ext(self, cal: bool = True) -> np.ndarray:
        """Calculates the extent for displaying the image with imshow.

        This helper function determines the extent of the image in either calibrated units
        (millimeters) or pixel coordinates.

        Args:
            cal: If True, return extent in calibrated units. If False, return extent in pixel coordinates.

        Returns:
            Array of shape (4,) representing the extent of the image.
        """
        if cal:
            ext = self.cal * np.array([-0.5, self.width - 0.5, self.height - 0.5, -0.5])
        else:
            ext = np.array([-0.5, self.width - 0.5, self.height - 0.5, -0.5])
        return ext

    def create_fig_ax(self, cal: bool = True):
        """Creates and configures a Matplotlib figure and axis for plotting the image.

        Args:
            cal: If True, the axes will be labeled with calibrated units (millimeters).
                        If False, the axes will be labeled with pixel coordinates.

        Returns:
            A tuple containing the following elements:
                - fig: Matplotlib figure object.
                - ax: Matplotlib axes object for plotting the image.
                - ext: Array of shape (4,) representing the extent of the image.
        """
        width = 4.85
        height = 0.8 * self.height / self.width * width
        if height > 4.85:
            pass
        fig = plt.figure(figsize=(width, height), dpi=300)
        ax = plt.subplot()
        if cal:
            ax.set_xlabel(r"$x$ (mm)")
            ax.set_ylabel(r"$y$ (mm)")
            ext = self.cal * np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        else:
            ax.set_xlabel(r"$x$ (px)")
            ax.set_ylabel(r"$y$ (px)")
            ext = np.array([-0.5, self.width + 0.5, self.height + 0.5, -0.5])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        return fig, ax, ext

    def plot_image(self, cal: bool = True, cmap="inferno", metadata: bool = True):
        """Plots the image with optional metadata and calibration.

        Args:
            cal: If True, display the image with calibrated axes. If False, use pixel coordinates.
            cmap: Colormap to be used for the image.
            metadata: If True, display metadata information on the plot.

        Returns:
            A tuple containing the following elements:
                - fig: Matplotlib figure object.
                - ax: Matplotlib axes object for plotting the image.
                - im: Matplotlib imshow object for the image.
                - cb: Matplotlib colorbar object.
                - ext: Array of shape (4,) representing the extent of the image.
        """
        fig, ax, ext = self.create_fig_ax(cal)
        if metadata:
            self.plot_dataset_text(ax)
            self.plot_metadata_text(ax)
        im = ax.imshow(self.data, extent=ext, cmap=cmap)
        cb = fig.colorbar(im, aspect=30 * self.height / self.width)
        cb.set_label("Counts")
        ax.tick_params(color="w")
        ax.spines["bottom"].set_color("w")
        ax.spines["top"].set_color("w")
        ax.spines["left"].set_color("w")
        ax.spines["right"].set_color("w")
        return fig, ax, im, cb, ext

    def plot_dataset_text(self, ax):
        """Adds text at the top of the figure stating the dataset and shot number.

        Overwrite to add dataset annotations to the plot.
        """
        pass

    def plot_metadata_text(self, ax):
        """Adds text at the bottom of the figure stating image parameters.

        Overwrite to add metadata annotations to the plot.
        """
        pass

    def plot_center(self, radius: float, cal: bool = True, cmap="inferno"):
        """Plots the image with a circle representing the beam center.

        Args:
            radius: Radius of the beam circle to plot.
            cal: If True, use calibrated axes. If False, use pixel coordinates.
            cmap: Colormap to be used for the image.

        Returns:
            A tuple containing the following elements:
                - fig: Matplotlib figure object.
                - ax: Matplotlib axes object for plotting the image.
                - im: Matplotlib imshow object for the image.
                - cb: Matplotlib colorbar object.
                - ext: Array of shape (4,) representing the extent of the image.
        """
        if self.center is None:
            print(
                "The image center is None, it needs to be calculated before it can be shown."
            )
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal * cen[0], cal * cen[0]], [ext[2], ext[3]], "tab:blue")
        ax.plot([ext[0], ext[1]], [cal * cen[1], cal * cen[1]], "tab:blue")
        phi = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(
            radius * np.cos(phi) + cal * cen[0], radius * np.sin(phi) + cal * cen[1]
        )
        ax.annotate(
            "Beam center:\n({:0.3f}, {:0.3f})".format(cal * cen[0], cal * cen[1]),
            xy=(0, 1),
            xytext=(3, -19),
            xycoords="axes fraction",
            textcoords="offset points",
            color="tab:blue",
            va="top",
        )
        return fig, ax, im, cb, ext

    def plot_lineouts(self, cal: bool = True, cmap="inferno"):
        """Plot the image with lineouts through the center.

        Args:
            cal: If True, use calibrated axes. If False, use pixel coordinates.
            cmap: Colormap to be used for the image.

        Returns:
            A tuple containing the following elements:
                - fig: Matplotlib figure object.
                - ax: Matplotlib axes object for plotting the image.
                - im: Matplotlib imshow object for the image.
                - cb: Matplotlib colorbar object.
                - ext: Array of shape (4,) representing the extent of the image.
        """
        if self.center is None:
            print(
                "The image center is None, it needs to be calculated before it can be shown."
            )
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal * cen[0], cal * cen[0]], [ext[2], ext[3]], "tab:blue")
        ax.plot([ext[0], ext[1]], [cal * cen[1], cal * cen[1]], "tab:blue")
        # TODO implement the lineout bit
        return fig, ax, im, cb, ext


class Elog(IMAGE):
    def __init__(self, path, camera, date, timestamp):
        self.path = path
        self.date = date
        self.timestamp = timestamp
        super().__init__(camera)

    def _load_image(self):
        """Load an image from a given data set.

        Returns:
            image : obj
                Pillow image object for the tiff.
        """
        self.filename = "ProfMon-CAMR_{!s}-{!s}-{!s}.mat".format(
            self.camera, self.date, self.timestamp
        )
        name = os.path.join(self.path, self.filename)
        try:
            self.mat = scipy.io.loadmat(name)
            image = Image.fromarray(self.mat["data"][0][0][1])
        except (NotImplementedError, ValueError):
            self.mat = None
            self.h5 = h5py.File(name, "r")
            image = Image.fromarray(np.transpose(np.array(self.h5["data"]["img"])))
        return image

    def _get_image_meta(self):
        """Return the meta data dictionary from a pillow image object.

        Returns:
            meta : dict
                The meta data dictionary contained in the tiff image.
        """
        # You can see the name of each field in the mat at mat['data'].dtype.names
        meta = {}
        if self.mat is not None:
            mat_meta = self.mat["data"][0][0]
            raise NotImplementedError
            names = self.mat["data"].dtype.names
            ind = names.index("roiX")
            meta["Width"] = meta[ind]
            meta["Offset"] = [mat_meta[10][0][0], mat_meta[11][0][0]]
            meta["Camera"] = self.camera
            meta["Pixel"] = [mat_meta[6][0][0], mat_meta[7][0][0]]
        else:
            h5_meta = self.h5["data"]
            meta["Width"] = int(h5_meta["nCol"][0][0])
            meta["Height"] = int(h5_meta["nRow"][0][0])
            meta["roiX"] = int(h5_meta["roiX"][0][0])
            meta["roiY"] = int(h5_meta["roiY"][0][0])
            meta["Timestamp"] = h5_meta["ts"][0][0]
            meta["pulseId"] = h5_meta["pulseId"][0][0]
            meta["bitdepth"] = int(h5_meta["bitdepth"][0][0])
            meta["orientX"] = h5_meta["orientX"][0][0]
            meta["orientY"] = h5_meta["orientY"][0][0]
        return meta


class DAQ(IMAGE):
    """An image taken by one of the FACET-II camera and saved as a tiff by the DAQ."""

    def __init__(self, camera, dataset, filename, ind, step):
        self.dataset = dataset
        self.filename = filename
        self.ind = ind
        self.step = step
        super().__init__(camera)

    def _load_image(self):
        self.path = os.path.join(self.dataset.datasetPath, "images", self.camera)
        name = os.path.join(self.path, self.filename)
        image = Image.open(name)
        data = np.array(image, dtype="float")
        image = self._orientImage(data)
        return image

    def _get_image_meta(self):
        return self.dataset.metadata[self.camera]

    def _orientImage(self, data):
        if "X_ORIENT" in self.meta and "Y_ORIENT" in self.meta:
            XOrient = self.meta["X_ORIENT"]
            YOrient = self.meta["Y_ORIENT"]
            if "IS_ROTATED" in self.meta:
                isRotated = self.meta["IS_ROTATED"]
            else:
                isRotated = None
            data = orientImage(data, XOrient, YOrient, isRotated)
            data = specialFlips(self.camera, data, isRotated)
        return Image.fromarray(data)

    def plot_dataset_text(self, ax):
        """Adds text at the top of the figure stating the dataset and shot number."""
        datasetText = ax.annotate(
            "{}, Dataset: {}\n{}".format(
                self.dataset.experiment, self.dataset.number, self.camera
            ),
            xy=(0, 1),
            xytext=(3, -3),
            xycoords="axes fraction",
            textcoords="offset points",
            color="w",
            va="top",
        )
        stepText = ax.annotate(
            "Shot: {:04d}\nStep: {:02d}".format(self.ind, self.step),
            xy=(1, 1),
            xytext=(-38, -3),
            xycoords="axes fraction",
            textcoords="offset points",
            color="w",
            va="top",
        )
        ax.stepText = stepText
        ax.datasetText = datasetText

    def plot_metadata_text(self, ax):
        """Adds text at the bottom of the figure stating image parameters."""
        s = mpl.rcParams["xtick.labelsize"]
        sx = self.meta["MinX_RBV"]
        sy = self.meta["MinY_RBV"]
        if self.cropBox is not None:
            sx += self.cropBox[0]
            sy += self.cropBox[1]
        exp = self.meta["AcquireTime_RBV"]
        gain = self.meta["Gain_RBV"]
        metadataText = ax.annotate(
            "Size:  {:4d}, {:4d}\nStart: {:4d}, {:4d}".format(
                self.width, self.height, sx, sy
            ),
            xy=(0, 0),
            xytext=(3, 3),
            xycoords="axes fraction",
            textcoords="offset points",
            color="w",
            size=s,
            transform=ax.transAxes,
        )
        ax.metadataText = metadataText
        # ax.text(
        #     0.84,
        #     y + dy,
        #     "Exp:  {:0.2f}ms".format(exp * 1e3),
        #     color="w",
        #     size=s,
        #     transform=ax.transAxes,
        # )
        # ax.text(
        #     0.84,
        #     y,
        #     "Gain: {:0.2f}".format(gain),
        #     color="w",
        #     size=s,
        #     transform=ax.transAxes,
        # )


class HDF5_DAQ(DAQ):
    """An image taken by one of the FACET-II camera and saved as part of an HDF5 dataset."""

    def _load_image(self):
        """Load an image from a given data set.

        Returns
        -------
        image : obj
            Pillow image object from the hdf5.
        """
        self.path = os.path.join(self.dataset.datasetPath, "images", self.camera)
        name = os.path.join(self.path, self.filename)
        common_index = self.dataset.images[self.camera]["common_index"][self.ind]
        python_common_index = common_index - 1
        pid = self.dataset.images[self.camera]["pid"][python_common_index]
        f = h5py.File(name, "r")
        pids = f["entry/instrument/NDAttributes/NDArrayUniqueId"]
        ind = np.where(pids == pid)[0][0]
        data = f["entry/data/data"][ind]
        image = Image.fromarray(data)
        data = np.array(image, dtype="float")
        image = self._orientImage(data)
        return image
