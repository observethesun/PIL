import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import resize

from scripts.helper import open_img_as_array, plot_3d_tensor
from scripts.config import FITS_SHAPE

class ImageData:
    """
    Class for image data processing
    
    Parameters
    ----------
    path_or_img : str or np.array
        Path to image or image itself in np.array format with shape (width, height)
    data_mode : str
        Type of data to process. Possible modes:
        'img' - image itself in np.array format
        'path' - path to image
        'fits' - path to fits file
    radius : float
        Radius of cylinder to project image on
    reduce_factor : int
        Reduce factor for image. It needs to reduce image size plotting 3d data if it is too large

    Attributes
    ----------
    img_array : np.array
        Image in np.array format
    width : int
        Width of image
    height : int
        Height of image
    total_pixel : int
        Total number of pixels in image
    reduce_factor : int
        Reduce factor for image
    radius : float
        Radius of cylinder to project image on
    mask : np.array
        Array of indices of pixels without filaments
    bound_mask : np.array
        Array of indices of pixels with filaments
    bound_length : int
        Number of pixels with filaments
    data_2d : torch.Tensor
        2d representation of image in coordinates (x, y)
    data_3d : torch.Tensor
        3d representation of image in coordinates (x, y, z)
    """
    def __init__(self, path_or_img, data_mode, radius=1, reduce_factor=1):
        modes = {
            'img': lambda img: img,
            'path': lambda path: open_img_as_array(path), 
            'fits': lambda path: self._from_fits(path)
        }
        self.target_img = None
        if data_mode not in modes:
            raise ValueError(f'Data mode must be in {[x for x in modes.keys()]}')
        self.img_array = modes[data_mode](path_or_img)
        self.width, self.height = self.img_array.shape
        self.total_pixel = self.width * self.height
        self.reduce_factor = reduce_factor
        self.radius = radius
        mask = self.img_array.reshape(-1, 1) == self.img_array.max()
        self.mask = np.where(mask)[0]
        self.bound_mask = np.where(~mask)[0]
        self.bound_length = len(self.bound_mask)
        self.data_2d, self.data_3d = self.make_data()

    def _from_fits(self, path):
        with fits.open(path) as hdul:
            img = hdul[0].data[10:-10, 10:-10]
        filament = np.array(img != 9)
        sign = np.array(img == 7)
        filament, sign = map(lambda x: resize(x.astype(float), FITS_SHAPE, anti_aliasing=True), [filament, sign])
        filament = (filament - filament.min()) / (filament.max() - filament.min())
        sign = (sign - sign.min()) / (sign.max() - sign.min()) * 2 - 1
        self.target_img = sign
        return filament > 0.99

    def make_data(self):
        """
        Makes 2d and 3d coordinates of image
        """
        gap_size = 50 # gap between left and right parts of the image on cylinder
        a = torch.linspace(-0.5, 0.5, self.width)
        b = torch.linspace(-1, 1, self.height)
        aa, bb = torch.meshgrid(a, b, indexing='ij')
        data_2d = torch.stack([aa, bb], dim=2).view(-1, 2)
        roots = 2 * torch.pi * torch.arange(self.height + gap_size) / (self.height + gap_size)
        zs = torch.linspace(1, -1, self.width)
        z, y = torch.stack(torch.meshgrid(zs, roots[:-gap_size], indexing='ij'), dim=2).view(-1, 2).T
        z = z * torch.pi * self.width / self.height
        x = torch.cos(y)
        y = torch.sin(y)
        return data_2d, self.radius * torch.stack((x, y, z), dim=1)

    def get_info(self):
        print(f'width: {self.width}\n'
            f'height: {self.height}\n'
            f'total_pixel: {self.total_pixel}\n'
            f'bound length: {self.bound_length}\n'
            f'percent of bound pixels: {100 * self.bound_length / self.total_pixel:.1f}%')

    def show_input_img(self, figsize=(10, 5), cmap='PuOr'):
        plt.figure(figsize=figsize)
        plt.title(f'Input data visualization')
        plt.imshow(self.img_array, cmap=cmap, vmin=-1, vmax=1)

    def show_target_img(self, figsize=(10, 5), cmap='PuOr'):
        if self.target_img is None:
            raise ValueError('Target image is not defined')
        plt.figure(figsize=figsize)
        plt.title(f'Target data visualization')
        plt.imshow(self.target_img, cmap=cmap, vmin=-1, vmax=1)

    def show_3d(self, marker_size=2, colorscale='oxy'):
        plot_3d_tensor(tensor=self.data_3d,
                       color=self.img_array.flatten(),
                       colorscale=colorscale,
                       marker_size=marker_size)