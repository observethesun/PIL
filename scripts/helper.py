import imageio
import glob
import numpy as np
import torch
from PIL import Image, ImageOps
import plotly.graph_objects as go
from typing import Union, Optional


def open_img_as_array(path_to_img: str) -> np.ndarray:
    """
    Opens image as a numpy array of shape (width, height) in grayscale (int values from 0 to 255 inclusive)

    """
    return np.array(ImageOps.grayscale(Image.open(path_to_img))).astype(int)

def plot_3d_tensor(
        tensor: Union[torch.Tensor, np.ndarray],
        color: Optional[Union[torch.Tensor, np.ndarray]] = None,
        colorscale: str = 'PuOr',
        marker_size: float = 2) -> None:
    """
    Plots 3d tensor with color values

    Parameters
    ----------
    tensor : torch.Tensor | np.ndarray
        Tensor of shape (n, 3)
    color : torch.Tensor | np.ndarray | None
        Tensor of shape (3n,) with color values
    colorscale : str
        Colorscale for plotly
    marker_size : float
        Size of markers
    """
    x, y, z = tensor.T
    fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=color,
                colorscale=colorscale))])
    fig.show()

def make_gif(path_to_imgs: str,
            path_to_save: str,
            gifname: str,
            fps: int = 20):
    """
    Makes gif from images in path_to_imgs

    Parameters
    ----------
    path_to_imgs : str
        Path to images
    path_to_save : str
        Path to save gif
    gifname : str
        Name of gif
    fps : int
        Frames per second
    """
    filenames = [name for name in glob.glob(f'{path_to_imgs}/*.png')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{path_to_save}/{gifname}.gif', images, format='gif', fps=fps)