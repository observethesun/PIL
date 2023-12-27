import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import clear_output
from typing import List, Tuple, Optional

from scripts.config import LOSSES
from scripts.helper import plot_3d_tensor
from scripts.imagedata import ImageData

class NeutralLiner(nn.Module):
    """
    A neural network class that extends PyTorch's nn.Module. 

    This class is designed to process a list of image data (ImageData objects),
    applying a customizable fully connected neural network architecture. It supports
    different modes for processing images ('2d' or '3d'), allows for adjustable learning
    rate and weight decay parameters, and includes methods for training, testing,
    saving state, and visualizing results.

    Parameters
    ----------
    image_list : List[ImageData]
        List of ImageData objects including the information about the filaments.
    lr : float
        Learning rate.
    help_step_size : int
        Step size for the grid-based reference points for the help loss.
    mode : str
        Dimension of the space for the input image ('2d' or '3d').
    arch : List[int]
        Architecture of the network (number of neurons in each layer respectively).
    weight_decay : float
        Regularization parameter for the optimizer.
    device : str
        Device to use for computations ('cpu' or 'cuda').

    Attributes
    ----------
    net : nn.Sequential
        Neural network architecture.
    arch : Tuple[int]
        Architecture of the network (number of neurons in each layer).
    image_list : List[ImageData]
        List of ImageData objects.
    data_list : List[torch.Tensor]
        List of image tensors.
    lr : float
        Learning rate.
    weight_decay : float
        Regularization parameter for the optimizer.
    optimizer : torch.optim
        Optimizer for the neural network.
    help_step_size : int
        Step size for the help loss.
    loss_dict : Dict[str, List[float]]
        Dictionary of loss values.
    """
    __slots__ = 'image_list'
    def __init__(
            self,
            image_list: List[ImageData],
            lr: float,
            help_step_size: Optional[int] = None,
            mode: str = '3d',
            arch: Tuple[int] = (3, 6, 12, 24, 12, 6, 3, 1),
            weight_decay: float = 1e-3,
            device: str = 'cuda:0'
        ):
        super(NeutralLiner, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(arch)-1):
            self.net.add_module(f'linear_{i}', nn.Linear(arch[i], arch[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.arch = arch
        self.image_list = image_list
        self.data_list = [img.data_3d if mode == '3d' else img.data_2d for img in image_list]
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.help_step_size = help_step_size
        self.loss_dict = {key: [] for key in LOSSES}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of coordinates.
        """
        output = self.net(x)
        return output

    def restart_model(self, lr: float, weight_decay: float = 1e-3) -> 'NeutralLiner':
        """
        Restarts and returns a new model with updated parameters.

        Parameters
        ----------
        lr : float
            Learning rate.
        weight_decay : float
            Regularization parameter for the optimizer.
        """
        model = NeutralLiner(image_list=self.image_list, lr=lr, weight_decay=weight_decay)
        model.to(self.device)
        return model

    def change_lr(self, lr: float) -> None:
        """
        Changes the learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            New learning rate.
        """
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def save_state_dict(self, path: str) -> None:
        """
        Saves the state dictionary of the model.

        :param path: Path to the file.
        """
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        torch.save(self.state_dict(), path)

    def show_3d(self,
                prediction_list: List[torch.Tensor],
                map_number: int = 0,
                marker_size: float = 2,
                colorscale: str = 'PuOr'
                ) -> None:
        """
        Plots the 3d map of the prediction in 3d.

        Parameters
        ----------
        prediction_list : List[torch.Tensor]
            List of prediction tensors.
        map_number : int
            Number of the map to plot.
        marker_size : float
            Size of the markers.
        colorscale : str
            Color scale for the plot.
        """
        plot_3d_tensor(tensor=self.data_list[map_number].cpu().detach(),
                       color=prediction_list[map_number].flatten(),
                       colorscale=colorscale,
                       marker_size=marker_size)

    def start_training(
            self,
            num_epochs: int,
            need_plot: bool,
            clear_loss: bool = True,
            path_to_save: Optional[str] = None,
            f_integral_weight: float = 1e-1,
            show_frequency: int = 100
            ) -> None:
        """
        Trains the model for a specified number of epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model for.
        need_plot : bool
            Indicates whether to plot the results after each show_frequency epochs.
        clear_loss : bool
            Indicates whether to clear the loss values before training.
        path_to_save : str | None
            Path to the directory to save the plots on train to.
        f_integral_weight : float
            Weight of the term with the integral of the function f over cylinder.
        show_frequency : int
            Number of epochs after which to plot the results.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)
        if clear_loss:
            for value in self.loss_dict.values():
                value.clear()
        data_list = [data.to(self.device) for data in self.data_list]
        for epoch in range(1, num_epochs + 1):
            output_list = [self(data) for data in data_list]
            self.optimizer.zero_grad()
            loss = self.compute_loss(output_list=output_list, f_integral_weight=f_integral_weight)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if need_plot and (epoch % show_frequency == 0 or epoch == num_epochs):
                self._plot_on_train(output_list, epoch, path_to_save)
                self.show_loss_items()

    def _plot_on_train(self, output_list, epoch, path_to_save=None):
        output_list = [output.cpu().detach().view(img.img_array.shape)
                       for output, img in zip(output_list, self.image_list)]
        clear_output(wait=True)
        gs = gridspec.GridSpec(len(self.image_list) + 1, 3 if self.image_list[0].target_img is not None else 2)
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(self.loss_dict['loss'])
        for (i, img), data in zip(enumerate(output_list), self.image_list):
            ax2 = fig.add_subplot(gs[i + 1, 0])
            ax2.title.set_text(f'Prediction №{i + 1}')
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])
            ax2.imshow(img, cmap='PuOr', vmin=-1, vmax=1)
            x, y = np.where(data.img_array < data.img_array.max())
            ax2.scatter(y, x, s=0.2, c='green', alpha=0.5)
            ax3 = fig.add_subplot(gs[i + 1, 1])
            ax3.get_xaxis().set_ticks([])
            ax3.get_yaxis().set_ticks([])
            ax3.title.set_text(f'Input image №{i + 1}')
            ax3.imshow(data.img_array, cmap='PuOr', vmin=-1, vmax=1)
            if data.target_img is not None:
                ax4 = fig.add_subplot(gs[i + 1, 2])
                ax4.get_xaxis().set_ticks([])
                ax4.get_yaxis().set_ticks([])
                ax4.title.set_text(f'Target image №{i + 1}')
                ax4.imshow(data.target_img, cmap='PuOr', vmin=-1, vmax=1)
        if path_to_save:
            plt.savefig(path_to_save + '/epoch%06d.png' % epoch)
        plt.show()

    def test_model(self,
                   input_list: Optional[List[torch.Tensor]] = None,
                   need_plot: bool = True,
                   full_path_to_save: Optional[str] = None
                   ) -> List[torch.Tensor]:
        """
        Tests the model on a specified list of inputs. If no list is provided, the model is tested on the data it was trained on.
        Otherwise, the model is tested on the provided list of input tensors of coordinates.

        Parameters
        ----------
        input_list : List[torch.Tensor] | None
            List of input tensors of coordinates.
        need_plot : bool
            Indicates whether to plot the results.
        full_path_to_save : str | None
            Full path to the directory to save the plots on test to.
        """
        input_list = self.data_list if input_list is None else input_list
        with torch.no_grad():
            output_list = [self(inpt.to(self.device)).cpu().detach() for inpt in input_list]
        if need_plot or full_path_to_save:
            plt.figure(figsize=(10, 18))
            for (i, output), img in zip(enumerate(output_list), self.image_list):
                prediction = output.view(img.img_array.shape).clone()
                plt.subplot(2 if img.target_img is None else 3, len(output_list), i + 1)
                plt.title('Input image')
                plt.imshow(img.img_array, cmap='PuOr', vmin=-1, vmax=1)
                plt.subplot(2 if img.target_img is None else 3, len(output_list), i + 2)
                plt.title('Prediction')
                plt.imshow(prediction, cmap='PuOr')
                x, y = np.where(img.img_array < img.img_array.max())
                plt.scatter(y, x, s=0.2, c='green', alpha=0.5)
                if img.target_img is not None:
                    plt.subplot(3, len(output_list), i + 3)
                    plt.title('Target image')
                    plt.imshow(img.target_img, cmap='PuOr', vmin=-1, vmax=1)
                plt.tight_layout()
                if full_path_to_save is not None:
                    os.makedirs('/'.join(full_path_to_save.split('/')[:-1]), exist_ok=True)
                    plt.savefig(full_path_to_save, bbox_inches='tight', pad_inches=0, facecolor='white')
                if need_plot:
                    plt.show()
                else:
                    plt.close()
        return output_list

    def compute_and_plot_gradient(self, input_list: Optional[List[torch.Tensor]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes and plots the gradient map for the specified list of inputs.
        If no list is provided, the gradient map is computed for the data the model was trained on.
        Otherwise, the gradient map is computed for the provided list of input tensors of coordinates.

        Parameters
        ----------
        input_list : List[torch.Tensor] | None
            List of input tensors of coordinates.
        """
        batch_list = self.data_list if input_list is None else input_list
        for batch in batch_list:
            batch.requires_grad_()
        tmp_list = [self(batch.to(self.device)) for batch in batch_list]
        for tmp in tmp_list:
            tmp.sum().backward()
        grad_out_list = [batch.grad.cpu() for batch in batch_list]
        grad_map_list = [((grad_out.pow(2).sum(dim=1)).pow(0.5)).view(img.img_array.shape) for grad_out, img in zip(grad_out_list, self.image_list)]
        plt.figure(figsize=(12, 6))
        for i, grad_map in enumerate(grad_map_list):
            plt.subplot(1, len(grad_map_list), i + 1)
            plt.title(r'Gradient map for $||\nabla f(x,y,z)||_2$' + f'\non input №{i + 1}', fontsize=15)
            plt.imshow(grad_map, cmap='plasma')
            plt.colorbar(location='bottom')
        return grad_out_list, grad_map_list

    def show_loss_items(self) -> None:
        """
        Plots the loss values for each loss item defined in the config file.
        """
        plt.figure(figsize=(10, 20))
        losses = list(self.loss_dict.items())[1:]
        for i, (key, value) in enumerate(losses):
            plt.subplot(len(losses), 1, i + 1)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.plot(value)
        plt.show()

    def compute_loss(self, 
                     output_list: List[torch.Tensor],
                     f_integral_weight: float,
                     help_size: int = 20
                     ) -> torch.Tensor:
        """
        Computes the loss for the specified list of outputs.

        Parameters
        ----------
        output_list : List[torch.Tensor]
            List of output tensors of the model.
        f_integral_weight : float
            Weight of the term with the integral of the function f over cylinder.
        help_size : int
            Size of the margin in pixels for the help loss region.
        """
        loss, f_abs_integral, bound_integral, orientation_integral, f_integral, MSE_help = torch.zeros(len(self.loss_dict))
        loss_, f_abs_integral_, bound_integral_, orientation_integral_, f_integral_, MSE_help_ = torch.zeros(len(self.loss_dict))
        for (output, img) in zip(output_list, self.image_list):
            output = output.flatten()
            for key in self.loss_dict.keys():
                locals()[key + "_"] = locals()[key].clone()
            rows = int(img.width / 18) * img.height
            upper_bound = (output[:rows].mean() - 1).abs()
            lower_bound = (output[-rows:].mean() + 1).abs()
            orientation_integral = orientation_integral_ + 0.25 * (upper_bound + lower_bound)
            f_integral = f_integral_ + f_integral_weight * output.mean().abs()
            f_abs_integral = f_abs_integral_ + 1 - output[img.mask].abs().mean() # +-1 for region without filaments
            bound_integral = bound_integral_ + output[img.bound_mask].pow(2).mean() # 0 for region with filaments
            loss = loss_ + f_abs_integral + bound_integral + orientation_integral + f_integral
            if self.help_step_size is not None:
                MSE_help = MSE_help_ + nn.functional.mse_loss(
                    output.view(img.img_array.shape)[help_size:-help_size, :][::self.help_step_size, ::self.help_step_size],
                    torch.FloatTensor(img.target_img[help_size:-help_size, :]).to(self.device)[::self.help_step_size, ::self.help_step_size])
                loss += MSE_help
        for key in self.loss_dict.keys():
            self.loss_dict[key].append(locals()[key].item())
        return loss