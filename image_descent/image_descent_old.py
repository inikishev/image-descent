import numpy as np
import torch

from .imagetools import imread, ensure_channel_last, to_black_and_white, normalize_to_range, get_interpolated_value, Compose

class ImageDescent(torch.nn.Module):
    def __init__(
        self,
        image: np.ndarray,
        init=torch.rand,
        interp_order=1,
        coord_scale=100,
        loader=Compose(ensure_channel_last, to_black_and_white, normalize_to_range),
        dtype=torch.float64,
    ):
        super().__init__()

        self.image:np.ndarray = loader(imread(image)) # type:ignore
        self.interp_order = interp_order
        self.coord_scale = coord_scale
        self.dtype = dtype
        
        # values outside of the image are set to this (would be nice to set them so that they increase)
        self.img_max = 10
        self.gradient_max = 10

        # gradients are evaluated using np.gradient, which returns a tuple for each dimension,
        # e.g. for a 2D black and white image we get x-axis gradient and y-axis gradient, both same shape as image
        self.gradient = np.gradient(image, edge_order=1)

        # set initial coordinates, init can be callable or a predefined coord
        if callable(init): self.coords = torch.nn.Parameter(init(image.ndim))
        else:
            if isinstance(init[0], int): init = [i/s for i, s in zip(init, image.shape)]
            if isinstance(init, torch.Tensor): self.coords = torch.nn.Parameter(init.to(self.dtype))
            else: self.coords = torch.nn.Parameter(torch.from_numpy(np.asanyarray(init)).to(self.dtype))

        # history of coords
        self.coord_history = []
        self.loss_history = []

    @torch.no_grad()
    def step(self):
        coords_numpy = self.coords.detach().cpu().numpy().copy()
        # save current coords to the history
        self.coord_history.append(coords_numpy)

        # we just get the gradient for each axis at current coordinates, and since coords will be floats, the values will be interpolated
        grad = []
        for grad_dim in self.gradient: grad.append(get_interpolated_value(grad_dim, coords_numpy, order=self.interp_order, img_max=self.gradient_max))

        # then we set that gradient into the grad attribute that all optimizers use
        self.coords.grad = torch.tensor(grad, dtype=self.dtype) * self.coord_scale # type:ignore

        # return loss, which is value of the image at current coords
        loss = get_interpolated_value(self.image, self.coords.detach().cpu().numpy(), order=self.interp_order, img_max = self.img_max) # pylint:disable=E1102
        self.loss_history.append(loss)
        return loss

    # so that its a proper nn.Module
    def forward(self): return self.step()

    def copy(self):
        d = ImageDescent(self.image)
        d.image = self.image.copy()
        d.interp_order = self.interp_order
        d.gradient = self.gradient.copy()
        d.coords = torch.nn.Parameter(self.coords.clone())
        d.coord_history = self.coord_history.copy()
        d.loss_history = self.loss_history.copy()
        return d