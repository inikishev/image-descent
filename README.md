# Pytorch image descent
A library to test optimizers by visualizing how they descend on a your images. You can draw your own custom loss landscape and see what different optimizers do. Example:
```py
from image_descent import ImageDescent

# you can put path to an image or a numpy array / torch tensor.
# It will be converted into black-and-white channel-last.
image = r"surfaces/spiral.jpg"

# init is the initial coordinate.
# The coordinates always start at (-1,-1) - top left corner, and (1,1) is bottom right corner.
descent = ImageDescent(image, init=(0.785,0))

optimizer = torch.optim.Adam(descent.parameters(), lr=0.05)
for i in range(2000):
    optimizer.zero_grad()
    descent.step() # sets the .grad attribute
    optimizer.step()
descent.plot_path()
```
Adam:

![image](https://github.com/stunlocked1/image-descent/assets/76593873/4e07bfaf-a275-4e2f-ae9d-cff6fd2449b2)

SGD:

![image](https://github.com/stunlocked1/image-descent/assets/76593873/ef8962ee-ee28-428d-a133-757cc6836f3a)

Lion (https://github.com/lucidrains/lion-pytorch):

![image](https://github.com/stunlocked1/image-descent/assets/76593873/cedc7001-5969-44bd-8ae4-9d06c42fef07)

Fromage (https://github.com/jxbz/fromage):

![image](https://github.com/stunlocked1/image-descent/assets/76593873/080a25ae-c9ff-4791-a606-343ad8ae6463)

Second order random search:

![image](https://github.com/stunlocked1/image-descent/assets/76593873/00d1601a-3ace-4c16-bb43-f7ce5f0a20be)

I doubt that a spiral tells you much about how well an optimizer works. Maybe there are some good images to test them on, but also I need to add some randomness to each step to make it more realistic. At least it shows if your momentum works okay.

## How it works
X and Y coordinates are the parameters that the optimizers try to optimize to find the lowest (darkest) spot on the image. Loss is given by `image[current_coordinates]`.

The gradients are pre-calculated as `numpy.gradient(image)`. This calculates differences between every adjascent pixel along all axes, which is equivalent to calculating mean difference between the original image and image shifted by one pixel to the right to get X-axis gradient, and one pixel to the bottom to get Y-axis gradient. Then `x_gradient[current_coordinates]` is the X-coordinate gradient for the current point. However since coordinates are not discrete, [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) is used to interpolate them. Thats why coordinates start at (-1,-1) and end at (1,1).
