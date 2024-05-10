# Pytorch image descent
A library to test optimizers by visualizing how they descend on a your images. You can draw your own custom loss landscape and see what different optimizers do, or you can use any function. The use is mostly to see if your momentum works as intended, but maybe there are other uses. Example:
```py
from image_descent import ImageDescent

# you can put path to an image or a numpy array / torch tensor.
# It will be converted into black-and-white channel-last.
image = r"surfaces/spiral.jpg"

# coords are the initial coordinate to start optimization from
# either pixel location (int), or relative coordinates in (-1,-1) to (1,1) range (float).
descent = ImageDescent(image, coords=(915, 500))

optimizer = torch.optim.Adam(descent.parameters(), lr=0.05)
for i in range(2000):
    optimizer.zero_grad()
    descent.step() # sets the .grad attribute
    optimizer.step()
descent.plot_path()
```
![image](https://github.com/stunlocked1/image-descent/assets/76593873/18dea516-3208-4966-9dcd-9282d2a9fc5d)

We can make it more stochastic - for example adding noise and randomly shifting the loss landscape before each step:
```py
import random

def add_noise(img:torch.Tensor):
    noise = torch.randn(img.shape)
    return (img + noise*0.003).clamp(0, 1)

def random_shift(img:torch.Tensor):
    shiftx = int(random.triangular(0, 50, 0))
    shifty = int(random.triangular(0, 50, 0))
    return img[shiftx:, shifty:]

# img_step accepts a functions or sequence of functions, that will be applied to the loss landscape image before each step.
descent = ImageDescent(image, initial_coords, img_step=[random_shift, add_noise])
optimizer = torch.optim.AdamW(descent.parameters(), 3e-2)

for i in range(1000):
    optimizer.zero_grad()
    loss = descent.step()
    optimizer.step()
    if i% 100 == 0: print(i, loss, end='        \r')

descent.plot_path()
```
![image](https://github.com/stunlocked1/image-descent/assets/76593873/d7525dfe-aefc-4bf2-b5c1-379611831214)

## Descending functions
You can optimize normal functions as well, for example:
```py
from image_descent import FunctionDescent2D
descent = FunctionDescent2D(lambda x,y: x**2 + y**2, coords = (-1, -0.9))
optimizer = torch.optim.Adam(descent.parameters(), lr=0.05)
for i in range(100):
    optimizer.zero_grad()
    loss = descent()
    loss.backward()
    optimizer.step()
descent.plot_path()
```
![image](https://github.com/stunlocked1/image-descent/assets/76593873/8c5cceae-f2fb-4274-8afd-909c5dd02baf)

## How it works
X and Y coordinates are the parameters that the optimizers try to optimize to find the lowest (darkest) spot on the image. Loss is given by `image[current_coordinates]`.

The gradients are calculated using finite differences, like this: `gradx, grady = (image[1:] - image[:-1], image[:,1:] - image[:,:-1]`. 

Then `x_gradient[current_coordinates]` is the X-coordinate gradient for the current point. 

However since coordinates are not discrete, [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) is used to interpolate them. Thats why coordinates start at (-1,-1) and end at (1,1).

## More images
I've used the same
```
descent = ImageDescent(image, initial_coords, img_step=[random_shift, add_noise])
```
with a bunch of optimizers, tuning their hyperparameters.

opt = torch.optim.SGD(descent.parameters(), 5e-2):

![image](https://github.com/stunlocked1/image-descent/assets/76593873/8e77e0b6-2bab-414e-8f66-9644dcb29b22)

opt = torch.optim.SGD(descent.parameters(), 3e-3, momentum=0.99)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/2f7a30d3-0790-4073-b526-e35f7dd54145)

opt = torch.optim.SGD(descent.parameters(), 3e-3, momentum=0.99, nesterov=True)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/e2fd59aa-3b47-4666-ba58-7be899a79ced)

opt = Lion(descent.parameters(), 3e-2) (https://github.com/lucidrains/lion-pytorch)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/463a18bb-c43c-4737-a07d-f75f9ef11ed1)

opt = Fromage(descent.parameters(), 2e-1) (https://github.com/jxbz/fromage)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/54cbfaa3-f293-49d5-af18-84a161dfedaa)

opt = SophiaG(descent.parameters(), 1e-2) (https://github.com/Liuhong99/Sophia)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/5a1d8734-a847-4ad5-bca8-5b7b7fda4c9c)

opt = RandomDeltaSearch(descent.parameters(), 4e-2) (second order random search)

![image](https://github.com/stunlocked1/image-descent/assets/76593873/27251178-5392-45b4-b88a-95002d70df04)


## Installation
```py
pip install git+https://github.com/stunlocked1/image-descent
```
