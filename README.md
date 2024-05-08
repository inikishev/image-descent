# Pytorch image descent
A library to test optimizers by visualizing how they descend on a your images. You can draw your own custom loss landscape and see what different optimizers do. Example:
```py
image = r"surfaces/spiral.jpg"
from image_descent import ImageDescent
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
