from setuptools import setup

setup(
   name='image_descent',
   version='0.1',
   description='A library to test optimizers by visualizing how they descend on a your images. You can draw your own custom loss landscape and see what different optimizers do. The use is mostly to see if your momentum works as intended, but maybe there are other uses.',
   author='Big Chungus',
   author_email='nkshv2@gmail.com',
   packages=['image_descent'],  #same as name
   install_requires=['torch', 'matplotlib', 'numpy', 'scipy'], #external packages as dependencies
)
