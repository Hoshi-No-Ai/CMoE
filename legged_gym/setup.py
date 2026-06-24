from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    author='The CMoE Authors (Fudan University)',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Isaac Gym environments for the CMoE humanoid locomotion framework (based on legged_gym)',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'numpy',
                      'tensorboard',
                      'tqdm',
                      'trimesh']
)