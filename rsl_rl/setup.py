from setuptools import setup, find_packages

setup(name='rsl_rl',
      version='1.0.2',
      author='The CMoE Authors (Fudan University)',
      author_email='',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='RL algorithms for the CMoE humanoid locomotion framework (based on rsl_rl)',
      python_requires='>=3.7.16',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
