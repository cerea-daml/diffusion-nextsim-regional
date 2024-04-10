from setuptools import find_packages, setup

setup(
    name='diffusion_nextsim',
    packages=find_packages(
        include=["diffusion_nextsim"]
    ),
    version='0.1.0',
    description='Training of diffusion models for regional sea-ice modelling',
    author='Tobias Finn',
    license='MIT',
)
