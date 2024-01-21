from setuptools import setup


setup(
    name="pc_denoising",
    version="0.0.1",
    packages=["pc_denoising"],
    install_requires=[
        "ipykernel==6.29.0",
        "matplotlib==3.7.4",
        "mayavi==4.8.1",
        "numpy==1.24.4",
        "opencv-python==4.9.0.80",
        "torch==2.1.2",
        "torchvision==0.16.2",
        "black==23.12.1",
    ],
)
