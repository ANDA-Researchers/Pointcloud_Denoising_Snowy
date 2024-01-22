from setuptools import setup


setup(
    name="pc_denoising",
    version="0.0.1",
    packages=["pc_denoising"],
    install_requires=[
        "ipykernel==6.29.0",
        "numpy==1.24.4",
        "matplotlib==3.7.2",
        "opencv-python==4.9.0.80",
        "mayavi==4.8.1",
    ],
)
