from setuptools import setup


setup(
    name="pc_denoising",
    version="0.0.1",
    packages=["pc_denoising"],
    install_requires=[
        "ipykernel==6.29.0",
        "ipywidgets==8.1.1",
        "jsonargparse[signatures]>=4.27.2",
        "numpy==1.24.4",
        "matplotlib==3.7.2",
        "opencv-python==4.9.0.80",
        "tqdm==4.66.1",
    ],
)
