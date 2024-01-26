from setuptools import setup


setup(
    name="pc_denoising",
    packages=["pc_denoising"],
    entry_points={
        "console_scripts": [
            "unet = pc_denoising.main:cli_main",
        ],
    },
    install_requires=[
        "ipykernel==6.29.0",
        "ipywidgets==8.1.1",
        "jsonargparse[signatures]>=4.27.2",
        "matplotlib==3.7.2",
        "numpy==1.24.4",
        "opencv-python==4.9.0.80",
        "scikit-learn==1.10.1",
        "tqdm==4.66.1",
    ],
)
