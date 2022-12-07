from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name = 'SyMBac',
    version = '0.3.1',
    description = 'A package for generating synthetic images of bactera in phase contrast or fluorescence. Used for creating training data for machine learning segmentation and tracking algorithms.',
    url = 'https://github.com/georgeoshardo/SyMBac',
    author = 'Georgeos Hardo',
    author_email = 'gh464@cam.ac.uk',
    license = 'GPL-2.0',
    packages = ['SyMBac', 'SyMBac.external', 'SyMBac.sample_images', 'SyMBac.external.DeLTA'],
    package_data = {'': ['sample_images/*.tiff']},
    include_package_data = True,
    long_description = (this_directory / "README.md").read_text(), 
    long_description_content_type = 'text/markdown',
    install_requires = [
        'tifffile',
        'scikit-image',
        'matplotlib==3.4.3',
        "tqdm",
        "pandas==1.3.4",
        "natsort",
        "jupyterlab",
        "ipywidgets",
        "joblib",
        "napari[all]",
        "pymunk==6.2.0",
        "pyglet==1.5.21",
        "raster-geometry==0.1.4.1",
        "matplotlib-scalebar==0.7.2",
        "psfmodels",
        "numba==0.56.4"
        ],

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',  
        'Operating System :: POSIX :: Linux', 
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering'
    ],
)
