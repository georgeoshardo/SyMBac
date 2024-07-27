from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name = 'SyMBac',
    version = '0.4.7',
    description = 'A package for generating synthetic images of bactera in phase contrast or fluorescence. Used for creating training data for machine learning segmentation and tracking algorithms.',
    url = 'https://github.com/georgeoshardo/SyMBac',
    author = 'Georgeos Hardo',
    author_email = 'gh464@cam.ac.uk',
    license = 'GPL-2.0',
    packages = ['SyMBac', 'SyMBac.external', 'SyMBac.sample_images', 'SyMBac.external.DeLTA', 'SyMBac.deep_learning'],
    package_data = {'': ['sample_images/*.tiff']},
    include_package_data = True,
    long_description = (this_directory / "README.md").read_text(), 
    long_description_content_type = 'text/markdown',
    install_requires = [ #  conda install -c conda-forge libstdcxx-ng sudo apt install --reinstall libgl1-mesa-dri    
        'tifffile',
        'scikit-image',
        'matplotlib',
        "tqdm",
        "pandas",
        "natsort",
        "ipython",
        "ipywidgets",
        "joblib",
        "scipy",
        "napari[all]",
        "pymunk",
        "pyglet",
        "matplotlib-scalebar",
        "psfmodels",
        "sphinx_automodapi",
        "sphinx-autobuild",
        "nbsphinx",
        "sphinx_copybutton",
        "sphinx_rtd_theme",
        "pandoc",
        "zarr"
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
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering'
    ],
)
