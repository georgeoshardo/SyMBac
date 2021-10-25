from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='SyMBac',
    version='0.1.4',    
    description='A package for generating synthetic images of bactera in phase contrast or fluorescence. Used for creating training data for machine learning segmentation and tracking algorithms.',
    url='https://github.com/georgeoshardo/SyMBac',
    author='Georgeos Hardo',
    author_email='gh464@cam.ac.uk',
    license='GPL-2.0',
    packages=['SyMBac','SyMBac.external','SyMBac.sample_images','SyMBac.external.DeLTA'],
    long_description = (this_directory / "README.md").read_text(), #"SyMBac is a tool to generate synthetic phase contrast or fluorescence images of bacteria. Currently the tool only supports bacteria growing in the mother machine, however support for bacteria growing in monolayers (and maybe even biofilms!) is coming.",
    long_description_content_type='text/markdown',
    install_requires=['tensorflow==2.6.0',
                      'elasticdeform==0.4.9', 
                      'tifffile==2021.10.12',
                      'scikit-image==0.18.3' ,
                      'matplotlib==3.4.3',
                      "tqdm==4.62.3",
                      "pandas==1.3.4",
                      "natsort==7.1.1",
                      "jupyterlab==3.2.0",
                      "ipywidgets==7.6.5",
                      "joblib==1.1.0",
                      "napari==0.4.11",
                      "pymunk==6.2.0",
                      "pyglet==1.5.21",
                      "raster-geometry==0.1.4.1",
                      "matplotlib-scalebar==0.7.2"
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',  
        'Operating System :: POSIX :: Linux', 
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering'
    ],
)
