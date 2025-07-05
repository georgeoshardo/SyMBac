SyMBac Documentation
====================



SyMBac (Synthetic Micrographs of Bacteria) is a tool to generate synthetic phase contrast or fluorescence images of bacteria. Currently the tool only supports bacteria growing in the mother machine and on agar pads. 

Coming soon:
 - Agar pad simulations (working, code being added)
 - Command line interface (currently testing)

Read the preprint_, SyMBac: Synthetic Micrographs for Accurate Segmentation of Bacterial Cells using Deep Neural Networks, Georgeos Hardo, Maximillian Noka, Somenath Bakshi

.. note::

   This project is under active development. Please report issues on GitHub, or if you want specific usage help, please email_ me.


Contents
--------


.. toctree::
   :caption: Introduction
   :maxdepth: 3

   intro
   installation
   faqs

.. toctree:: 
   :caption: Guide: Mother Machine
   :maxdepth: 3

   Mother machine simulations <examples/simple_mother_machine.ipynb>

.. toctree:: 
   :caption: Guide: Agar Pad
   :maxdepth: 3

   Agar pad simulations <examples/agar_pad.ipynb>

.. toctree::
   :caption: Examples
   :maxdepth: 1

   Training Omnipose <examples/omnipose_training_data_generator.ipynb>
   Segmenting with Omnipose <examples/seg_with_omnipose.ipynb>

.. toctree::
   :caption: Classes reference
   :maxdepth: 3

   symbac_cell
   symbac_PSF
   symbac_renderer
   symbac_simulation

.. toctree:: 
   :caption: API reference
   :maxdepth: 3

   symbac_cell_geometry
   symbac_cell_simulation
   symbac_drawing
   symbac_misc
   symbac_pySHINE




.. image:: https://readthedocs.org/projects/symbac/badge/?version=latest
   :target: https://symbac.readthedocs.io/en/latest/?badge=latest

.. image:: https://pepy.tech/badge/symbac
   :target: https://pepy.tech/project/symbac

.. image:: https://badge.fury.io/py/symbac.svg
   :target: https://badge.fury.io/py/symbac

.. _preprint: https://www.biorxiv.org/content/10.1101/2021.07.21.453284v4

.. _email: gh464@cam.ac.uk

.. _DeLTA: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007673

.. _Omnipose: https://www.biorxiv.org/content/10.1101/2021.11.03.467199v4
