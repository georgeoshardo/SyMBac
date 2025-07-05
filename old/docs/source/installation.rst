Installation
====================

Prerequisites
-------------

Please make sure you have an NVIDIA GPU and a working installation of CUDA and cudNN. If you don't have an NVIDIA GPU then the convolution will default to the CPU, and be very slow.

SyMBac is meant to be run interactively (in a notebook + with a small Qt/GTK interface), so make sure that you are running this on a local machine (you should have access to the machine's display).

If you are running SyMBac on a remote machine, say through an SSH tunnel, you can still use it, but you will need to ensure you have an active VNC screen available, as SyMBac needs access to a screen to render the live simulation. You do not need to be actively accessing the VNC session, it just needs to be running.

Installation
------------

.. code-block:: bash

    pip install SyMBac

Or to install the development version (recommended for now), run:

.. code-block:: bash

    pip install git+https://github.com/georgeoshardo/SyMBac

Activate the Jupyter widgets extension. This is needed to interact with slides in the notebooks to optimise images.

.. code-block:: bash
    
    jupyter nbextension enable --py widgetsnbextension

If you're using a GPU
^^^^^^^^^^^^^^^^^^^^^

Check the version of CUDA you have installed using nvcc --version and install the appropriate version of cupy. For example, if you have CUDA 11.4 you would install as follows:

.. code-block:: bash

    pip install cupy-cuda114

If you installed CUDA on Ubuntu 18.04+ using the new Nvidia supplied repositories, it is a real possibility that nvcc won't work. Instead check your CUDA version using nvidia-smi.

If you aren't using a GPU
^^^^^^^^^^^^^^^^^^^^^^^^^

See FAQs "Do I need to have a GPU?"