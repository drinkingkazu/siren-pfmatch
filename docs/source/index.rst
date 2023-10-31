.. pfmatch documentation master file, created by
   sphinx-quickstart on Wed Oct 26 19:52:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pfmatch's documentation!
===========================================
`pfmatch` is implementation of [siren](https://github.com/CIDeR-ML/siren-lartpc) for applications in Liquid Argon Time Projection Chamber (LArTPC) experiments.  In particular:

* Toy simulation of a particle trajectory with PMT response
* "Flash matching" data reconstruction
* Optimization of siren using rael data (calibration datasets)

For the installation and tutorial notebooks, please see the `software repository <https://github.com/CIDeR-ML/siren-pfmatch>`_.


Getting started
---------------

You can find a quick guide to get started below.

Install ``pfmatch``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/cider-ml/siren-pfmatch
   cd siren-pfmatch
   pip install . --user


You can install to your system path by omitting ``--user`` flag. 
If you used ``--user`` flag to install in your personal space, assuming ``$HOME``, you may have to export ``$PATH`` environment variable to find executables.

.. code-block:: bash
   
   export PATH=$HOME/.local/bin:$PATH


.


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :glob:

   pfmatch <pfmatch>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`