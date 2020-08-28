..  CatCutifier documentation master file, created by
   sphinx-quickstart on Wed Apr 24 15:19:01 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

CorrelationMatrixManager manages the correlation matrix for hadron spectroscopy calculations.

Current functionality includes

- Read in a list of operators constructed out of mesons.
- Compute the wick contractions for all correlation matrix elements.
- Check on disk for numerical values of diagrams.

  - If none exist, output C++ code to compute diagrams using GWU-QCD library.

- Return the correlation functions averaged over all times.


Follow the read me on `github <https://github.com/chrisculver/CorrelationMatrixManager>`_ for
installation instructions.  Once the program is successfully installed head to :ref:`getting_started`
for a quick-start guide to the code.  To see how the input is formatted head to :ref:`input`.

..  toctree::
    :caption: User Guide
    :maxdepth: 2

    USER_GUIDE/getting_started.rst
    USER_GUIDE/input.rst

..  toctree::
    :caption: API
    :maxdepth:  2

    API/manager.rst

:ref:`genindex`
