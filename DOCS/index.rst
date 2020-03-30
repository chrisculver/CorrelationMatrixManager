..  CatCutifier documentation master file, created by
   sphinx-quickstart on Wed Apr 24 15:19:01 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

This code is written as an add-on to the gwu-qcd library.  It takes as input information
about a lattice and a list of operators.  It then manages the computation of the correlation
matrix, exiting if wick contractions or numerical evaluation of diagrams are missing.


..  toctree::
    :caption: User Guide
    :maxdepth: 2

    USER_GUIDE/getting_started.rst
    USER_GUIDE/input.rst

..  toctree::
    :caption: Physics
    :maxdepth: 2

    PHYSICS/intro.rst
    PHYSICS/operators.rst

..  toctree::
    :caption: API
    :maxdepth:  2

    API/manager.rst

:ref:`genindex`
