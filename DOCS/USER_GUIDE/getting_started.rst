.. _getting_started:

Getting Started
===============

As an example of how the code runs, we will compute the pion correlation
function on the 4x4 test lattice, using a saved value for the diagram.

To accomplish this we need to take two input files out of EXAMPLES,
`pion.in` and `pion.op`.  The second file can specify as many pion operators
as desired.

The file `pion.in` contains

.. literalinclude:: pion.in
    :linenos:
    :language: none

The file `pion.ops` contains a \bar{q}q operator for the \pi^0.

.. literalinclude:: pion.ops
    :linenos:
    :language: none

We also need the diagram file, `diags_4444_100.dat`.  Now by running

`./compute_correlation_matrix pion.in

The code will read in the operators, compute the relevant wick contractions,
and output a file containing the correlation functions from dt=0 to dt=NT-1.
