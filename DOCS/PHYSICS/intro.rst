Physics
=======

The correlation matrix is defined as

..  math::
    C_{ij}(t) = \langle O_i(t) O_j(0)^{\dagger} \rangle

where the operators :math:`O` are built from quark fields, gamma matrices, and possibly
spatial displacements.  See :ref:`operators-label` for more details on how to constrcut
operators.

Conventions
===========

The adjoint of $\Gamma(p)$ is $\pm \Gamma(-p)$ where the sign depends on the specific
gamma matrix structure.  Currenlty the operators are constructed in Mathematica, we will
choose to track this sign there(its an overall sign).  The momenta will be flipped in the 
current code though, since this will effect what the diagram labels are.
