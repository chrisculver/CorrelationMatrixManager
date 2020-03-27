Input
=====

There are three sources of file input to the program.



Runtime Parameters
------------------
The complete list of runtime parameters is specified in the 
``pion_input.txt`` file in the ``EXAMPLES`` directory.  This file is passed as the
first argument to the executable ``./compute_correlation_matrix pion_input.txt``.  

The input file has a row for each variable, with the first column specifying the name
of the variable, the second column specifying the value, and the delimeter between the columns being
a single space.  This is used to create a map between the names and values.

The code first checks that the file exists, then begins reading it.
Then a check is performed to make sure the configuration number ``cfg`` is specified, 
which labels log files.  As each data element is placed into memory, the correspdonding map 
element is deleted.  We then check that the map has no extraneous parameters specified - which
could be ones with typos. 

TODO : We should probably just check the map for EACH named variable.  and exit if any of the ones
that cannot have a default value are absent.  (So far I think only the log level makes sense to have
a default value).  


Operator List
-------------
Currently this code takes an input of operators, for examples see ``simple_pions.op`` and 
``multi_pions.op`` for an example of some multi-meson operators.  See physics/operators.rst
for a full description of how operators are specified.



Diagram File(s)
---------------
