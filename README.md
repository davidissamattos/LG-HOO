#Limited Growth Hierarchical Optimistic Optimizer

This algorithm is a modification of the proposed algorithm by Sebastian Bubeck et al. in http://www.jmlr.org/papers/volume12/bubeck11a/bubeck11a.pdf, that allows better control and arm selection for online continuous experiments.

In particular, this algorithm restricts how each leaf can grow by setting a minimum number of iterations before it is allowed to grow and limits the size of the tree
Setting a minimum number of iterations before allowing the leaf to grow has several implications in the performance. It limits the growth size of the tree while allowing the tree to grow more in the correct directions.
This comes with the expense of needing of more users to make get more refinement.

Additionally this modification adds an arm selection criteria when it is time to stop the experiment.

The simulation.py file allows you to test the algorithm with different parameters and different underlying functions.

The class hoo in the hoo.py contains several auxiliary functions including plot functions that you can plot the tree graph together with the underlying function.

The dependencies of this project are:
* numpy
* networkx for making graph plots
* scipy for some of the underlying functions