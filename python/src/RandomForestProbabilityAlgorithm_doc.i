%feature("docstring", "RandomForestProbabilityAlgorithm builds random forests for class probability estimation via Ranger.

Parameters
----------
inputSample : :class:`~openturns.Sample`
    Input training features.
outputSample : :class:`~openturns.Sample`
    Categorical class output labels (1-dimensional).
importanceMode : int, optional
    Variable importance mode (0: none, 1: impurity, 2: impurity corrected, 3: permutation). Default is 0.
num_trees : int, optional
    Number of trees in the forest. Default is 500.
min_node_size : int, optional
    Minimal node size. Default is 0 (uses Ranger defaults).
min_bucket : int, optional
    Minimal bucket size. Default is 0.
max_depth : int, optional
    Maximal tree depth (0 for unbounded). Default is 0.
mtry : int, optional
    Number of variables randomly sampled at each split (0 for default sqrt(p)). Default is 0.") OpenTURNS::RandomForestProbabilityAlgorithm;

%feature("docstring", "run()
Train the probability forest model.") OpenTURNS::RandomForestProbabilityAlgorithm::run;

%feature("docstring", "predict(inputSample)
Estimate class probabilities for input points.

Parameters
----------
inputSample : :class:`~openturns.Sample`
    Input points to evaluate.

Returns
-------
probabilities : :class:`~openturns.Sample`
    Class probabilities matrix where each column corresponds to a class.") OpenTURNS::RandomForestProbabilityAlgorithm::predict;

%feature("docstring", "getClassValues()
Accessor to the distinct class labels.

Returns
-------
classValues : :class:`~openturns.Point`
    Array of class labels associated with each probability output column.") OpenTURNS::RandomForestProbabilityAlgorithm::getClassValues;

%feature("docstring", "getResult()
Accessor to the metamodel estimation result.

Returns
-------
result : :class:`~openturns.RandomForestProbabilityResult`
    The trained probability forest metamodel result.") OpenTURNS::RandomForestProbabilityAlgorithm::getResult;