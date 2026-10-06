%feature("docstring") OT::ClassificationRandomForestAlgorithm
"Classification Random Forest algorithm.

This class wraps the `ranger::ForestClassification` algorithm.

It is used to train a classification random forest model from a pair of input
and output samples. The output sample must contain integer-valued class
labels encoded as `Scalar` (e.g., 0.0, 1.0, 2.0, ...).

Parameters
----------
inputSample : 2-d sequence of float
    Training inputs, of shape `(n_samples, n_features)`.
outputSample : 2-d sequence of float
    Training outputs (class labels), of shape `(n_samples, 1)`.

Examples
--------
>>> import openturns as ot
>>> from openturns.viewer import View
>>> N = 500
>>> # Create data from a XOR problem
>>> dist = ot.ComposedDistribution([ot.Normal()] * 2)
>>> inSample = dist.getSample(N)
>>> outSample = ot.Sample(N, 1)
>>> for i in range(N):
...     outSample[i,0] = 1.0 if (inSample[i,0] * inSample[i,1] > 0.0) else 0.0
>>> # Create and run the algorithm
>>> algo = ot.ClassificationRandomForestAlgorithm(inSample, outSample)
>>> algo.run()
>>> # Get the result
>>> result = algo.getResult()
>>> # Get the metamodel
>>> metamodel = result.getMetaModel()"

// ---------------------------------------------------------------------

%feature("docstring") OT::ClassificationRandomForestAlgorithm::run
"Train the model."

// ---------------------------------------------------------------------

%feature("docstring") OT::ClassificationRandomForestAlgorithm::predict
"Predict class labels for new inputs.

Parameters
----------
inputSample : 2-d sequence of float
    The input sample for which to predict the output.

Returns
-------
outputSample : :class:`~openturns.Sample`
    The predicted class labels."
    
// ---------------------------------------------------------------------

%feature("docstring") OT::ClassificationRandomForestAlgorithm::getRandomForestAsFunction
"Get the trained model as a function.

Returns
-------
function : :class:`~openturns.Function`
    The metamodel, which predicts the class label for a new input."
