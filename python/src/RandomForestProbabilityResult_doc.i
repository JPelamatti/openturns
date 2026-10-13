%feature("docstring", "RandomForestProbabilityResult stores the results of a Random Forest probability estimation metamodel.

Parameters
----------
metaModel : :class:`~openturns.Function`, optional
    The probability estimation metamodel mapping input variables to class probabilities.
classValues : :class:`~openturns.Point`, optional
    The distinct class values.
variableImportance : :class:`~openturns.Point`, optional
    The importance score for each input variable.
outOfBagError : float, optional
    The overall out-of-bag prediction error.") OpenTURNS::RandomForestProbabilityResult;

%feature("docstring", "getClassValues()
Accessor to the distinct class labels.

Returns
-------
classValues : :class:`~openturns.Point`
    Unique class target values corresponding to each output dimension.") OpenTURNS::RandomForestProbabilityResult::getClassValues;

%feature("docstring", "getVariableImportance()
Accessor to the variable importance scores.

Returns
-------
variableImportance : :class:`~openturns.Point`
    Importance scores per input feature.") OpenTURNS::RandomForestProbabilityResult::getVariableImportance;

%feature("docstring", "getOutOfBagError()
Accessor to the out-of-bag prediction error.

Returns
-------
outOfBagError : float
    The overall out-of-bag prediction error computed by Ranger.") OpenTURNS::RandomForestProbabilityResult::getOutOfBagError;