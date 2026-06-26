// SWIG file RandomForestRegressionResult.i

%{
#include "openturns/RanndomForestResult.hxx"
%}

%include RandomForestRegressionResult_doc.i

//%rename(RandomForestRegressionResult_operator___eq__) OT::operator ==(const RandomForestRegressionResult & lhs, const RandomForestRegressionResult & rhs);

%copyctor OT::RandomForestRegressionResult;

%include openturns/RandomForestRegressionResult.hxx
