// SWIG file RandomForestResult.i

%{
#include "openturns/RanndomForestResult.hxx"
%}

%include RandomForestResult_doc.i

//%rename(RandomForestResult_operator___eq__) OT::operator ==(const RandomForestResult & lhs, const RandomForestResult & rhs);

%copyctor OT::RandomForestResult;

%include openturns/RandomForestResult.hxx
