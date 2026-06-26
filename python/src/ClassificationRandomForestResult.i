// SWIG file ClassificationRandomForestResult.i

%{
#include "openturns/ClassificationRandomForestResult.hxx"
%}

%include ClassificationRandomForestResult_doc.i

//%rename(ClassificationRandomForestResult_operator___eq__) OT::operator ==(const ClassificationRandomForestResult & lhs, const ClassificationRandomForestResult & rhs);

%copyctor OT::ClassificationRandomForestResult;

%include openturns/ClassificationRandomForestResult.hxx
