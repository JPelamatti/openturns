// SWIG file RandomForestClassificationResult.i

%{
#include "openturns/RandomForestClassificationResult.hxx"
%}

%include RandomForestClassificationResult_doc.i

//%rename(RandomForestClassificationResult_operator___eq__) OT::operator ==(const RandomForestClassificationResult & lhs, const RandomForestClassificationResult & rhs);

%copyctor OT::RandomForestClassificationResult;

%include openturns/RandomForestClassificationResult.hxx
