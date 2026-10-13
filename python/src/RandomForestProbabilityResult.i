//                                               -*- C++ -*-
/**
 *  @brief Implement Ranger wrapping for random forests probability result
 *
 *  Copyright 2005-2026 Airbus-EDF-IMACS-ONERA-Phimeca
 *
 *  This library is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this library.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef OPENTURNS_RANDOMFORESTPROBABILITYRESULT_I
#define OPENTURNS_RANDOMFORESTPROBABILITYRESULT_I

%{
#include "openturns/RandomForestProbabilityResult.hxx"
%}

%include RandomForestProbabilityResult_doc.i

namespace OpenTURNS {

class RandomForestProbabilityResult
  : public MetaModelResultImplementation
{
public:
  RandomForestProbabilityResult();
  RandomForestProbabilityResult(const Function & metaModel,
                                const Point & classValues);
  RandomForestProbabilityResult(const Function & metaModel,
                                const Point & classValues,
                                const Point & variableImportance,
                                const Scalar outOfBagError);

  Point getClassValues() const;
  void setClassValues(const Point & classValues);

  Point getVariableImportance() const;
  void setVariableImportance(const Point & variableImportance);

  Scalar getOutOfBagError() const;
  void setOutOfBagError(const Scalar outOfBagError);
};

} // namespace OpenTURNS

#endif /* OPENTURNS_RANDOMFORESTPROBABILITYRESULT_I */