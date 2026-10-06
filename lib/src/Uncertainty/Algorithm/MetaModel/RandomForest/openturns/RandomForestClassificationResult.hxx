//                                               -*- C++ -*-
/**
 *  @brief The result of a random forest classification model estimation
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
#ifndef OPENTURNS_RANDOMFORESTCLASSIFICATIONRESULT_HXX
#define OPENTURNS_RANDOMFORESTCLASSIFICATIONRESULT_HXX

#include "openturns/MetaModelResult.hxx"

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class RandomForestClassificationResult
 *
 * The result of a random forest classification model evaluation
 */

class OT_API RandomForestClassificationResult : public MetaModelResult {
  CLASSNAME

 public:
  /** Default constructor */
  RandomForestClassificationResult();

  /** Parameter constructor */
  RandomForestClassificationResult(const Sample &inputSample,
                                   const Sample &outputSample,
                                   const Function &metaModel,
                                   const Scalar outOfBagError,
                                   const Point &variableImportance);

  /** Virtual constructor */
  RandomForestClassificationResult *clone() const override;

  /** Accessor for the out of bag error */
  Scalar getOutOfBagError() const;
  Point getVariableImportance() const;

 private:
  Scalar outOfBagError_;
  Point variableImportance_;
}; /* class RandomForestClassificationResult */

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_RANDOMFORESTCLASSIFICATIONRESULT_HXX */