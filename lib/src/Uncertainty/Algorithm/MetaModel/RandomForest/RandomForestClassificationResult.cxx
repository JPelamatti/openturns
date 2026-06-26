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
#include "openturns/RandomForestClassificationResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(RandomForestClassificationResult)
static const Factory<RandomForestClassificationResult>
    Factory_RandomForestClassificationResult;

/* Default constructor */
RandomForestClassificationResult::RandomForestClassificationResult()
    : MetaModelResult() {}

/*Parameter constructor */
RandomForestClassificationResult::RandomForestClassificationResult(
    const Sample &inputSample, const Sample &outputSample,
    const Function &metaModel, const Scalar outOfBagError,
    const Point &variableImportance)
    : MetaModelResult(inputSample, outputSample, metaModel),
      outOfBagError_(outOfBagError),
      variableImportance_(variableImportance) {
  // Nothing to do
}

/* Virtual constructor */
RandomForestClassificationResult *RandomForestClassificationResult::clone()
    const {
  return new RandomForestClassificationResult(*this);
}

Scalar RandomForestClassificationResult::getOutOfBagError() const {
  return outOfBagError_;
}

Point RandomForestClassificationResult::getVariableImportance() const {
  return variableImportance_;
}

END_NAMESPACE_OPENTURNS