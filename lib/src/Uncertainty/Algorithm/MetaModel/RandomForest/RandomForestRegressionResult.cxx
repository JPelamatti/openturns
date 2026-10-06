//                                               -*- C++ -*-
/**
 *  @brief The result of a linear model estimation
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
#include "openturns/RandomForestRegressionResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"
#include "openturns/OSS.hxx"
#include "openturns/MatrixImplementation.hxx"
#include "openturns/SampleImplementation.hxx"
#include "openturns/LeastSquaresMethod.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(RandomForestRegressionResult)
static const Factory<RandomForestRegressionResult> Factory_RandomForestRegressionResult;

/* Default constructor */
RandomForestRegressionResult::RandomForestRegressionResult()
  : MetaModelResult()
{
  // Nothing to do
}

/*Parameter constructor */
RandomForestRegressionResult::RandomForestRegressionResult(const Sample & inputSample,
                                      const Sample & outputSample,
                                      const Function & metaModel,
                                      const Scalar outOfBagError,
                                      const Point &variableImportance)
  : MetaModelResult(inputSample, outputSample, metaModel)
  , outOfBagError_(outOfBagError)
  , variableImportance_(variableImportance)
  {
    // Nothing to do
}

/* Virtual constructor */
RandomForestRegressionResult * RandomForestRegressionResult::clone() const
{
  return new RandomForestRegressionResult(*this);
}

/* Method save() stores the object through the StorageManager */
void RandomForestRegressionResult::save(Advocate & adv) const
{
  MetaModelResult::save(adv);
}

/* Method load() reloads the object from the StorageManager */
void RandomForestRegressionResult::load(Advocate & adv)
{
  MetaModelResult::load(adv);
}

Scalar RandomForestRegressionResult::getOutOfBagError() const
{
  return outOfBagError_;
}

Point RandomForestRegressionResult::getVariableImportance() const
{
  return variableImportance_;
}

END_NAMESPACE_OPENTURNS
