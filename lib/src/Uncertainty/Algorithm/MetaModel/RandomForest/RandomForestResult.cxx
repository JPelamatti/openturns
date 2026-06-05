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
#include "openturns/RandomForestResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"
#include "openturns/OSS.hxx"
#include "openturns/MatrixImplementation.hxx"
#include "openturns/SampleImplementation.hxx"
#include "openturns/LeastSquaresMethod.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(RandomForestResult)
static const Factory<RandomForestResult> Factory_RandomForestResult;

/* Default constructor */
RandomForestResult::RandomForestResult()
  : MetaModelResult()
{
  // Nothing to do
}

/*Parameter constructor */
RandomForestResult::RandomForestResult(const Sample & inputSample,
                                      const Sample & outputSample,
                                      const Function & metaModel)
  : MetaModelResult(inputSample, outputSample, metaModel)
{
  // Nothing to do
}

/* Virtual constructor */
RandomForestResult * RandomForestResult::clone() const
{
  return new RandomForestResult(*this);
}

/* Method save() stores the object through the StorageManager */
void RandomForestResult::save(Advocate & adv) const
{
  MetaModelResult::save(adv);
}


/* Method load() reloads the object from the StorageManager */
void RandomForestResult::load(Advocate & adv)
{
  MetaModelResult::load(adv);
}

END_NAMESPACE_OPENTURNS
