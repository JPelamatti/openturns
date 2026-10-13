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
#include "openturns/RandomForestProbabilityResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(RandomForestProbabilityResult)

static const Factory<RandomForestProbabilityResult> Factory_RandomForestProbabilityResult;

RandomForestProbabilityResult::RandomForestProbabilityResult()
  : MetaModelResult()
  , classValues_()
  , variableImportance_()
  , outOfBagError_(0.0)
{
}

RandomForestProbabilityResult::RandomForestProbabilityResult(const Function & metaModel,
                                                             const Point & classValues)
  : MetaModelResult(metaModel)
  , classValues_(classValues)
  , variableImportance_()
  , outOfBagError_(0.0)
{
}

RandomForestProbabilityResult::RandomForestProbabilityResult(const Function & metaModel,
                                                             const Point & classValues,
                                                             const Point & variableImportance,
                                                             const Scalar outOfBagError)
  : MetaModelResult(metaModel)
  , classValues_(classValues)
  , variableImportance_(variableImportance)
  , outOfBagError_(outOfBagError)
{
}

RandomForestProbabilityResult * RandomForestProbabilityResult::clone() const
{
  return new RandomForestProbabilityResult(*this);
}

Point RandomForestProbabilityResult::getClassValues() const
{
  return classValues_;
}

void RandomForestProbabilityResult::setClassValues(const Point & classValues)
{
  classValues_ = classValues;
}

Point RandomForestProbabilityResult::getVariableImportance() const
{
  return variableImportance_;
}

void RandomForestProbabilityResult::setVariableImportance(const Point & variableImportance)
{
  variableImportance_ = variableImportance;
}

Scalar RandomForestProbabilityResult::getOutOfBagError() const
{
  return outOfBagError_;
}

void RandomForestProbabilityResult::setOutOfBagError(const Scalar outOfBagError)
{
  outOfBagError_ = outOfBagError;
}

String RandomForestProbabilityResult::__repr__() const
{
  return OSS() << "class=" << RandomForestProbabilityResult::GetClassName()
               << " metaModel=" << getMetaModel()
               << " classValues=" << classValues_
               << " outOfBagError=" << outOfBagError_;
}

String RandomForestProbabilityResult::__str__(const String & offset) const
{
  return OSS() << offset << __repr__();
}

void RandomForestProbabilityResult::save(Advocate & adv) const
{
  MetaModelResult::save(adv);
  adv.saveAttribute("classValues_", classValues_);
  adv.saveAttribute("variableImportance_", variableImportance_);
  adv.saveAttribute("outOfBagError_", outOfBagError_);
}

void RandomForestProbabilityResult::load(Advocate & adv)
{
  MetaModelResult::load(adv);
  adv.loadAttribute("classValues_", classValues_);
  adv.loadAttribute("variableImportance_", variableImportance_);
  adv.loadAttribute("outOfBagError_", outOfBagError_);
}

END_NAMESPACE_OPENTURNS