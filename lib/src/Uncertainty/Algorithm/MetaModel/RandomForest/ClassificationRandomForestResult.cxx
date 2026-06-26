//                                               -*- C++ -*-
/**
 *  @brief The result of a classification random forest estimation
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
#include "openturns/ClassificationRandomForestResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"
#include "openturns/OSS.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(ClassificationRandomForestResult)
static const Factory<ClassificationRandomForestResult> Factory_ClassificationRandomForestResult;

/* Default constructor */
ClassificationRandomForestResult::ClassificationRandomForestResult()
  : MetaModelResult()
  , classValues_()
  , oobError_(0.0)
{
  // Nothing to do
}

/* Parameter constructor */
ClassificationRandomForestResult::ClassificationRandomForestResult(
    const Sample   & inputSample,
    const Sample   & outputSample,
    const Function & metaModel,
    const Point    & classValues,
    Scalar           oobError)
  : MetaModelResult(inputSample, outputSample, metaModel)
  , classValues_(classValues)
  , oobError_(oobError)
{
  // Nothing to do
}

/* Virtual constructor */
ClassificationRandomForestResult * ClassificationRandomForestResult::clone() const
{
  return new ClassificationRandomForestResult(*this);
}

/* Accessors */
Point ClassificationRandomForestResult::getClassValues() const
{
  return classValues_;
}

Scalar ClassificationRandomForestResult::getOOBError() const
{
  return oobError_;
}

/* Method save() stores the object through the StorageManager */
void ClassificationRandomForestResult::save(Advocate & adv) const
{
  MetaModelResult::save(adv);
  adv.saveAttribute("classValues_", classValues_);
  adv.saveAttribute("oobError_",    oobError_);
}

/* Method load() reloads the object from the StorageManager */
void ClassificationRandomForestResult::load(Advocate & adv)
{
  MetaModelResult::load(adv);
  adv.loadAttribute("classValues_", classValues_);
  adv.loadAttribute("oobError_",    oobError_);
}

END_NAMESPACE_OPENTURNS