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
#ifndef OPENTURNS_RANDOMFORESTPROBABILITYRESULT_HXX
#define OPENTURNS_RANDOMFORESTPROBABILITYRESULT_HXX

#include "openturns/MetaModelResult.hxx"
#include "openturns/Point.hxx"
#include "openturns/Function.hxx"

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class RandomForestProbabilityResult
 *
 * Result of a Random Forest probability estimation metamodel.
 */
class OT_API RandomForestProbabilityResult
  : public MetaModelResult
{
  CLASSNAME

public:
  /** Default constructor */
  RandomForestProbabilityResult();

  /** Constructor with metamodel and class values */
  RandomForestProbabilityResult(const Function & metaModel,
                                const Point & classValues);

  /** Constructor with metamodel, class values, variable importance, and out-of-bag error */
  RandomForestProbabilityResult(const Function & metaModel,
                                const Point & classValues,
                                const Point & variableImportance,
                                const Scalar outOfBagError);

  /** Virtual constructor */
  RandomForestProbabilityResult * clone() const override;

  /** Class values accessor */
  Point getClassValues() const;
  void setClassValues(const Point & classValues);

  /** Variable importance accessor */
  Point getVariableImportance() const;
  void setVariableImportance(const Point & variableImportance);

  /** Out-of-bag prediction error accessor */
  Scalar getOutOfBagError() const;
  void setOutOfBagError(const Scalar outOfBagError);

  /** String representation */
  String __repr__() const override;
  String __str__(const String & offset = "") const override;

  /** Method save() stores the object through the StorageManager */
  void save(Advocate & adv) const override;

  /** Method load() reloads the object from the StorageManager */
  void load(Advocate & adv) override;

private:
  Point classValues_;
  Point variableImportance_;
  Scalar outOfBagError_;
};

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_RANDOMFORESTPROBABILITYRESULT_HXX */