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
#ifndef OPENTURNS_CLASSIFICATIONRANDOMFORESTRESULT_HXX
#define OPENTURNS_CLASSIFICATIONRANDOMFORESTRESULT_HXX

#include "openturns/MetaModelResult.hxx"
#include "openturns/Point.hxx"

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class ClassificationRandomForestResult
 *
 * Stores the artefacts produced by ClassificationRandomForestAlgorithm::run():
 *  - the trained meta-model Function (predicts the most-probable class label),
 *  - the ordered list of class values seen during training,
 *  - the OOB (out-of-bag) error rate.
 *
 * These extra fields are serialised so that the result can be saved and
 * reloaded through the standard OpenTURNS StorageManager mechanism.
 */
class OT_API ClassificationRandomForestResult
  : public MetaModelResult
{
  CLASSNAME

public:
  /** Default constructor */
  ClassificationRandomForestResult();

  /**
   * Parameter constructor
   *
   * @param inputSample   Training inputs  (n × d)
   * @param outputSample  Training outputs (n × 1) — class labels
   * @param metaModel     Function returning the predicted class label
   * @param classValues   Ordered vector of distinct class values found
   *                      during training (same order as Ranger's internal map)
   * @param oobError      Out-of-bag classification error rate ∈ [0, 1]
   */
  ClassificationRandomForestResult(const Sample    & inputSample,
                                   const Sample    & outputSample,
                                   const Function  & metaModel,
                                   const Point     & classValues,
                                   Scalar            oobError = 0.0);

  /** Virtual constructor */
  ClassificationRandomForestResult * clone() const override;

  // ---- Accessors -------------------------------------------------------

  /**
   * Ordered list of class values as discovered by Ranger.
   * The k-th entry corresponds to class index k used in predictProbabilities().
   */
  Point getClassValues() const;

  /**
   * Out-of-bag classification error rate (proportion of misclassified
   * OOB samples), as returned by Ranger after training.
   */
  Scalar getOOBError() const;

  // ---- Persistence -----------------------------------------------------

  /** Method save() stores the object through the StorageManager */
  void save(Advocate & adv) const override;

  /** Method load() reloads the object from the StorageManager */
  void load(Advocate & adv) override;

private:
  /** Distinct class labels (numeric encoding), ordered as in Ranger */
  Point  classValues_;
  /** Out-of-bag classification error rate */
  Scalar oobError_;

}; /* class ClassificationRandomForestResult */

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_CLASSIFICATIONRANDOMFORESTRESULT_HXX */