
//                                               -*- C++ -*-
/**
 * @brief HSICStatImplementation implements the HSIC sensivity index for
 *        one marginal.
 *
 *  Copyright 2005-2025 Airbus-EDF-IMACS-ONERA-Phimeca
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
#ifndef OPENTURNS_HSICSTATIMPLEMENTATION_HXX
#define OPENTURNS_HSICSTATIMPLEMENTATION_HXX

#include "openturns/PersistentObject.hxx"
#include "openturns/CovarianceModel.hxx"
#include "openturns/Gamma.hxx"
#include "openturns/Point.hxx"

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class HSICStatImplementation models an atomic HSIC index over one marginal.
 *
 */
class OT_API HSICStatImplementation
  : public PersistentObject
{
  CLASSNAME
public:
  /** Default constructor */
  HSICStatImplementation();

  /** Here is the interface that all derived class must implement */

  /** Virtual constructor */
  HSICStatImplementation * clone() const override;

  /** Compute the HSIC index for one marginal*/
  virtual Scalar computeHSICIndex(const CovarianceMatrix & covarianceMatrix1,
                                  const CovarianceMatrix & covarianceMatrix2,
                                  const SquareMatrix & weightMatrix) const;

  /** Compute the HSIC index for one marginal*/
  virtual Scalar computeHSICIndex(const CovarianceMatrix &covarianceMatrix1,
                                  const CovarianceMatrix &covarianceMatrix2,
                                  const Point & weights) const;

  /** Compute the asymptotic p-value */
  virtual Scalar computePValue(const Gamma & distribution,
                               const UnsignedInteger n,
                               const Scalar HSICObs,
                               const Scalar mHSIC) const;

  /** Is compatible with a Conditional HSIC Estimator ? */
  virtual Bool isCompatibleWithConditionalAnalysis() const;

};

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_HSICSTATIMPLEMENTATION_HXX */
