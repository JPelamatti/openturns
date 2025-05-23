//                                               -*- C++ -*-
/**
 *  @brief Abstract top-level class for elliptical distributions
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
#ifndef OPENTURNS_ELLIPTICALDISTRIBUTION_HXX
#define OPENTURNS_ELLIPTICALDISTRIBUTION_HXX

#include "openturns/DistributionImplementation.hxx"
#include "openturns/CorrelationMatrix.hxx"

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class EllipticalDistribution
 *
 * A subclass for elliptical usual distributions.
 */
class OT_API EllipticalDistribution
  : public DistributionImplementation
{
  CLASSNAME

public:

  /** Default constructor */
  EllipticalDistribution();

  /** Parameter constructor */
  EllipticalDistribution(const Point & mu,
                         const Point & sigma,
                         const CorrelationMatrix & R,
                         const Scalar covarianceNormalizationFactor);

  /** Virtual copy constructor */
  EllipticalDistribution * clone() const override;

  /** Comparison operator */
protected:
  Bool equals(const DistributionImplementation & other) const override;
public:

  /** String converter */
  String __repr__() const override;

  /** Tell if the distribution is elliptical */
  Bool isElliptical() const override;

  /** Tell if the distribution has elliptical copula */
  Bool hasEllipticalCopula() const override;

  /** Get the DDF of the distribution */
  using DistributionImplementation::computeDDF;
  Point computeDDF(const Point & point) const override;

  /** Get the PDF of the distribution */
  using DistributionImplementation::computePDF;
  Scalar computePDF(const Point & point) const override;
  using DistributionImplementation::computeLogPDF;
  Scalar computeLogPDF(const Point & point) const override;

  /** Get the PDF gradient of the distribution */
  using DistributionImplementation::computePDFGradient;
  Point computePDFGradient(const Point & point) const override;

  /** Compute the density generator of the elliptical distribution, i.e.
   *  the function phi such that the density of the distribution can
   *  be written as p(x) = phi(t(x-mu)R^{-1}(x-mu))                      */
  virtual Scalar computeDensityGenerator(const Scalar betaSquare) const;
  virtual Scalar computeLogDensityGenerator(const Scalar betaSquare) const;

  /** Compute the derivative of the density generator */
  virtual Scalar computeDensityGeneratorDerivative(const Scalar betaSquare) const;

  /** Compute the second derivative of the density generator */
  virtual Scalar computeDensityGeneratorSecondDerivative(const Scalar betaSquare) const;

  /** Compute the survival function */
  using DistributionImplementation::computeSurvivalFunction;
  Scalar computeSurvivalFunction(const Point & point) const override;

  /** Get the minimum volume level set containing a given probability of the distribution */
  LevelSet computeMinimumVolumeLevelSetWithThreshold(const Scalar prob, Scalar & thresholdOut) const override;

  /** Mu accessor */
  void setMu(const Point & mu);
  Point getMu() const;

  /** Sigma vector accessor */
  void setSigma(const Point & sigma);

  /** Sigma vector accessor */
  Point getSigma() const;

  /** Get the standard deviation of the distribution */
  Point getStandardDeviation() const override;

  /** Correlation matrix accessor */
  void setR(const CorrelationMatrix & R);
  CorrelationMatrix getR() const;

protected:
  /** Compute the mean of the distribution */
  void computeMean() const override;

  /** Compute the covariance of the distribution */
  void computeCovariance() const override;

  /** Normalize the given point u_i = (x_i - mu_i) / sigma_i */
  Point normalize(const Point & x) const;

  /** Denormalize the given point x_i = mu_i + sigma_i * x_i */
  Point denormalize(const Point & u) const;

public:
  /** Cholesky factor of the correlation matrix accessor */
  TriangularMatrix getCholesky() const;

  /** Inverse of the Cholesky factor of the correlation matrix accessor */
  TriangularMatrix getInverseCholesky() const;

  /** Get the isoprobabilist transformation */
  IsoProbabilisticTransformation getIsoProbabilisticTransformation() const override;

  /** Get the inverse isoprobabilist transformation */
  InverseIsoProbabilisticTransformation getInverseIsoProbabilisticTransformation() const override;

  /** Get the standard distribution, i.e. a distribution of the same kind but with zero mean,
   * unit marginal standard distribution and identity correlation */
  Distribution getStandardDistribution() const override;

  /** Parameters value and description accessor */
  PointWithDescriptionCollection getParametersCollection() const override;
  using DistributionImplementation::setParametersCollection;
  void setParametersCollection(const PointCollection & parametersCollection) override;

  /** Parameters value accessor */
  Point getParameter() const override;
  void setParameter(const Point & parameters) override;

  /** Parameters description accessor */
  Description getParameterDescription() const override;

  /** Method save() stores the object through the StorageManager */
  void save(Advocate & adv) const override;

  /** Method load() reloads the object from the StorageManager */
  void load(Advocate & adv) override;

protected:

  /** The sigma vector of the distribution */
  Point sigma_;

  /** The correlation matrix (Rij) of the distribution */
  CorrelationMatrix R_;

  /** The inverse Cholesky factor of the covariance matrix */
  TriangularMatrix inverseCholesky_;

  /** The normalization factor of the distribution */
  Scalar normalizationFactor_;

  /** The scaling factor of the covariance matrix covariance = covarianceScalingFactor_ * shape */
  Scalar covarianceScalingFactor_;

private:
  // Class used to wrap the computeRadialCDF() method for interpolation purpose
  class RadialCDFWrapper: public FunctionImplementation
  {
  public:
    RadialCDFWrapper(const EllipticalDistribution * p_distribution)
      : FunctionImplementation()
      , p_distribution_(p_distribution)
    {
      // Nothing to do
    }

    RadialCDFWrapper * clone() const override
    {
      return new RadialCDFWrapper(*this);
    }

    Point operator() (const Point & point) const override
    {
      return Point(1, p_distribution_->computeRadialDistributionCDF(point[0]));
    }

    UnsignedInteger getInputDimension() const override
    {
      return 1;
    }

    UnsignedInteger getOutputDimension() const override
    {
      return 1;
    }

    Description getInputDescription() const override
    {
      return Description(1, "R");
    }

    Description getOutputDescription() const override
    {
      return Description(1, "radialCDF");
    }

    String __repr__() const override
    {
      OSS oss;
      oss << "RadialCDFWrapper(" << p_distribution_->__str__() << ")";
      return oss;
    }

    String __str__(const String & ) const override
    {
      OSS oss;
      oss << "RadialCDFWrapper(" << p_distribution_->__str__() << ")";
      return oss;
    }

  private:
    const EllipticalDistribution * p_distribution_;
  };  // class RadialCDFWrapper

  /** Compute the value of the auxiliary attributes */
  void update();

  /** The shape matrix of the distribution = Diag(sigma_) * R_ * Diag(sigma_) */
  CovarianceMatrix getShape() const;

}; /* class EllipticalDistribution */


END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_ELLIPTICALDISTRIBUTION_HXX */

