//                                               -*- C++ -*-
/**
 *  @brief The test file of class Trapezoidal for standard methods
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
#include "openturns/OT.hxx"
#include "openturns/OTtestcode.hxx"

using namespace OT;
using namespace OT::Test;

class TestObject : public Trapezoidal
{
public:
  TestObject() : Trapezoidal(1.0, 2.0, 3.0, 4.0) {}
  virtual ~TestObject() {}
};


int main(int, char *[])
{
  TESTPREAMBLE;
  OStream fullprint(std::cout);

  try
  {
    // Test basic functionnalities
    checkClassWithClassName<TestObject>();

    Collection< Trapezoidal > coll;
    coll.add(Trapezoidal(1.0, 1.2, 3.0, 14.0));   // a<>b<>c<>d
    coll.add(Trapezoidal(1.0, 1.0, 3.0, 14.0));   // a==b<>c<>d
    coll.add(Trapezoidal(1.0, 1.2, 1.2, 14.0));   // a<>b==c<>d
    coll.add(Trapezoidal(1.0, 1.0, 1.0, 14.0));   // a==b==c<>d
    coll.add(Trapezoidal(1.0, 1.2, 14.0, 14.0));  // a<>b<>c==d
    coll.add(Trapezoidal(1.0, 1.0, 14.0, 14.0));  // a==b<>c==d
    coll.add(Trapezoidal(1.0, 14.0, 14.0, 14.0)); // a<>b==c==d
    Point u = {0.1, 0.01, 0.001, 0.0001, 0.00001};
    Collection< Collection<Complex> > refValues(7);
    refValues[0].add(Complex(8.1648062395158700174e-01, 4.9558226288439289407e-01));
    refValues[0].add(Complex(9.9804203188353452648e-01, 5.4755809968087283756e-02));
    refValues[0].add(Complex(9.9998040760593556200e-01, 5.4810260548082906477e-03));
    refValues[0].add(Complex(9.9999980407478762063e-01, 5.4810805308158094778e-04));
    refValues[0].add(Complex(9.9999999804074774903e-01, 5.4810810755784281107e-05));
    refValues[1].add(Complex(8.1885162171376256180e-01, 4.9039401027722149958e-01));
    refValues[1].add(Complex(9.9806737813236500099e-01, 5.4167952011195103802e-02));
    refValues[1].add(Complex(9.9998066123785714894e-01, 5.4221679269201802622e-03));
    refValues[1].add(Complex(9.9999980661112378576e-01, 5.4222216792666920181e-04));
    refValues[1].add(Complex(9.9999999806611111238e-01, 5.4222222167926666692e-05));
    refValues[2].add(Complex(8.2010530024929859062e-01, 4.8858783460852941056e-01));
    refValues[2].add(Complex(9.9808092313888015117e-01, 5.3946166331792846592e-02));
    refValues[2].add(Complex(9.9998079679235743119e-01, 5.3999461414513856650e-03));
    refValues[2].add(Complex(9.9999980796667923579e-01, 5.3999994614120251386e-04));
    refValues[2].add(Complex(9.9999999807966666792e-01, 5.3999999946141200025e-05));
    refValues[3].add(Complex(8.2278530264722867394e-01, 4.8270899588142185511e-01));
    refValues[3].add(Complex(9.9810957048360230445e-01, 5.3280324745167937987e-02));
    refValues[3].add(Complex(9.9998108345709123351e-01, 5.3332803002475183738e-03));
    refValues[3].add(Complex(9.9999981083334570917e-01, 5.3333328030000247519e-04));
    refValues[3].add(Complex(9.9999999810833333457e-01, 5.3333333280300000025e-05));
    refValues[4].add(Complex(6.7881647967479417697e-01, 6.3873828620489879412e-01));
    refValues[4].add(Complex(9.9645998689553485985e-01, 7.5374711169201023985e-02));
    refValues[4].add(Complex(9.9996456548954769714e-01, 7.5497467239641588934e-03));
    refValues[4].add(Complex(9.9999964565145593167e-01, 7.5498695602548717650e-04));
    refValues[4].add(Complex(9.9999999645651421536e-01, 7.5498707886258037290e-05));
    refValues[5].add(Complex(6.8124331795510156027e-01, 6.3464386336752679037e-01));
    refValues[5].add(Complex(9.9648677930822936181e-01, 7.4876955415600814472e-02));
    refValues[5].add(Complex(9.9996483367809005780e-01, 7.4998768758044346845e-03));
    refValues[5].add(Complex(9.9999964833336780917e-01, 7.4999987687500804437e-04));
    refValues[5].add(Complex(9.9999999648333333678e-01, 7.4999999876875000080e-05));
    refValues[6].add(Complex(5.3970133326297444660e-01, 7.8657873085363172562e-01));
    refValues[6].add(Complex(9.9486398813285641918e-01, 9.6473586086033690957e-02));
    refValues[6].add(Complex(9.9994858389908888208e-01, 9.6664734513613509951e-03));
    refValues[6].add(Complex(9.9999948583338990916e-01, 9.6666647345001361356e-04));
    refValues[6].add(Complex(9.9999999485833333899e-01, 9.6666666473450000136e-05));

    // Instantiate one distribution object
    for (UnsignedInteger nTrapezoidal = 0; nTrapezoidal < coll.getSize(); ++nTrapezoidal)
    {
      // Instantiate one distribution object
      Trapezoidal distribution(coll[nTrapezoidal]);
      fullprint << "Distribution " << distribution << std::endl;
      std::cout << "Distribution " << distribution << std::endl;

      // Is this distribution elliptical ?
      fullprint << "Elliptical = " << (distribution.isElliptical() ? "true" : "false") << std::endl;

      // Is this distribution continuous ?
      fullprint << "Continuous = " << (distribution.isContinuous() ? "true" : "false") << std::endl;

      // Test for realization of distribution
      Point oneRealization = distribution.getRealization();
      fullprint << "oneRealization=" << oneRealization << std::endl;

      // Test for sampling
      UnsignedInteger size = 10000;
      Sample oneSample = distribution.getSample( size );
      fullprint << "oneSample first=" << oneSample[0] << " last=" << oneSample[size - 1] << std::endl;
      fullprint << "mean=" << oneSample.computeMean() << std::endl;
      fullprint << "covariance=" << oneSample.computeCovariance() << std::endl;
      fullprint << "skewness=" << oneSample.computeSkewness() << std::endl;
      fullprint << "kurtosis=" << oneSample.computeKurtosis() << std::endl;
      size = 100;
      for (UnsignedInteger i = 0; i < 2; ++i)
      {
        fullprint << "Kolmogorov test for the generator, sample size=" << size << " is " << (FittingTest::Kolmogorov(distribution.getSample(size), distribution).getBinaryQualityMeasure() ? "accepted" : "rejected") << std::endl;
        size *= 10;
      }

      // Define a point
      Point point(distribution.getDimension(), 1.1);
      fullprint << "Point= " << point << std::endl;

      // Show PDF and CDF of point
      Scalar eps = 1e-5;
      Point DDF = distribution.computeDDF( point );
      fullprint << "ddf     =" << DDF << std::endl;
      Scalar LPDF = distribution.computeLogPDF( point );
      fullprint << "log pdf=" << LPDF << std::endl;
      Scalar PDF = distribution.computePDF( point );
      fullprint << "pdf     =" << PDF << std::endl;
      fullprint << "pdf (FD)=" << (distribution.computeCDF( point + Point(1, eps) ) - distribution.computeCDF( point  + Point(1, -eps) )) / (2.0 * eps) << std::endl;
      Scalar CDF = distribution.computeCDF( point );
      fullprint << "cdf=" << CDF << std::endl;
      Scalar CCDF = distribution.computeComplementaryCDF( point );
      fullprint << "ccdf=" << CCDF << std::endl;
      Scalar Survival = distribution.computeSurvivalFunction( point );
      fullprint << "survival=" << Survival << std::endl;
      Complex CF = distribution.computeCharacteristicFunction( point[0] );
      Point InverseSurvival = distribution.computeInverseSurvivalFunction(0.95);
      fullprint << "Inverse survival=" << InverseSurvival << std::endl;
      fullprint << "Survival(inverse survival)=" << distribution.computeSurvivalFunction(InverseSurvival) << std::endl;

      fullprint << "characteristic function=" << CF << std::endl;
      Complex LCF = distribution.computeLogCharacteristicFunction( point[0] );
      fullprint << "log characteristic function=" << LCF << std::endl;
      for (UnsignedInteger j = 0; j < refValues[nTrapezoidal].getSize(); ++j)
        assert_almost_equal(distribution.computeCharacteristicFunction(u[j]), refValues[nTrapezoidal][j]);
      try
      {
        Point PDFgr = distribution.computePDFGradient( point );
        fullprint << "pdf gradient     =" << PDFgr << std::endl;
        Point PDFgrFD(4);
        PDFgrFD[0] = (Trapezoidal(distribution.getA() + eps, distribution.getB(), distribution.getC(), distribution.getD()).computePDF(point) -
                      Trapezoidal(distribution.getA() - eps, distribution.getB(), distribution.getC(), distribution.getD()).computePDF(point)) / (2.0 * eps);
        PDFgrFD[1] = (Trapezoidal(distribution.getA(), distribution.getB() + eps, distribution.getC(), distribution.getD()).computePDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB() - eps, distribution.getC(), distribution.getD()).computePDF(point)) / (2.0 * eps);
        PDFgrFD[2] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() + eps, distribution.getD()).computePDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() - eps, distribution.getD()).computePDF(point)) / (2.0 * eps);
        PDFgrFD[3] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() + eps).computePDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() - eps).computePDF(point)) / (2.0 * eps);
        fullprint << "pdf gradient (FD)=" << PDFgrFD << std::endl;
        Point logPDFgr = distribution.computeLogPDFGradient( point );
        fullprint << "log-pdf gradient     =" << logPDFgr << std::endl;
        Point logPDFgrFD(4);
        logPDFgrFD[0] = (Trapezoidal(distribution.getA() + eps, distribution.getB(), distribution.getC(), distribution.getD()).computeLogPDF(point) -
                         Trapezoidal(distribution.getA() - eps, distribution.getB(), distribution.getC(), distribution.getD()).computeLogPDF(point)) / (2.0 * eps);
        logPDFgrFD[1] = (Trapezoidal(distribution.getA(), distribution.getB() + eps, distribution.getC(), distribution.getD()).computeLogPDF(point) -
                         Trapezoidal(distribution.getA(), distribution.getB() - eps, distribution.getC(), distribution.getD()).computeLogPDF(point)) / (2.0 * eps);
        logPDFgrFD[2] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() + eps, distribution.getD()).computeLogPDF(point) -
                         Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() - eps, distribution.getD()).computeLogPDF(point)) / (2.0 * eps);
        logPDFgrFD[3] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() + eps).computeLogPDF(point) -
                         Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() - eps).computeLogPDF(point)) / (2.0 * eps);
        fullprint << "log-pdf gradient (FD)=" << logPDFgrFD << std::endl;
        Point CDFgr = distribution.computeCDFGradient( point );
        fullprint << "cdf gradient     =" << CDFgr << std::endl;
        Point CDFgrFD(4);
        CDFgrFD[0] = (Trapezoidal(distribution.getA() + eps, distribution.getB(), distribution.getC(), distribution.getD()).computeCDF(point) -
                      Trapezoidal(distribution.getA() - eps, distribution.getB(), distribution.getC(), distribution.getD()).computeCDF(point)) / (2.0 * eps);
        CDFgrFD[1] = (Trapezoidal(distribution.getA(), distribution.getB() + eps, distribution.getC(), distribution.getD()).computeCDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB() - eps, distribution.getC(), distribution.getD()).computeCDF(point)) / (2.0 * eps);
        CDFgrFD[2] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() + eps, distribution.getD()).computeCDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC() - eps, distribution.getD()).computeCDF(point)) / (2.0 * eps);
        CDFgrFD[3] = (Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() + eps).computeCDF(point) -
                      Trapezoidal(distribution.getA(), distribution.getB(), distribution.getC(), distribution.getD() - eps).computeCDF(point)) / (2.0 * eps);
        fullprint << "cdf gradient (FD)=" << CDFgrFD << std::endl;
      }
      catch (const NotDefinedException &)
      {
      }
      Point quantile = distribution.computeQuantile( 0.95 );
      fullprint << "quantile=" << quantile << std::endl;
      fullprint << "cdf(quantile)=" << distribution.computeCDF(quantile) << std::endl;
      // Confidence regions
      Scalar threshold;
      fullprint << "Minimum volume interval=" << distribution.computeMinimumVolumeIntervalWithMarginalProbability(0.95, threshold) << std::endl;
      fullprint << "threshold=" << threshold << std::endl;
      Scalar beta;
      LevelSet levelSet(distribution.computeMinimumVolumeLevelSetWithThreshold(0.95, beta));
      fullprint << "Minimum volume level set=" << levelSet << std::endl;
      fullprint << "beta=" << beta << std::endl;
      fullprint << "Bilateral confidence interval=" << distribution.computeBilateralConfidenceIntervalWithMarginalProbability(0.95, beta) << std::endl;
      fullprint << "beta=" << beta << std::endl;
      fullprint << "Unilateral confidence interval (lower tail)=" << distribution.computeUnilateralConfidenceIntervalWithMarginalProbability(0.95, false, beta) << std::endl;
      fullprint << "beta=" << beta << std::endl;
      fullprint << "Unilateral confidence interval (upper tail)=" << distribution.computeUnilateralConfidenceIntervalWithMarginalProbability(0.95, true, beta) << std::endl;
      fullprint << "beta=" << beta << std::endl;
      fullprint << "entropy=" << distribution.computeEntropy() << std::endl;
      fullprint << "entropy (MC)=" << -distribution.computeLogPDF(distribution.getSample(1000000)).computeMean()[0] << std::endl;
      Point mean = distribution.getMean();
      fullprint << "mean=" << mean << std::endl;
      Point standardDeviation = distribution.getStandardDeviation();
      fullprint << "standard deviation=" << standardDeviation << std::endl;
      Point skewness = distribution.getSkewness();
      fullprint << "skewness=" << skewness << std::endl;
      Point kurtosis = distribution.getKurtosis();
      fullprint << "kurtosis=" << kurtosis << std::endl;
      CovarianceMatrix covariance = distribution.getCovariance();
      fullprint << "covariance=" << covariance << std::endl;
      CovarianceMatrix correlation = distribution.getCorrelation();
      fullprint << "correlation=" << correlation << std::endl;
      CovarianceMatrix spearman = distribution.getSpearmanCorrelation();
      fullprint << "spearman=" << spearman << std::endl;
      CovarianceMatrix kendall = distribution.getKendallTau();
      fullprint << "kendall=" << kendall << std::endl;
      Trapezoidal::PointWithDescriptionCollection parameters = distribution.getParametersCollection();
      fullprint << "parameters=" << parameters << std::endl;
      fullprint << "Standard representative=" << distribution.getStandardRepresentative().__str__() << std::endl;

      Scalar roughness = distribution.getRoughness();
      fullprint << "roughness=" << roughness << std::endl;
    }
  }
  catch (const TestFailed & ex)
  {
    std::cerr << ex << std::endl;
    return ExitCode::Error;
  }


  return ExitCode::Success;
}
