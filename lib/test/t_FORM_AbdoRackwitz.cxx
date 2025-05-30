//                                               -*- C++ -*-
/**
 *  @brief The test file of FORM class
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


int main(int, char *[])
{
  TESTPREAMBLE;
  OStream fullprint(std::cout);

  try
  {

    /* We create a numerical math function */
    Description input(4);
    input[0] = "E";
    input[1] = "F";
    input[2] = "L";
    input[3] = "I";
    SymbolicFunction myFunction(input, Description(1, "-F*L^3/(3*E*I)"));

    UnsignedInteger dim = myFunction.getInputDimension();
    /* We create a normal distribution point of dimension 1 */
    Point mean(dim, 0.0);
    mean[0] = 50.0; // E
    mean[1] =  1.0; // F
    mean[2] = 10.0; // L
    mean[3] =  5.0; // I
    Point sigma(dim, 1.0);
    IdentityMatrix R(dim);
    Normal myDistribution(mean, sigma, R);

    /* We create a 'usual' RandomVector from the Distribution */
    RandomVector vect(myDistribution);

    /* We create a composite random vector */
    CompositeRandomVector output(myFunction, vect);

    /* We create an Event from this RandomVector */
    ThresholdEvent myEvent(output, Less(), -3.0);

    /* We create a NearestPoint algorithm */
    AbdoRackwitz myAlgorithm;
    myAlgorithm.setStartingPoint(mean);
    myAlgorithm.setMaximumIterationNumber(100);
    myAlgorithm.setMaximumAbsoluteError(1.0e-10);
    myAlgorithm.setMaximumRelativeError(1.0e-10);
    myAlgorithm.setMaximumResidualError(1.0e-10);
    myAlgorithm.setMaximumConstraintError(1.0e-10);

    /* We create a FORM algorithm */
    /* The first parameter is an OptimizationAlgorithm */
    /* The second parameter is an event */
    /* The third parameter is a starting point for the design point research */
    FORM myAlgo(myAlgorithm, myEvent);

    fullprint << "FORM=" << myAlgo << std::endl;

    /* Perform the simulation */
    myAlgo.run();

    /* Stream out the result */
    FORMResult result(myAlgo.getResult());
    const UnsignedInteger digits = 5;
    fullprint << "event probability=" << result.getEventProbability() << std::endl;
    fullprint << "generalized reliability index=" << std::setprecision(digits) << result.getGeneralisedReliabilityIndex() << std::endl;
    fullprint << "standard space design point=" << result.getStandardSpaceDesignPoint() << std::endl;
    fullprint << "physical space design point=" << result.getPhysicalSpaceDesignPoint() << std::endl;
    fullprint << "is standard point origin in failure space? " << (result.getIsStandardPointOriginInFailureSpace() ? "true" : "false") << std::endl;
    fullprint << "importance factors=" << result.getImportanceFactors() << std::endl;
    fullprint << "importance factors (classical)=" << result.getImportanceFactors(AnalyticalResult::CLASSICAL) << std::endl;
    fullprint << "Hasofer reliability index=" << std::setprecision(digits) << result.getHasoferReliabilityIndex() << std::endl;
  }
  catch (TestFailed & ex)
  {
    std::cerr << ex << std::endl;
    return ExitCode::Error;
  }


  return ExitCode::Success;
}
