//                                               -*- C++ -*-
/**
 *  @brief The test file of class Pairs for standard methods
 *
 *  Copyright 2005-2019 Airbus-EDF-IMACS-Phimeca
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
  setRandomGenerator();

  try
  {

    // Instanciate one distribution object
    UnsignedInteger dim = 5;
    Point meanPoint(dim, 0.0);
    Point sigma(dim, 5.0);
    CorrelationMatrix R(dim);

    Normal distribution(meanPoint, sigma, R);

    // Instanciate another distribution object
    for (UnsignedInteger i = 1; i < dim; i++) R(i, i - 1) = -0.25;

    // Test for sampling
    UnsignedInteger size = 1000;
    Sample sample(distribution.getSample( size ));

    // Create an empty graph
    Graph myGraph("Pairs", " ", " ", true, "topright");

    // Create the first cloud
    Pairs myPairs(sample, "Pairs example", sample.getDescription(), "green", "bullet");

    // Then, draw it
    myGraph.add(Drawable(myPairs));
    myGraph.draw("Graph_Pairs_OT");

    // Check that the correct files have been generated by computing their checksum
  }
  catch (TestFailed & ex)
  {
    std::cerr << ex << std::endl;
    return ExitCode::Error;
  }


  return ExitCode::Success;
}
