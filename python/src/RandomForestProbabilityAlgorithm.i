//                                               -*- C++ -*-
/**
 *  @brief Implement Ranger wrapping for random forests probability estimation
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
#ifndef OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_I
#define OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_I

%{
#include "openturns/RandomForestProbabilityAlgorithm.hxx"
%}

%include RandomForestProbabilityResult.i
%include RandomForestProbabilityAlgorithm_doc.i

namespace OpenTURNS {

class RandomForestProbabilityAlgorithm
  : public MetaModelAlgorithm
{
public:
  RandomForestProbabilityAlgorithm();
  RandomForestProbabilityAlgorithm(const Sample & inputSample,
                                   const Sample & outputSample,
                                   const UnsignedInteger importanceMode = 0,
                                   const UnsignedInteger num_trees = 500,
                                   const UnsignedInteger min_node_size = 0,
                                   const UnsignedInteger min_bucket = 0,
                                   const UnsignedInteger max_depth = 0,
                                   const UnsignedInteger mtry = 0);
  void run();
  Sample predict(const Sample & inputSample) const;
  Point getClassValues() const;
  RandomForestProbabilityResult getResult() const;
};

} // namespace OpenTURNS

#endif /* OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_I */