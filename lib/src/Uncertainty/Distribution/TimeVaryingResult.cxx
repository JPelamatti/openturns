//                                               -*- C++ -*-
/**
 *  @brief Result of GEV time-varying likelihood
 *
 *  Copyright 2005-2023 Airbus-EDF-IMACS-ONERA-Phimeca
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
#include "openturns/TimeVaryingResult.hxx"
#include "openturns/PersistentObjectFactory.hxx"
#include "openturns/ChiSquare.hxx"
#include "openturns/Brent.hxx"
#include "openturns/Curve.hxx"
#include "openturns/Text.hxx"

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(TimeVaryingResult)

static const Factory<TimeVaryingResult> Factory_TimeVaryingResult;


TimeVaryingResult::TimeVaryingResult()
 : PersistentObject()
 {}

TimeVaryingResult::TimeVaryingResult(const DistributionFactory & factory,
                                     const Function & parameterFunction,
                                     const Mesh & mesh,
                                     const Distribution & parameterDistribution,
                                     const Scalar logLikelihood)
 : PersistentObject()
 , factory_(factory)
 , parameterFunction_(parameterFunction)
 , mesh_(mesh)
 , parameterDistribution_(parameterDistribution)
 , logLikelihood_(logLikelihood)
{
  if (mesh.getDimension() != parameterFunction.getInputDimension())
    throw InvalidArgumentException(HERE) << "the mesh dimension must match the parameter function input dimension";
  if (parameterDistribution.getDimension() != parameterFunction.getParameter().getDimension())
    throw InvalidArgumentException(HERE) << "the parameter distribution dimension must match the parameter function parameter dimension";
}

TimeVaryingResult * TimeVaryingResult::clone() const
{
  return new TimeVaryingResult(*this);
}

Point TimeVaryingResult::getOptimalParameter() const
{
  return parameterDistribution_.getMean();
}

void TimeVaryingResult::setParameterDistribution(const Distribution & parameterDistribution)
{
  parameterDistribution_ = parameterDistribution;
}

Distribution TimeVaryingResult::getParameterDistribution() const
{
  return parameterDistribution_;
}

void TimeVaryingResult::setLogLikelihood(const Scalar logLikelihood)
{
  logLikelihood_ = logLikelihood;
}

Scalar TimeVaryingResult::getLogLikelihood() const
{
  return logLikelihood_;
}

/* Draw parameter for all time values */
Graph TimeVaryingResult::drawParameterFunction(const UnsignedInteger parameterIndex) const
{
  const Sample grid(mesh_.getVertices());
  const Scalar xMin = grid.getMin()[0];
  const Scalar xMax = grid.getMax()[0];
  Graph result(parameterFunction_.getMarginal(parameterIndex).draw(xMin, xMax));
  result.setTitle("Parameter function");
  return result;
}

class TimeVaryingResultQuantileEvaluation : public EvaluationImplementation
{
public:
  TimeVaryingResultQuantileEvaluation(const TimeVaryingResult & result, const Scalar p)
  : result_(result)
  , p_(p)
  {
    setInputDescription({"t"});
    setOutputDescription({"quantile(t)"});
  }

  TimeVaryingResultQuantileEvaluation * clone() const override
  {
    return new TimeVaryingResultQuantileEvaluation(*this);
  }

  Point operator()(const Point & inP) const override
  {
    const Scalar t = inP[0];
    return result_.getDistribution(t).computeQuantile(p_);
  }

  UnsignedInteger getInputDimension() const override
  {
    return 1;
  }

  UnsignedInteger getOutputDimension() const override
  {
    return 1;
  }

private:
  TimeVaryingResult result_;
  Scalar p_ = 0.0;
};

/* Draw quantile for all time values */
Graph TimeVaryingResult::drawQuantileFunction(const Scalar p) const
{
  const Sample grid(mesh_.getVertices());
  const Scalar xMin = grid.getMin()[0];
  const Scalar xMax = grid.getMax()[0];

  Function quantileFunction(TimeVaryingResultQuantileEvaluation(*this, p));
  Graph result(quantileFunction.draw(xMin, xMax));
  result.setTitle("Quantile function");
  return result;
}

String TimeVaryingResult::__repr__() const
{
  return OSS() << PersistentObject::__repr__();
}

Function TimeVaryingResult::getParameterFunction() const
{
  return parameterFunction_;
}

/* Accessor to the distribution at a given time */
Distribution TimeVaryingResult::getDistribution(const Scalar t) const
{
  const Point parameters(parameterFunction_(Point(1, t)));
  return factory_.build(parameters);
}

/* Method save() stores the object through the StorageManager */
void TimeVaryingResult::save(Advocate & adv) const
{
  PersistentObject::save(adv);
  adv.saveAttribute("parameterFunction_", parameterFunction_);
  adv.saveAttribute("mesh_", mesh_);
  adv.saveAttribute("parameterDistribution_", parameterDistribution_);
  adv.saveAttribute("logLikelihood_", logLikelihood_);
}

/* Method load() reloads the object from the StorageManager */
void TimeVaryingResult::load(Advocate & adv)
{
  PersistentObject::load(adv);
  adv.loadAttribute("parameterFunction_", parameterFunction_);
  adv.loadAttribute("mesh_", mesh_);
  adv.loadAttribute("parameterDistribution_", parameterDistribution_);
  adv.loadAttribute("logLikelihood_", logLikelihood_);
}

END_NAMESPACE_OPENTURNS
