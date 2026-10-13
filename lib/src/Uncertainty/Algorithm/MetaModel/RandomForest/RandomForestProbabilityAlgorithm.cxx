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
#include "openturns/RandomForestProbabilityAlgorithm.hxx"
#include "openturns/RandomForestProbabilityResult.hxx"
#include "openturns/Function.hxx"
#include "openturns/Exception.hxx"
#include "openturns/PersistentObjectFactory.hxx"
#include <set>

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(RandomForestProbabilityAlgorithm)

static const Factory<RandomForestProbabilityAlgorithm> Factory_RandomForestProbabilityAlgorithm;

RandomForestProbabilityAlgorithm::RandomForestProbabilityAlgorithm()
  : MetaModelAlgorithm()
#ifdef OPENTURNS_HAVE_RANGER
  , forest_()
  , result_()
  , min_node_size_(1, 0)
  , min_bucket_(1, 0)
  , max_depth_(0)
  , mtry_(0)
  , importanceMode_(0)
  , num_trees_(500)
  , split_var_ids_()
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , classValues_()
#endif
{
}

RandomForestProbabilityAlgorithm::RandomForestProbabilityAlgorithm(const Sample &inputSample,
                                                                   const Sample &outputSample,
                                                                   const UnsignedInteger importanceMode,
                                                                   const UnsignedInteger num_trees,
                                                                   const UnsignedInteger min_node_size,
                                                                   const UnsignedInteger min_bucket,
                                                                   const UnsignedInteger max_depth,
                                                                   const UnsignedInteger mtry)
  : MetaModelAlgorithm(inputSample, outputSample)
#ifdef OPENTURNS_HAVE_RANGER
  , forest_()
  , result_()
  , min_node_size_(1, min_node_size)
  , min_bucket_(1, min_bucket)
  , max_depth_(max_depth)
  , mtry_(mtry)
  , importanceMode_(importanceMode)
  , num_trees_(num_trees)
  , split_var_ids_()
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , classValues_()
#endif
{
  if (outputSample.getDimension() != 1) {
    throw InvalidArgumentException(HERE) << "RandomForestProbabilityAlgorithm requires a 1D categorical output sample.";
  }
}

RandomForestProbabilityAlgorithm *RandomForestProbabilityAlgorithm::clone() const
{
  return new RandomForestProbabilityAlgorithm(*this);
}

void RandomForestProbabilityAlgorithm::run()
{
#ifdef OPENTURNS_HAVE_RANGER
  const Sample &input = getInputSample();
  const Sample &output = getOutputSample();
  const UnsignedInteger n = input.getSize();

  if (n == 0) {
    throw InvalidArgumentException(HERE) << "The input sample is empty.";
  }
  if (output.getSize() != n) {
    throw InvalidArgumentException(HERE) << "Input and output samples have inconsistent sizes.";
  }

  // Extract unique sorted class values
  std::set<Scalar> uniqueClasses;
  for (UnsignedInteger i = 0; i < n; ++i) {
    uniqueClasses.insert(output(i, 0));
  }
  classValues_ = Point(uniqueClasses.size());
  UnsignedInteger cIdx = 0;
  for (const Scalar val : uniqueClasses) {
    classValues_[cIdx++] = val;
  }

  forest_ = std::make_shared<ranger::ForestProbability>();
  std::unique_ptr<DataRanger> data = std::make_unique<DataRanger>(
      input, output, input.getDescription(), output.getDescription());

  ranger::uint seed = 0;
  ranger::ImportanceMode imp_mode = static_cast<ranger::ImportanceMode>(importanceMode_);
  std::vector<std::vector<double>> split_select_weights;
  std::vector<std::string> always_split_var_names;
  std::vector<std::string> unordered_var_names;
  std::vector<double> case_weights;
  std::vector<std::vector<size_t>> manual_inbag;
  std::vector<double> sample_fraction = {1.0};
  std::vector<double> regularization_factor;

  forest_->initR(
      std::move(data),                                // input_data
      mtry_,                                          // mtry
      num_trees_,                                     // num_trees
      nullptr,                                        // verbose_out
      seed,                                           // seed
      1,                                              // num_threads
      imp_mode,                                       // importance_mode
      min_node_size_,                                 // min_node_size
      min_bucket_,                                    // min_bucket
      split_select_weights,                           // split_select_weights
      always_split_var_names,                         // always_split_variable_names
      false,                                          // prediction_mode
      true,                                           // sample_with_replacement
      unordered_var_names,                            // unordered_variable_names
      false,                                          // memory_saving_splitting
      ranger::DEFAULT_SPLITRULE,                      // splitrule
      case_weights,                                   // case_weights
      manual_inbag,                                   // manual_inbag
      false,                                          // predict_all
      false,                                          // keep_inbag
      sample_fraction,                                // sample_fraction
      ranger::DEFAULT_ALPHA,                          // alpha
      ranger::DEFAULT_MINPROP,                        // minprop
      ranger::DEFAULT_POISSON_TAU,                    // poisson_tau
      false,                                          // holdout
      ranger::DEFAULT_PREDICTIONTYPE,                 // prediction_type
      ranger::DEFAULT_NUM_RANDOM_SPLITS,              // num_random_splits
      false,                                          // order_snps
      max_depth_,                                     // max_depth
      regularization_factor,                          // regularization_factor
      false,                                          // regularization_usedepth
      false                                           // node_stats
  );

  forest_->run(true, false);

  // Collect variable importance and out-of-bag prediction error
  const std::vector<double> &imp = forest_->getVariableImportance();
  Point variableImportance(imp.size());
  for (size_t i = 0; i < imp.size(); ++i) {
    variableImportance[i] = imp[i];
  }
  const Scalar oobError = forest_->getOverallPredictionError();

  Function metaModel(new RandomForestEvaluation(*this));
  result_ = RandomForestProbabilityResult(metaModel, classValues_, variableImportance, oobError);
#else
  throw NotYetImplementedException(HERE) << "OpenTURNS was built without Ranger support.";
#endif
}

Sample RandomForestProbabilityAlgorithm::predict(const Sample &inputSample) const
{
#ifdef OPENTURNS_HAVE_RANGER
  if (!forest_) {
    throw InvalidArgumentException(HERE) << "Algorithm must be run before predicting.";
  }

  const UnsignedInteger n = inputSample.getSize();
  const UnsignedInteger numClasses = classValues_.getDimension();
  Sample dummyOutput(n, 1);

  std::unique_ptr<DataRanger> testData = std::make_unique<DataRanger>(
      inputSample, dummyOutput, inputSample.getDescription(), dummyOutput.getDescription());

  ranger::uint seed = 0;
  ranger::ImportanceMode imp_mode = ranger::ImportanceMode::IMP_NONE;
  std::vector<std::vector<double>> split_select_weights;
  std::vector<std::string> always_split_var_names;
  std::vector<std::string> unordered_var_names;
  std::vector<double> case_weights;
  std::vector<std::vector<size_t>> manual_inbag;
  std::vector<double> sample_fraction = {1.0};
  std::vector<double> regularization_factor;

  forest_->initR(
      std::move(testData),                            // input_data
      mtry_,                                          // mtry
      num_trees_,                                     // num_trees
      nullptr,                                        // verbose_out
      seed,                                           // seed
      1,                                              // num_threads
      imp_mode,                                       // importance_mode
      min_node_size_,                                 // min_node_size
      min_bucket_,                                    // min_bucket
      split_select_weights,                           // split_select_weights
      always_split_var_names,                         // always_split_variable_names
      true,                                           // prediction_mode
      true,                                           // sample_with_replacement
      unordered_var_names,                            // unordered_variable_names
      false,                                          // memory_saving_splitting
      ranger::DEFAULT_SPLITRULE,                      // splitrule
      case_weights,                                   // case_weights
      manual_inbag,                                   // manual_inbag
      false,                                          // predict_all
      false,                                          // keep_inbag
      sample_fraction,                                // sample_fraction
      ranger::DEFAULT_ALPHA,                          // alpha
      ranger::DEFAULT_MINPROP,                        // minprop
      ranger::DEFAULT_POISSON_TAU,                    // poisson_tau
      false,                                          // holdout
      ranger::DEFAULT_PREDICTIONTYPE,                 // prediction_type
      ranger::DEFAULT_NUM_RANDOM_SPLITS,              // num_random_splits
      false,                                          // order_snps
      max_depth_,                                     // max_depth
      regularization_factor,                          // regularization_factor
      false,                                          // regularization_usedepth
      false                                           // node_stats
  );
  forest_->run(false, false);

  const auto &predictions = forest_->getPredictions();
  Sample predictedProbabilities(n, numClasses);

  for (UnsignedInteger i = 0; i < n; ++i) {
    for (UnsignedInteger c = 0; c < numClasses; ++c) {
      predictedProbabilities(i, c) = predictions[0][i][c];
    }
  }
  return predictedProbabilities;
#else
  throw NotYetImplementedException(HERE) << "OpenTURNS was built without Ranger support.";
#endif
}

Point RandomForestProbabilityAlgorithm::getClassValues() const
{
  return classValues_;
}

RandomForestProbabilityResult RandomForestProbabilityAlgorithm::getResult() const
{
  return result_;
}

END_NAMESPACE_OPENTURNS