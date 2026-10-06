//                                               -*- C++ -*-
/**
 *  @brief Implement Ranger wrapping for random forests
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

#include "openturns/RandomForestRegressionAlgorithm.hxx"
#include "openturns/MemoizeFunction.hxx"

#ifdef OPENTURNS_HAVE_RANGER
#include <Forest.h>
#include <ForestRegression.h>
#include <Data.h>
#include <DataDouble.h>
#endif

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class RandomForestRegressionAlgorithm
 */

CLASSNAMEINIT(RandomForestRegressionAlgorithm)

/* Default constructor */
RandomForestRegressionAlgorithm::RandomForestRegressionAlgorithm()
  : MetaModelAlgorithm()
  , num_trees_(500)
  , min_node_size_({0})
  , min_bucket_({0})
  , max_depth_(0)
  , mtry_(0)
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , importanceMode_(0)
{
  // Nothing to do
}

RandomForestRegressionAlgorithm::RandomForestRegressionAlgorithm(const Sample & inputSample,
    const Sample & outputSample,
    const UnsignedInteger importanceMode,
    const UnsignedInteger num_trees,
    const UnsignedInteger min_node_size,
    const UnsignedInteger min_bucket,
    const UnsignedInteger max_depth,
    const UnsignedInteger mtry)
  : MetaModelAlgorithm(inputSample, outputSample)
  , num_trees_(num_trees)
  , min_node_size_({static_cast<unsigned int>(min_node_size)})
  , min_bucket_({static_cast<unsigned int>(min_bucket)})
  , max_depth_(max_depth)
  , mtry_(mtry)
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , importanceMode_(importanceMode)
{
  // Nothing to do
}


/* Virtual constructor */
RandomForestRegressionAlgorithm * RandomForestRegressionAlgorithm::clone() const
{
  return new RandomForestRegressionAlgorithm(*this);
}

void RandomForestRegressionAlgorithm::run()
{
#ifdef OPENTURNS_HAVE_RANGER

  auto data = std::make_unique<RandomForestRegressionAlgorithm::DataRanger>(inputSample_, outputSample_, inputSample_.getDescription(), outputSample_.getDescription());
  data->setIsOrderedVariable({}); // no unordered variable

  std::vector<bool> is_ordered_variable = data->getIsOrderedVariable();
  is_ordered_variable_ = PersistentCollection<UnsignedInteger>(is_ordered_variable.size());
  std::copy(is_ordered_variable.begin(), is_ordered_variable.end(), is_ordered_variable_.begin());


  // Create forest
  std::shared_ptr<ranger::ForestRegression> forest(
      new ranger::ForestRegression()
  );

  // Initialize forest
  std::vector<uint> zero_uint_vector = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>> empty_sample = {};
  std::vector<double> empty_point = {};
  std::vector<double> sample_fraction = {1.0}; // valeur par defaut en cas de remplacement
  forest->initR(
      std::move(data),
      /* mtry */ 0,  // 0 = auto-select
      /* num_trees */ num_trees_,
      /* verbose_out */ &std::cout,
      /* seed */ 42,
      /* num_threads */ 4,
      /* importance_mode */ static_cast<ranger::ImportanceMode>(importanceMode_),
      /* min_node_size */ min_node_size_, // {0} = auto-select
      /* min_bucket */ min_bucket_, // {0} = auto-select
      /* split_select_weights */ empty_sample, // {} = desactivation
      /* always_split_variable_names */ {},
      /* prediction_mode */ false,
      /* sample_with_replacement */ true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */ false,
      /* splitrule */ ranger::DEFAULT_SPLITRULE,
      /* case_weights */ empty_point,
      /* manual_inbag */ empty_uint_sample,
      /* predict_all */ false,
      /* keep_inbag */ false,
      /* sample_fraction */ sample_fraction, 
      /* alpha */ 0.5, // sans importance
      /* minprop */ 0.1, // sans importance
      /* poisson_tau */ 0.1, //sans_importance
      /* holdout */ false,
      /* prediction_type */ ranger::RESPONSE,
      /* num_random_splits */ 1, // sans importance
      /* order_snps */ false, // sans importance
      /* max_depth */ max_depth_, // no maximum depth
      /* regularization_factor */ {}, // no regularization
      /* regularization_usedepth */ false, // inutile sans regularisation
      /* node_stats */ false
  );

  // Train
  forest->run(/* verbose */ false, /* compute_oob_error */ true);

  std::vector<std::vector<std::vector<long unsigned int> > > child_node_ids(forest->getChildNodeIDs());
  std::vector<std::vector<long unsigned int> >  split_var_ids(forest->getSplitVarIDs());
  std::vector<std::vector<double> > split_values(forest->getSplitValues());
  num_trees_ = forest->getNumTrees();
  Scalar outOfBagError(forest->getOverallPredictionError());
// We transform the training output as OT compatible objects

split_var_ids_ = PersistentCollection<PersistentCollection<UnsignedInteger> >(split_var_ids.size());
for (UnsignedInteger i = 0; i < split_var_ids.size(); ++i)
{
  split_var_ids_[i] = PersistentCollection<UnsignedInteger>(split_var_ids[i].size()); 
  std::copy(split_var_ids[i].begin(), split_var_ids[i].end(), split_var_ids_[i].begin());
}

child_node_ids_ = PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger> > >(child_node_ids.size());
for (UnsignedInteger j = 0; j < child_node_ids.size(); ++j)
{
  child_node_ids_[j] = PersistentCollection<PersistentCollection<UnsignedInteger> >(child_node_ids[j].size());
  PersistentCollection<PersistentCollection<UnsignedInteger> > buffer(child_node_ids[j].size());
  for (UnsignedInteger i = 0; i < child_node_ids[j].size(); ++i)
  {
    buffer[i] = PersistentCollection<UnsignedInteger>(child_node_ids[j][i].size());
    std::copy(child_node_ids[j][i].begin(), child_node_ids[j][i].end(), buffer[i].begin());
  }
  std::copy(buffer.begin(), buffer.end(), child_node_ids_[j].begin());
}

split_values_ = PersistentCollection<PersistentCollection<Scalar> >(split_values.size());
for (UnsignedInteger i = 0; i < split_values.size(); ++i)
{
  split_values_[i] = PersistentCollection<Scalar>(split_values[i].size());
  std::copy(split_values[i].begin(), split_values[i].end(), split_values_[i].begin());
}

Function metaModel(getRandomForestAsFunction());
Point variableImportance(0);
if (importanceMode_!=0)
{
  std::vector<double> rangerVariableImportance(forest->getVariableImportance());
  for (UnsignedInteger i = 0; i < rangerVariableImportance.size(); ++i)
  {
    variableImportance.add(rangerVariableImportance[i]);
  }
}

result_ = RandomForestRegressionResult(inputSample_, outputSample_, metaModel, outOfBagError, variableImportance);
#else
        throw NotYetImplementedException(HERE) 
            << "Random forest requires Ranger library";
#endif
}

Function RandomForestRegressionAlgorithm::getRandomForestAsFunction()
{
  #ifdef OPENTURNS_HAVE_RANGER
  MemoizeFunction randomforestevaluation(RandomForestEvaluation(*this));
  // Here we change the finite difference gradient for a non centered one in order to reduce the computational cost
  randomforestevaluation.enableCache();
  return randomforestevaluation;

  #else
        throw NotYetImplementedException(HERE) 
            << "Random forest requires Ranger library";
  #endif
}

Sample RandomForestRegressionAlgorithm::predict(const Sample& inputSample) const
{
#ifdef OPENTURNS_HAVE_RANGER

  Sample syntheticOutputSample(inputSample.getSize(), outputSample_.getDimension());
  auto data = std::make_unique<RandomForestRegressionAlgorithm::DataRanger>(inputSample, syntheticOutputSample, inputSample_.getDescription(), outputSample_.getDescription());

  // Create forest
  std::shared_ptr<ranger::ForestRegression> forest(
      new ranger::ForestRegression()
  );

  // Initialize forest
  std::vector<uint> zero_uint_vector = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>> empty_sample = {};
  std::vector<double> empty_point = {};
  std::vector<double> sample_fraction = {1.0}; // valeur par defaut en cas de remplacement

  forest->initR(
      std::move(data),
      /* mtry */ 0,  // 0 = auto-select
      /* num_trees */ 500,
      /* verbose_out */ &std::cout,
      /* seed */ 42,
      /* num_threads */ 1,
      /* importance_mode */ ranger::IMP_NONE,
      /* min_node_size */ zero_uint_vector, // {0} = auto-select
      /* min_bucket */ zero_uint_vector, // {0} = auto-select
      /* split_select_weights */ empty_sample, // {} = desactivation
      /* always_split_variable_names */ {},
      /* prediction_mode */ true,
      /* sample_with_replacement */ true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */ false,
      /* splitrule */ ranger::DEFAULT_SPLITRULE,
      /* case_weights */ empty_point,
      /* manual_inbag */ empty_uint_sample,
      /* predict_all */ false,
      /* keep_inbag */ false,
      /* sample_fraction */ sample_fraction, 
      /* alpha */ 0.5, // sans importance
      /* minprop */ 0.1, // sans importance
      /* poisson_tau */ 0.1, //sans_importance
      /* holdout */ false,
      /* prediction_type */ ranger::RESPONSE,
      /* num_random_splits */ 1, // sans importance
      /* order_snps */ false, // sans importance
      /* max_depth */ 0, // no maximum depth
      /* regularization_factor */ {}, // no regularization
      /* regularization_usedepth */ false, // inutile sans regularisation
      /* node_stats */ false
  );

// We make ranger compatible copies of LoadForest arguments as the method is not declared const
std::vector<std::vector<long unsigned int>> split_var_ids(split_var_ids_.getSize());
for (UnsignedInteger i = 0; i < split_var_ids_.getSize(); ++i)
{
  split_var_ids[i] = std::vector<long unsigned int>(split_var_ids_[i].getSize());
  std::copy(split_var_ids_[i].begin(), split_var_ids_[i].end(), split_var_ids[i].begin());
}

std::vector<std::vector<std::vector<long unsigned int> > > child_node_ids(child_node_ids_.getSize());
for (UnsignedInteger j = 0; j < child_node_ids_.getSize(); ++j)
{
  child_node_ids[j] = std::vector<std::vector<long unsigned int> >(child_node_ids_[j].getSize());
  std::vector<std::vector<long unsigned int> > buffer(child_node_ids_[j].getSize());
  for (UnsignedInteger i = 0; i < child_node_ids_[j].getSize(); ++i)
  {
    buffer[i] = std::vector<long unsigned int>(child_node_ids_[j][i].getSize());
    std::copy(child_node_ids_[j][i].begin(), child_node_ids_[j][i].end(), buffer[i].begin());
  }
  std::copy(buffer.begin(), buffer.end(), child_node_ids[j].begin());
}

std::vector<std::vector<double>> split_values(split_values_.getSize());
for (UnsignedInteger i = 0; i < split_values_.getSize(); ++i)
{
  split_values[i] = std::vector<double>(split_values_[i].getSize());
  std::copy(split_values_[i].begin(), split_values_[i].end(), split_values[i].begin());
}

std::vector<bool> is_ordered_variable(is_ordered_variable_.getSize());
std::copy(is_ordered_variable_.begin(), is_ordered_variable_.end(), is_ordered_variable.begin());


  forest->loadForest(
    num_trees_,
    child_node_ids,
    split_var_ids,
    split_values,
    is_ordered_variable
  );

  //LOGWARN(OSS() << "Dans predict : apres le loadForest ");

  // Predict
  forest->run(/* verbose */ true, /* compute_oob_error */ false);

  //LOGWARN(OSS() << "Dans predict : apres le run ");


  std::vector<std::vector<double>> predictions(forest->getPredictions()[0]);

  Sample result(predictions[0].size(), predictions.size());
  result.setDescription(outputSample_.getDescription());
  for (UnsignedInteger i = 0; i < result.getSize(); ++i)
  {
    for (UnsignedInteger j = 0; j < result.getDimension(); ++j)
    {
      result(i, j) = predictions[j][i];
    }
  }

  return result;
        
#else
        throw NotYetImplementedException(HERE) 
            << "Random forest requires Ranger library";
#endif
}

END_NAMESPACE_OPENTURNS
