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

#include "openturns/RandomForestClassificationAlgorithm.hxx"
#include "openturns/MemoizeFunction.hxx"

#ifdef OPENTURNS_HAVE_RANGER
#include <Data.h>
#include <DataDouble.h>
#include <Forest.h>
#include <ForestClassification.h>
#endif

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class RandomForestClassificationAlgorithm
 */

CLASSNAMEINIT(RandomForestClassificationAlgorithm)

/* Default constructor */
RandomForestClassificationAlgorithm::RandomForestClassificationAlgorithm()
    : MetaModelAlgorithm(),
      num_trees_(500),
      min_node_size_({0}),
      min_bucket_({0}),
      max_depth_(0),
      mtry_(0),
      child_node_ids_(),
      split_values_(),
      is_ordered_variable_(),
      importanceMode_(0),
      classValues_() {
  // Nothing to do
}

RandomForestClassificationAlgorithm::RandomForestClassificationAlgorithm(
    const Sample &inputSample, const Sample &outputSample,
    const UnsignedInteger importanceMode, const UnsignedInteger num_trees,
    const UnsignedInteger min_node_size, const UnsignedInteger min_bucket,
    const UnsignedInteger max_depth, const UnsignedInteger mtry)
    : MetaModelAlgorithm(inputSample, outputSample),
      num_trees_(num_trees),
      min_node_size_({static_cast<unsigned int>(min_node_size)}),
      min_bucket_({static_cast<unsigned int>(min_bucket)}),
      max_depth_(max_depth),
      mtry_(mtry),
      child_node_ids_(),
      split_values_(),
      is_ordered_variable_(),
      importanceMode_(importanceMode),
      classValues_() 
{
  if (outputSample.getDimension() != 1)
    throw InvalidArgumentException(HERE)
        << "RandomForestClassificationAlgorithm: output sample must be "
           "1-dimensional (class labels), got dimension "
        << outputSample.getDimension();
}

/* Virtual constructor */
RandomForestClassificationAlgorithm *RandomForestClassificationAlgorithm::clone()
    const {
  return new RandomForestClassificationAlgorithm(*this);
}

void RandomForestClassificationAlgorithm::run() {
#ifdef OPENTURNS_HAVE_RANGER

  auto data = std::make_unique<RandomForestClassificationAlgorithm::DataRanger>(
      inputSample_, outputSample_, inputSample_.getDescription(), outputSample_.getDescription());
  data->setIsOrderedVariable({});  // no unordered variable

  std::vector<bool> is_ordered_variable = data->getIsOrderedVariable();
  is_ordered_variable_ =
      PersistentCollection<UnsignedInteger>(is_ordered_variable.size());
  std::copy(is_ordered_variable.begin(), is_ordered_variable.end(),
            is_ordered_variable_.begin());

  // Create forest
  std::shared_ptr<ranger::ForestClassification> forest(
      new ranger::ForestClassification());

  // Initialize forest
  std::vector<uint> zero_uint_vector = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>> empty_sample = {};
  std::vector<double> empty_point = {};
  std::vector<double> sample_fraction = {
      1.0};  // valeur par defaut en cas de remplacement
  forest->initR(
      std::move(data),
      /* mtry */ 0,  // 0 = auto-select
      /* num_trees */ num_trees_,
      /* verbose_out */ &std::cout,
      /* seed */ 42,
      /* num_threads */ 4,
      /* importance_mode */ static_cast<ranger::ImportanceMode>(importanceMode_),
      /* min_node_size */ min_node_size_,  // {0} = auto-select
      /* min_bucket */ min_bucket_,      // {0} = auto-select
      /* split_select_weights */ empty_sample,  // {} = desactivation
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
      /* alpha */ 0.5,                  // sans importance
      /* minprop */ 0.1,                // sans importance
      /* poisson_tau */ 0.1,            // sans_importance
      /* holdout */ false,
      /* prediction_type */ ranger::RESPONSE,
      /* num_random_splits */ 1,  // sans importance
      /* order_snps */ false,     // sans importance
      /* max_depth */ max_depth_,  // no maximum depth
      /* regularization_factor */ {},          // no regularization
      /* regularization_usedepth */ false,  // inutile sans regularisation
      /* node_stats */ false);

  // Train
  forest->run(/* verbose */ false, /* compute_oob_error */ true);

  // We transform the training output as OT compatible objects
  std::vector<std::vector<long unsigned int>> split_var_ids(
      forest->getSplitVarIDs());
  split_var_ids_ =
      PersistentCollection<PersistentCollection<UnsignedInteger>>(split_var_ids.size());
  for (UnsignedInteger i = 0; i < split_var_ids.size(); ++i) {
    split_var_ids_[i] =
        PersistentCollection<UnsignedInteger>(split_var_ids[i].size());
    std::copy(split_var_ids[i].begin(), split_var_ids[i].end(),
              split_var_ids_[i].begin());
                std::vector<std::vector<std::vector<long unsigned int>>> child_node_ids(
      forest->getChildNodeIDs());
  child_node_ids_ = PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger>>>(
          child_node_ids.size());
  for (UnsignedInteger j = 0; j < child_node_ids.size(); ++j) {
    PersistentCollection<PersistentCollection<UnsignedInteger>> buffer(
        child_node_ids[j].size());
    for (UnsignedInteger i = 0; i < child_node_ids[j].size(); ++i){
      buffer[i] = PersistentCollection<UnsignedInteger>(child_node_ids[j][i].size());
      std::copy(child_node_ids[j][i].begin(), child_node_ids[j][i].end(),
                buffer[i].begin());
    }
      child_node_ids_[j] = buffer;
  }
  }

  std::vector<std::vector<double>> split_values(forest->getSplitValues());
  split_values_ =
      PersistentCollection<PersistentCollection<Scalar>>(split_values.size());
  for (UnsignedInteger i = 0; i < split_values.size(); ++i) {
    split_values_[i] = PersistentCollection<Scalar>(split_values[i].size());
    std::copy(split_values[i].begin(), split_values[i].end(),
              split_values_[i].begin());
  }

  std::vector<double> class_values = forest->getClassValues();
  classValues_ = Point(class_values.size());
  std::copy(class_values.begin(), class_values.end(), classValues_.begin());
  num_trees_ = forest->getNumTrees();
  Scalar outOfBagError(forest->getOverallPredictionError()); 
  Function metaModel(getRandomForestAsFunction());

  std::vector<double> rangerVariableImportance(forest->getVariableImportance());
  Point variableImportance(rangerVariableImportance.size());
  std::copy(rangerVariableImportance.begin(), rangerVariableImportance.end(), variableImportance.begin());
  

  result_ = RandomForestClassificationResult(
      inputSample_, outputSample_, metaModel, outOfBagError, variableImportance);
#else
  throw NotYetImplementedException(HERE)
      << "Random forest requires Ranger library";
#endif
}

Function RandomForestClassificationAlgorithm::getRandomForestAsFunction() {
#ifdef OPENTURNS_HAVE_RANGER
  MemoizeFunction randomforestevaluation(RandomForestEvaluation(*this));
  // Here we change the finite difference gradient for a non centered one in
  // order to reduce the computational cost
  randomforestevaluation.enableCache();
  return randomforestevaluation;

#else
  throw NotYetImplementedException(HERE)
      << "Random forest requires Ranger library";
#endif
}

Point RandomForestClassificationAlgorithm::getClassValues() const
{
return classValues_;
}



Sample RandomForestClassificationAlgorithm::predict(
    const Sample &inputSample) const {
#ifdef OPENTURNS_HAVE_RANGER

  if (classValues_.isEmpty())
    throw InvalidRangeException(HERE)
        << "RandomForestClassificationAlgorithm::predict: "
           "the forest has not been trained yet (classValues_ is empty)";
  Sample syntheticOutputSample(inputSample.getSize(),
                               outputSample_.getDimension());
  auto data = std::make_unique<RandomForestClassificationAlgorithm::DataRanger>(
      inputSample, syntheticOutputSample, inputSample_.getDescription(),
      outputSample_.getDescription());

  // Create forest
  std::shared_ptr<ranger::ForestClassification> forest(
      new ranger::ForestClassification());

  // Initialize forest
  std::vector<uint> zero_uint_vector = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>> empty_sample = {};
  std::vector<double> empty_point = {};
  std::vector<double> sample_fraction = {
      1.0};  // valeur par defaut en cas de remplacement

  forest->initR(
      std::move(data),
      /* mtry */ 0,  // 0 = auto-select
      /* num_trees */ 500,
      /* verbose_out */ &std::cout,
      /* seed */ 42,
      /* num_threads */ 1,
      /* importance_mode */ ranger::IMP_NONE,
      /* min_node_size */ zero_uint_vector,  // {0} = auto-select
      /* min_bucket */ zero_uint_vector,    // {0} = auto-select
      /* split_select_weights */ empty_sample,  // {} = desactivation
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
      /* alpha */ 0.5,                  // sans importance
      /* minprop */ 0.1,                // sans importance
      /* poisson_tau */ 0.1,            // sans_importance
      /* holdout */ false,
      /* prediction_type */ ranger::RESPONSE,
      /* num_random_splits */ 1,  // sans importance
      /* order_snps */ false,     // sans importance
      /* max_depth */ 0,          // no maximum depth
      /* regularization_factor */ {},          // no regularization
      /* regularization_usedepth */ false,  // inutile sans regularisation
      /* node_stats */ false);

  // We make ranger compatible copies of LoadForest arguments as the method is not
  // declared const
  std::vector<std::vector<long unsigned int>> split_var_ids(
      split_var_ids_.getSize());
  for (UnsignedInteger i = 0; i < split_var_ids_.getSize(); ++i) {
    split_var_ids[i].resize(split_var_ids_[i].getSize());
    std::copy(split_var_ids_[i].begin(), split_var_ids_[i].end(),
              split_var_ids[i].begin());
  }

   std::vector<std::vector<std::vector<long unsigned int>>> child_node_ids(child_node_ids_.getSize());
  for (UnsignedInteger j = 0; j < child_node_ids_.getSize(); ++j)
  {
    child_node_ids[j].resize(child_node_ids_[j].getSize());
    for (UnsignedInteger i = 0; i < child_node_ids_[j].getSize(); ++i)
    {
      child_node_ids[j][i].resize(child_node_ids_[j][i].getSize());
      std::copy(child_node_ids_[j][i].begin(), child_node_ids_[j][i].end(),
                child_node_ids[j][i].begin());
    }
  }

  std::vector<std::vector<double>> split_values(split_values_.getSize());
  for (UnsignedInteger i = 0; i < split_values_.getSize(); ++i)
  {
    split_values[i].resize(split_values_[i].getSize());
    std::copy(split_values_[i].begin(), split_values_[i].end(), split_values[i].begin());
  }

  std::vector<bool> is_ordered_variable(is_ordered_variable_.getSize());
  std::copy(is_ordered_variable_.begin(), is_ordered_variable_.end(),
            is_ordered_variable.begin());

  std::vector<double> class_values(classValues_.getSize());
  std::copy(classValues_.begin(), classValues_.end(), class_values.begin());

  forest->loadForest(num_trees_, child_node_ids, split_var_ids, split_values,
                     class_values, is_ordered_variable);

  // Predict
  forest->run(/* verbose */ true, /* compute_oob_error */ false);

  std::vector<double> predictions(forest->getPredictions()[0][0]);

  Sample result(predictions.size(),1);
  result.setDescription(outputSample_.getDescription());
  for (UnsignedInteger i = 0; i < result.getSize(); ++i) {
      result(i, 0) = predictions[i];
  }

  return result;

#else
  throw NotYetImplementedException(HERE)
      << "Random forest requires Ranger library";
#endif
}



Sample RandomForestClassificationAlgorithm::predictAllTrees(
    const Sample &inputSample) const {
#ifdef OPENTURNS_HAVE_RANGER

  if (classValues_.isEmpty())
    throw InvalidRangeException(HERE)
        << "RandomForestClassificationAlgorithm::predict: "
           "the forest has not been trained yet (classValues_ is empty)";
  Sample syntheticOutputSample(inputSample.getSize(),
                               outputSample_.getDimension());
  auto data = std::make_unique<RandomForestClassificationAlgorithm::DataRanger>(
      inputSample, syntheticOutputSample, inputSample_.getDescription(),
      outputSample_.getDescription());

  // Create forest
  std::shared_ptr<ranger::ForestClassification> forest(
      new ranger::ForestClassification());

  // Initialize forest
  std::vector<uint> zero_uint_vector = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>> empty_sample = {};
  std::vector<double> empty_point = {};
  std::vector<double> sample_fraction = {
      1.0};  // valeur par defaut en cas de remplacement

  forest->initR(
      std::move(data),
      /* mtry */ 0,  // 0 = auto-select
      /* num_trees */ 500,
      /* verbose_out */ &std::cout,
      /* seed */ 42,
      /* num_threads */ 1,
      /* importance_mode */ ranger::IMP_NONE,
      /* min_node_size */ zero_uint_vector,  // {0} = auto-select
      /* min_bucket */ zero_uint_vector,    // {0} = auto-select
      /* split_select_weights */ empty_sample,  // {} = desactivation
      /* always_split_variable_names */ {},
      /* prediction_mode */ true,
      /* sample_with_replacement */ true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */ false,
      /* splitrule */ ranger::DEFAULT_SPLITRULE,
      /* case_weights */ empty_point,
      /* manual_inbag */ empty_uint_sample,
      /* predict_all */ true,    // <-- return per-tree predictions (label per tree per obs)
      /* keep_inbag */ false,
      /* sample_fraction */ sample_fraction,
      /* alpha */ 0.5,                  // sans importance
      /* minprop */ 0.1,                // sans importance
      /* poisson_tau */ 0.1,            // sans_importance
      /* holdout */ false,
      /* prediction_type */ ranger::RESPONSE,
      /* num_random_splits */ 1,  // sans importance
      /* order_snps */ false,     // sans importance
      /* max_depth */ 0,          // no maximum depth
      /* regularization_factor */ {},          // no regularization
      /* regularization_usedepth */ false,  // inutile sans regularisation
      /* node_stats */ false);

  // We make ranger compatible copies of LoadForest arguments as the method is not
  // declared const
  std::vector<std::vector<long unsigned int>> split_var_ids(
      split_var_ids_.getSize());
  for (UnsignedInteger i = 0; i < split_var_ids_.getSize(); ++i) {
    split_var_ids[i].resize(split_var_ids_[i].getSize());
    std::copy(split_var_ids_[i].begin(), split_var_ids_[i].end(),
              split_var_ids[i].begin());
  }

   std::vector<std::vector<std::vector<long unsigned int>>> child_node_ids(child_node_ids_.getSize());
  for (UnsignedInteger j = 0; j < child_node_ids_.getSize(); ++j)
  {
    child_node_ids[j].resize(child_node_ids_[j].getSize());
    for (UnsignedInteger i = 0; i < child_node_ids_[j].getSize(); ++i)
    {
      child_node_ids[j][i].resize(child_node_ids_[j][i].getSize());
      std::copy(child_node_ids_[j][i].begin(), child_node_ids_[j][i].end(),
                child_node_ids[j][i].begin());
    }
  }

  std::vector<std::vector<double>> split_values(split_values_.getSize());
  for (UnsignedInteger i = 0; i < split_values_.getSize(); ++i)
  {
    split_values[i].resize(split_values_[i].getSize());
    std::copy(split_values_[i].begin(), split_values_[i].end(), split_values[i].begin());
  }

  std::vector<bool> is_ordered_variable(is_ordered_variable_.getSize());
  std::copy(is_ordered_variable_.begin(), is_ordered_variable_.end(),
            is_ordered_variable.begin());

  std::vector<double> class_values(classValues_.getSize());
  std::copy(classValues_.begin(), classValues_.end(), class_values.begin());

  forest->loadForest(num_trees_, child_node_ids, split_var_ids, split_values,
                     class_values, is_ordered_variable);

  // Predict
  forest->run(/* verbose */ false, /* compute_oob_error */ false);

  // With predict_all=true, getPredictions() has the same outer structure as
  // predict_all=false: [output_var][middle][inner], i.e. [0][.][.].
  // The difference is in what middle and inner index:
  //   predict_all=false : [0][0    ][obs  ]  — 1 aggregated row, obs in inner
  //   predict_all=true  : [0][obs  ][tree ]  — obs in middle, tree in inner
  // This matches the R documentation: predict.all returns a matrix (sample x tree).
 
  const std::vector<std::vector<double>> & allPreds = forest->getPredictions()[0];

  const UnsignedInteger n = inputSample.getSize();

  // Build description: "tree_t" for each tree column
  Description desc(num_trees_);
  for (UnsignedInteger t = 0; t < num_trees_; ++t)
  {
    OSS oss;
    oss << "tree_" << t;
    desc[t] = oss;
  }

  // Result shape: (n × num_trees), each cell is a numeric class label
  Sample result(n, num_trees_);
  result.setDescription(desc);

  for (UnsignedInteger t = 0; t < num_trees_; ++t)
    for (UnsignedInteger i = 0; i < n; ++i)
      result(i, t) = allPreds[i][t];

  return result;

#else
  throw NotYetImplementedException(HERE)
      << "Random forest requires Ranger library";
#endif
}

END_NAMESPACE_OPENTURNS

