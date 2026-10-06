//                                               -*- C++ -*-
/**
 *  @brief Implement Ranger wrapping for classification random forests
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

#include "openturns/ClassificationRandomForestAlgorithm.hxx"
#include "openturns/MemoizeFunction.hxx"

#ifdef OPENTURNS_HAVE_RANGER
#include <Forest.h>
#include <ForestClassification.h>
#include <Data.h>
#include <DataDouble.h>
#endif

BEGIN_NAMESPACE_OPENTURNS

CLASSNAMEINIT(ClassificationRandomForestAlgorithm)

// ============================================================
// Constructors
// ============================================================

ClassificationRandomForestAlgorithm::ClassificationRandomForestAlgorithm()
  : MetaModelAlgorithm()
  , result_()
  , num_trees_(0)
  , split_var_ids_()
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , classValues_()
{
}

ClassificationRandomForestAlgorithm::ClassificationRandomForestAlgorithm(
    const Sample & inputSample,
    const Sample & outputSample)
  : MetaModelAlgorithm(inputSample, outputSample)
  , result_()
  , num_trees_(0)
  , split_var_ids_()
  , child_node_ids_()
  , split_values_()
  , is_ordered_variable_()
  , classValues_()
{
  if (outputSample.getDimension() != 1)
    throw InvalidArgumentException(HERE)
        << "ClassificationRandomForestAlgorithm: output sample must be "
           "1-dimensional (class labels), got dimension "
        << outputSample.getDimension();
}

ClassificationRandomForestAlgorithm *
ClassificationRandomForestAlgorithm::clone() const
{
  return new ClassificationRandomForestAlgorithm(*this);
}

// ============================================================
// run() — train the classification forest
// ============================================================

void ClassificationRandomForestAlgorithm::run()
{
#ifdef OPENTURNS_HAVE_RANGER

  // ---- Build the Ranger data bridge ------------------------------------
  auto data = std::make_unique<ClassificationRandomForestAlgorithm::DataRanger>(
      inputSample_,
      outputSample_,
      inputSample_.getDescription(),
      outputSample_.getDescription());

  // No unordered categorical variables among predictors
  data->setIsOrderedVariable({});

  std::vector<bool> is_ordered_variable_vec = data->getIsOrderedVariable();
  is_ordered_variable_ = PersistentCollection<UnsignedInteger>(is_ordered_variable_vec.size());
  std::copy(is_ordered_variable_vec.begin(), is_ordered_variable_vec.end(),
            is_ordered_variable_.begin());

  // ---- Create and initialise ForestClassification ----------------------
  std::shared_ptr<ranger::ForestClassification> forest =
      std::make_shared<ranger::ForestClassification>();

  std::vector<uint>                             zero_uint_vector  = {0};
  std::vector<std::vector<long unsigned int>>   empty_uint_sample = {};
  std::vector<std::vector<double>>              empty_sample      = {};
  std::vector<double>                           empty_point       = {};
  std::vector<double>                           sample_fraction   = {1.0};

  forest->initR(
      std::move(data),
      /* mtry */                     0,            // 0 = auto-select
      /* num_trees */                500,
      /* verbose_out */              &std::cout,
      /* seed */                     42,
      /* num_threads */              4,
      /* importance_mode */          ranger::IMP_NONE,
      /* min_node_size */            zero_uint_vector,
      /* min_bucket */               zero_uint_vector,
      /* split_select_weights */     empty_sample,
      /* always_split_variable_names */ {},
      /* prediction_mode */          false,
      /* sample_with_replacement */  true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */  false,
      /* splitrule */                ranger::DEFAULT_SPLITRULE,
      /* case_weights */             empty_point,
      /* manual_inbag */             empty_uint_sample,
      /* predict_all */              false,        // majority-vote prediction
      /* keep_inbag */               false,
      /* sample_fraction */          sample_fraction,
      /* alpha */                    0.5,
      /* minprop */                  0.1,
      /* poisson_tau */              0.1,
      /* holdout */                  false,
      /* prediction_type */          ranger::RESPONSE,
      /* num_random_splits */        1,
      /* order_snps */               false,
      /* max_depth */                0,
      /* regularization_factor */    {},
      /* regularization_usedepth */ false,
      /* node_stats */               false
  );

  // ---- Train -----------------------------------------------------------
  forest->run(/* verbose */ false, /* compute_oob_error */ true);

  // ---- Retrieve and convert forest structure ---------------------------
  num_trees_ = forest->getNumTrees();

  {
    std::vector<std::vector<long unsigned int>> split_var_ids = forest->getSplitVarIDs();
    split_var_ids_ = PersistentCollection<PersistentCollection<UnsignedInteger>>(split_var_ids.size());
    for (UnsignedInteger i = 0; i < split_var_ids.size(); ++i)
    {
      split_var_ids_[i] = PersistentCollection<UnsignedInteger>(split_var_ids[i].size());
      std::copy(split_var_ids[i].begin(), split_var_ids[i].end(), split_var_ids_[i].begin());
    }
  }

  {
    std::vector<std::vector<std::vector<long unsigned int>>> child_node_ids =
        forest->getChildNodeIDs();
    child_node_ids_ = PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger>>>(
        child_node_ids.size());
    for (UnsignedInteger j = 0; j < child_node_ids.size(); ++j)
    {
      PersistentCollection<PersistentCollection<UnsignedInteger>> buf(child_node_ids[j].size());
      for (UnsignedInteger i = 0; i < child_node_ids[j].size(); ++i)
      {
        buf[i] = PersistentCollection<UnsignedInteger>(child_node_ids[j][i].size());
        std::copy(child_node_ids[j][i].begin(), child_node_ids[j][i].end(), buf[i].begin());
      }
      child_node_ids_[j] = buf;
    }
  }

  {
    std::vector<std::vector<double>> split_values = forest->getSplitValues();
    split_values_ = PersistentCollection<PersistentCollection<Scalar>>(split_values.size());
    for (UnsignedInteger i = 0; i < split_values.size(); ++i)
    {
      split_values_[i] = PersistentCollection<Scalar>(split_values[i].size());
      std::copy(split_values[i].begin(), split_values[i].end(), split_values_[i].begin());
    }
  }

  // ---- Store the class-value map ---------------------------------------
  // ForestClassification::getClassValues() returns std::vector<double>
  {
    std::vector<double> cv = forest->getClassValues();
    classValues_ = Point(cv.size());
    std::copy(cv.begin(), cv.end(), classValues_.begin());
  }

  // ---- OOB error -------------------------------------------------------
  const Scalar oobError = forest->getOverallPredictionError();
  LOGWARN(OSS() << "inputSample_=" << inputSample_);
  LOGWARN(OSS() << "outputSample_=" << outputSample_);
  LOGWARN(OSS() << "classValues_=" << classValues_);
  LOGWARN(OSS() << "oobError=" << oobError);



  // ---- Build the result ------------------------------------------------
  Function metaModel(getRandomForestAsFunction());
  result_ = ClassificationRandomForestResult(
      inputSample_, outputSample_, metaModel, classValues_, oobError);

#else
  throw NotYetImplementedException(HERE)
      << "ClassificationRandomForestAlgorithm requires the Ranger library";
#endif
}

// ============================================================
// getRandomForestAsFunction()
// ============================================================

Function ClassificationRandomForestAlgorithm::getRandomForestAsFunction()
{
#ifdef OPENTURNS_HAVE_RANGER
  MemoizeFunction eval(ClassificationForestEvaluation(*this));
  eval.enableCache();
  return eval;
#else
  throw NotYetImplementedException(HERE)
      << "ClassificationRandomForestAlgorithm requires the Ranger library";
#endif
}

// ============================================================
// predict() — majority-vote class label
// ============================================================

Sample ClassificationRandomForestAlgorithm::predict(const Sample & inputSample) const
{
#ifdef OPENTURNS_HAVE_RANGER

  if (classValues_.isEmpty())
    throw InvalidRangeException(HERE)
        << "ClassificationRandomForestAlgorithm::predict: "
           "the forest has not been trained yet (classValues_ is empty)";

  // Build a synthetic output sample (content ignored during prediction)
  Sample syntheticOutput(inputSample.getSize(), 1);
  syntheticOutput.setDescription(outputSample_.getDescription());

  auto data = std::make_unique<ClassificationRandomForestAlgorithm::DataRanger>(
      inputSample,
      syntheticOutput,
      inputSample_.getDescription(),
      outputSample_.getDescription());

  std::shared_ptr<ranger::ForestClassification> forest =
      std::make_shared<ranger::ForestClassification>();

  std::vector<uint>                           zero_uint_vector  = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>>            empty_sample      = {};
  std::vector<double>                         empty_point       = {};
  std::vector<double>                         sample_fraction   = {1.0};

  forest->initR(
      std::move(data),
      /* mtry */                     0,
      /* num_trees */                static_cast<uint>(num_trees_),
      /* verbose_out */              &std::cout,
      /* seed */                     42,
      /* num_threads */              1,
      /* importance_mode */          ranger::IMP_NONE,
      /* min_node_size */            zero_uint_vector,
      /* min_bucket */               zero_uint_vector,
      /* split_select_weights */     empty_sample,
      /* always_split_variable_names */ {},
      /* prediction_mode */          true,
      /* sample_with_replacement */  true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */  false,
      /* splitrule */                ranger::DEFAULT_SPLITRULE,
      /* case_weights */             empty_point,
      /* manual_inbag */             empty_uint_sample,
      /* predict_all */              false,   // majority-vote only
      /* keep_inbag */               false,
      /* sample_fraction */          sample_fraction,
      /* alpha */                    0.5,
      /* minprop */                  0.1,
      /* poisson_tau */              0.1,
      /* holdout */                  false,
      /* prediction_type */          ranger::RESPONSE,
      /* num_random_splits */        1,
      /* order_snps */               false,
      /* max_depth */                0,
      /* regularization_factor */    {},
      /* regularization_usedepth */ false,
      /* node_stats */               false
  );

  // ---- Rebuild Ranger-compatible copies of the stored forest -----------

  std::vector<std::vector<long unsigned int>> split_var_ids(split_var_ids_.getSize());
  for (UnsignedInteger i = 0; i < split_var_ids_.getSize(); ++i)
  {
    split_var_ids[i].resize(split_var_ids_[i].getSize());
    std::copy(split_var_ids_[i].begin(), split_var_ids_[i].end(), split_var_ids[i].begin());
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

  // Convert classValues_ (Point) back to std::vector<double> for Ranger
  std::vector<double> class_values(classValues_.getSize());
  std::copy(classValues_.begin(), classValues_.end(), class_values.begin());

  // ---- Load the serialised forest and run prediction -------------------
  forest->loadForest(
      num_trees_,
      child_node_ids,
      split_var_ids,
      split_values,
      class_values,          // <-- classification-specific argument
      is_ordered_variable
  );

  forest->run(/* verbose */ false, /* compute_oob_error */ false);

  // ---- Convert Ranger predictions to OT Sample -------------------------
  // getPredictions() for ForestClassification with predict_all=false returns
  // a 3-D structure whose layout is [1][1][n_obs]:
  //   - outer dimension : 1 block (one output variable)
  //   - middle dimension: 1 row   (aggregated over all trees)
  //   - inner dimension : n_obs values, each being the predicted class label
  //                       (the actual label value, not an index into class_values)
  //
  // Note: unlike regression, ForestClassification stores the winning class
  // label directly (not an index), so no remapping via classValues_ is needed.
  const std::vector<std::vector<std::vector<double>>> & allPreds = forest->getPredictions();
  // allPreds[0][0][obs] == predicted class label for observation obs

  const UnsignedInteger n = inputSample.getSize();
  Sample result(n, 1);
  result.setDescription(outputSample_.getDescription());
  for (UnsignedInteger i = 0; i < n; ++i)
  {
    result(i, 0) = allPreds[0][0][i];
  }

  return result;

#else
  throw NotYetImplementedException(HERE)
      << "ClassificationRandomForestAlgorithm requires the Ranger library";
#endif
}

// ============================================================
// predictAllTrees() — per-tree class label predictions
// ============================================================
//
// With predict_all=true, ForestClassification returns the label predicted
// by each individual tree for each observation.
// Layout: getPredictions()[tree][obs][0] = class label (as double) predicted
// by tree `tree` for observation `obs`.
// This is NOT a probability: to get per-class probabilities, a
// ForestProbability (probability=true in the R API) must be trained instead.
//
// The returned Sample has shape (n × num_trees_), where column t is the
// prediction of tree t, expressed as a numeric class label.

Sample ClassificationRandomForestAlgorithm::predictAllTrees(
    const Sample & inputSample) const
{
#ifdef OPENTURNS_HAVE_RANGER

  if (classValues_.isEmpty())
    throw InvalidRangeException(HERE)
        << "ClassificationRandomForestAlgorithm::predictAllTrees: "
           "the forest has not been trained yet (classValues_ is empty)";

  Sample syntheticOutput(inputSample.getSize(), 1);
  syntheticOutput.setDescription(outputSample_.getDescription());

  auto data = std::make_unique<ClassificationRandomForestAlgorithm::DataRanger>(
      inputSample,
      syntheticOutput,
      inputSample_.getDescription(),
      outputSample_.getDescription());

  std::shared_ptr<ranger::ForestClassification> forest =
      std::make_shared<ranger::ForestClassification>();

  std::vector<uint>                           zero_uint_vector  = {0};
  std::vector<std::vector<long unsigned int>> empty_uint_sample = {};
  std::vector<std::vector<double>>            empty_sample      = {};
  std::vector<double>                         empty_point       = {};
  std::vector<double>                         sample_fraction   = {1.0};

  forest->initR(
      std::move(data),
      /* mtry */                     0,
      /* num_trees */                static_cast<uint>(num_trees_),
      /* verbose_out */              &std::cout,
      /* seed */                     42,
      /* num_threads */              1,
      /* importance_mode */          ranger::IMP_NONE,
      /* min_node_size */            zero_uint_vector,
      /* min_bucket */               zero_uint_vector,
      /* split_select_weights */     empty_sample,
      /* always_split_variable_names */ {},
      /* prediction_mode */          true,
      /* sample_with_replacement */  true,
      /* unordered_variable_names */ {},
      /* memory_saving_splitting */  false,
      /* splitrule */                ranger::DEFAULT_SPLITRULE,
      /* case_weights */             empty_point,
      /* manual_inbag */             empty_uint_sample,
      /* predict_all */              true,    // <-- return per-tree predictions (label per tree per obs)
      /* keep_inbag */               false,
      /* sample_fraction */          sample_fraction,
      /* alpha */                    0.5,
      /* minprop */                  0.1,
      /* poisson_tau */              0.1,
      /* holdout */                  false,
      /* prediction_type */          ranger::RESPONSE,
      /* num_random_splits */        1,
      /* order_snps */               false,
      /* max_depth */                0,
      /* regularization_factor */    {},
      /* regularization_usedepth */ false,
      /* node_stats */               false
  );

  std::vector<std::vector<long unsigned int>> split_var_ids(split_var_ids_.getSize());
  for (UnsignedInteger i = 0; i < split_var_ids_.getSize(); ++i)
  {
    split_var_ids[i].resize(split_var_ids_[i].getSize());
    std::copy(split_var_ids_[i].begin(), split_var_ids_[i].end(), split_var_ids[i].begin());
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

  forest->loadForest(
      num_trees_,
      child_node_ids,
      split_var_ids,
      split_values,
      class_values,          // <-- classification-specific argument
      is_ordered_variable
  );

  forest->run(/* verbose */ false, /* compute_oob_error */ false);

  // With predict_all=true, ForestClassification stores one prediction block
  // per tree. Layout: getPredictions()[tree][obs][0] = class label (double)
  // predicted by that tree for that observation.
  const std::vector<std::vector<double>> & allPreds = forest->getPredictions()[0];

  const UnsignedInteger n         = inputSample.getSize();

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
      << "ClassificationRandomForestAlgorithm requires the Ranger library";
#endif
}

END_NAMESPACE_OPENTURNS