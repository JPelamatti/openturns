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
#ifndef OPENTURNS_CLASSIFICATIONRANDOMFORESTALGORITHM_HXX
#define OPENTURNS_CLASSIFICATIONRANDOMFORESTALGORITHM_HXX

#include "openturns/MetaModelAlgorithm.hxx"
#include "openturns/ClassificationRandomForestResult.hxx"

#ifdef OPENTURNS_HAVE_RANGER
#include <Forest.h>
#include <ForestClassification.h>
#include <Data.h>
#include <DataDouble.h>
#endif

BEGIN_NAMESPACE_OPENTURNS

/**
 * @class ClassificationRandomForestAlgorithm
 *
 * Wraps the Ranger ForestClassification for use within OpenTURNS.
 *
 * The output sample must contain integer-valued class labels encoded as
 * Scalar (e.g. 0.0, 1.0, 2.0, …).  Each unique value is treated as a
 * separate class.  The trained model predicts the most-probable class
 * label and, optionally, the per-class probability vector.
 */
class OT_API ClassificationRandomForestAlgorithm
  : public MetaModelAlgorithm
{
  CLASSNAME

public:

  /** Default constructor */
  ClassificationRandomForestAlgorithm();

  /**
   * Parameters constructor
   *
   * @param inputSample   Training inputs  (n × d)
   * @param outputSample  Training outputs (n × 1) — integer-valued class labels
   */
  ClassificationRandomForestAlgorithm(const Sample & inputSample,
                                      const Sample & outputSample);

  /** Virtual constructor */
  ClassificationRandomForestAlgorithm * clone() const override;

  /** Train the forest and build the result object */
  void run() override;

  /**
   * Predict the most-probable class label for each row of inputSample.
   * Returns a Sample of size (n × 1) whose values are class labels
   * (same numeric encoding as the training outputSample).
   */
  Sample predict(const Sample & inputSample) const;

  /**
   * Return the per-tree class label predicted by each individual tree for
   * each row of inputSample (predict_all=true in Ranger terms).
   *
   * Returns a Sample of shape (n × num_trees) where column t contains the
   * numeric class label voted by tree t — NOT a probability.
   *
   * To obtain per-class probabilities, a probability forest (ForestProbability)
   * must be used instead; that is a separate forest type in Ranger.
   */
  Sample predictAllTrees(const Sample & inputSample) const;

  /** Return the Function wrapping predict() */
  Function getRandomForestAsFunction();

  /** Ordered list of class values seen during training */
  Point getClassValues() const { return classValues_; }

private:
  ClassificationRandomForestResult result_;

#ifdef OPENTURNS_HAVE_RANGER

  UnsignedInteger num_trees_;
  PersistentCollection<PersistentCollection<UnsignedInteger>>                        split_var_ids_;
  PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger>>>  child_node_ids_;
  PersistentCollection<PersistentCollection<Scalar>>                                 split_values_;
  PersistentCollection<UnsignedInteger>                                              is_ordered_variable_;
  /** Class values discovered by Ranger during training (stored as doubles) */
  Point                                                                              classValues_;

  // ------------------------------------------------------------------
  // Inner helper: bridge between OpenTURNS Sample and ranger::Data
  // ------------------------------------------------------------------
  /**
   * @class DataRanger
   *
   * Adapts a pair of OT Samples (inputs / outputs) to the ranger::Data
   * interface without copying the underlying memory.
   *
   * For classification, Ranger reads the response through get_y() and
   * expects a single column of numeric class labels.  The distinct values
   * are collected internally by Ranger to build the class map.
   */
  class DataRanger : public ranger::Data
  {
  public:
    DataRanger() = default;

    DataRanger(const Sample & input,
               const Sample & output,
               const Description & inputDescription,
               const Description & outputDescription)
      : ranger::Data()
      , x_(input)
      , y_(output)
    {
      if (output.getDimension() != 1)
        throw InvalidArgumentException(HERE)
            << "ClassificationRandomForestAlgorithm: output sample must be "
               "1-dimensional (class labels), got dimension "
            << output.getDimension();

      num_rows        = input.getSize();
      num_cols        = input.getDimension();
      num_cols_no_snp = num_cols;

      for (UnsignedInteger i = 0; i < input.getDimension(); ++i)
        variable_names.push_back("var_" + inputDescription[i]);

      // Ranger expects the dependent variable name to be the last entry
      variable_names.push_back("var_" + outputDescription[0]);
    }

    DataRanger(const DataRanger &)             = delete;
    DataRanger & operator=(const DataRanger &) = delete;
    virtual ~DataRanger() override             = default;

    // ---- ranger::Data pure-virtual interface -------------------------

    double get_x(size_t row, size_t col) const override
    {
      if (col >= num_cols) {
        col = getUnpermutedVarID(col);
        row = getPermutedSampleID(row);
      }
      if (col < num_cols_no_snp)
        return x_(row, col);
      throw InvalidArgumentException(HERE) << "get_x: column index out of range";
    }

    double get_y(size_t row, size_t col) const override
    {
      // col is always 0 for single-output classification
      return y_(row, col);
    }

    /**
     * Called by Ranger before it starts filling response data.
     * For classification, y_cols == 1.
     */
    void reserveMemory(size_t y_cols) override
    {
      if (y_cols != 1)
        throw InvalidArgumentException(HERE)
            << "ClassificationRandomForestAlgorithm::DataRanger::reserveMemory: "
               "expected y_cols == 1, got "
            << y_cols;
    }

    void set_x(size_t col, size_t row, double value, bool & /*error*/) override
    {
      x_(row, col) = value;
    }

    void set_y(size_t col, size_t row, double value, bool & /*error*/) override
    {
      y_(row, col) = value;
    }

  private:
    Sample x_;
    Sample y_;
  }; // DataRanger

  // ------------------------------------------------------------------
  // Inner helper: EvaluationImplementation wrapping predict()
  // ------------------------------------------------------------------
  /**
   * @class ClassificationForestEvaluation
   *
   * Bridges the ClassificationRandomForestAlgorithm::predict() method to
   * the OpenTURNS Function / EvaluationImplementation interface so that
   * the trained model can be used everywhere a Function is accepted.
   *
   * The function maps ℝ^d → ℝ^1 where the single output is the predicted
   * class label (same numeric encoding as the training labels).
   */
  class ClassificationForestEvaluation : public EvaluationImplementation
  {
  public:
    explicit ClassificationForestEvaluation(const ClassificationRandomForestAlgorithm & algorithm)
      : EvaluationImplementation()
      , algorithm_(algorithm.clone())
    {}

    ClassificationForestEvaluation * clone() const override
    {
      return new ClassificationForestEvaluation(*this);
    }

    Point operator()(const Point & point) const override
    {
      Sample sample(1, point);
      return algorithm_->predict(sample)[0];
    }

    Sample operator()(const Sample & sample) const override
    {
      return algorithm_->predict(sample);
    }

    UnsignedInteger getInputDimension() const override
    {
      return algorithm_->getInputSample().getDimension();
    }

    UnsignedInteger getOutputDimension() const override
    {
      // predict() always returns a single class-label column
      return 1;
    }

    Description getInputDescription() const override
    {
      return algorithm_->getInputSample().getDescription();
    }

    Description getOutputDescription() const override
    {
      return algorithm_->getOutputSample().getDescription();
    }

    Description getDescription() const override
    {
      Description desc(getInputDescription());
      desc.add(getOutputDescription());
      return desc;
    }

    String __repr__() const override
    {
      return OSS() << "ClassificationForestEvaluation";
    }

    String __str__(const String & offset = "") const override
    {
      return OSS() << offset << __repr__();
    }

  private:
    // Owned copy of the algorithm (contains the serialised forest state)
    ClassificationRandomForestAlgorithm * algorithm_;
  }; // ClassificationForestEvaluation

#endif // OPENTURNS_HAVE_RANGER
}; // ClassificationRandomForestAlgorithm

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_CLASSIFICATIONRANDOMFORESTALGORITHM_HXX */