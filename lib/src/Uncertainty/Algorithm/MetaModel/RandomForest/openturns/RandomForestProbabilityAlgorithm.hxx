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
#ifndef OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_HXX
#define OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_HXX

#include "openturns/MetaModelAlgorithm.hxx"
#include "openturns/RandomForestProbabilityResult.hxx"
#include "openturns/EvaluationImplementation.hxx"
#include "openturns/PersistentCollection.hxx"
#include <memory>

#ifdef OPENTURNS_HAVE_RANGER
#include <Data.h>
#include <DataDouble.h>
#include <Forest.h>
#include <ForestProbability.h>
#endif

BEGIN_NAMESPACE_OPENTURNS

class RandomForestProbabilityResult;

/**
 * @class RandomForestProbabilityAlgorithm
 *
 * Class for training random forests with class probability outputs using Ranger.
 */
class OT_API RandomForestProbabilityAlgorithm : public MetaModelAlgorithm {
  CLASSNAME

 public:
  /** Default constructor */
  RandomForestProbabilityAlgorithm();

  /** Parameters constructor */
  RandomForestProbabilityAlgorithm(const Sample &inputSample,
                                   const Sample &outputSample,
                                   const UnsignedInteger importanceMode = 0,
                                   const UnsignedInteger num_trees = 500,
                                   const UnsignedInteger min_node_size = 0,
                                   const UnsignedInteger min_bucket = 0,
                                   const UnsignedInteger max_depth = 0,
                                   const UnsignedInteger mtry = 0);

  /** Virtual constructor */
  RandomForestProbabilityAlgorithm *clone() const override;

  void run() override;
  Sample predict(const Sample &inputSample) const;
  Sample predictAllTrees(const Sample &inputSample) const;

  Function getRandomForestAsFunction();
  Point getClassValues() const;
  RandomForestProbabilityResult getResult() const;

 private:
#ifdef OPENTURNS_HAVE_RANGER
  std::shared_ptr<ranger::ForestProbability> forest_;
  RandomForestProbabilityResult result_;
  std::vector<uint> min_node_size_;
  std::vector<uint> min_bucket_;
  UnsignedInteger max_depth_;
  UnsignedInteger mtry_;
  UnsignedInteger importanceMode_;
  UnsignedInteger num_trees_;
  PersistentCollection<PersistentCollection<UnsignedInteger>> split_var_ids_;
  PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger>>> child_node_ids_;
  PersistentCollection<PersistentCollection<Scalar>> split_values_;
  PersistentCollection<UnsignedInteger> is_ordered_variable_;
  Point classValues_;

  class DataRanger : public ranger::Data {
   public:
    DataRanger() = default;
    DataRanger(const Sample &input, const Sample &output,
               const Description &inputDescription,
               const Description &outputDescription)
        : ranger::Data(), x(input), y(output) {
      num_rows = input.getSize();
      num_cols = input.getDimension();
      num_cols_no_snp = num_cols;
      for (size_t i = 0; i < input.getDimension(); ++i) {
        variable_names.push_back("var_" + inputDescription[i]);
      }
      variable_names.push_back("var_" + outputDescription[0]);
    };

    DataRanger(const DataRanger &) = delete;
    DataRanger &operator=(const DataRanger &) = delete;

    ~DataRanger() override = default;

    double get_x(size_t row, size_t col) const override {
      if (col >= num_cols) {
        col = getUnpermutedVarID(col);
        row = getPermutedSampleID(row);
      }

      if (col < num_cols_no_snp) {
        return x(row, col);
      } else {
        throw InvalidArgumentException(HERE) << "Outside the table";
      }
    }

    double get_y(size_t row, size_t col) const override { return y(row, col); }

    void reserveMemory(size_t y_cols) override {
      if (y_cols != 1)
        throw InvalidArgumentException(HERE)
            << "Only 1 output dimension possible";
    }

    void set_x(size_t col, size_t row, double value, bool &error) override {
      x(row, col) = value;
    }

    void set_y(size_t col, size_t row, double value, bool &error) override {
      y(row, col) = value;
    }

   private:
    Sample x;
    Sample y;
  };

  // Evaluation wrapper exposing the probability metamodel as a Function
  class RandomForestEvaluation : public EvaluationImplementation {
   public:
    RandomForestEvaluation(const RandomForestProbabilityAlgorithm &algorithm)
        : EvaluationImplementation(), algorithm_(algorithm.clone()) {}

    RandomForestEvaluation(const RandomForestEvaluation &other)
        : EvaluationImplementation(other), algorithm_(other.algorithm_->clone()) {}

    ~RandomForestEvaluation() override {
      delete algorithm_;
    }

    RandomForestEvaluation *clone() const override {
      return new RandomForestEvaluation(*this);
    }

    Point operator()(const Point &point) const override {
      Sample sample(1, point);
      return algorithm_->predict(sample)[0];
    }

    Sample operator()(const Sample &sample) const override {
      return algorithm_->predict(sample);
    }

    UnsignedInteger getInputDimension() const override {
      return algorithm_->getInputSample().getDimension();
    }

    UnsignedInteger getOutputDimension() const override {
      return algorithm_->getClassValues().getDimension();
    }

    Description getInputDescription() const override {
      return algorithm_->getInputSample().getDescription();
    }

    Description getOutputDescription() const override {
      const Point classValues(algorithm_->getClassValues());
      const UnsignedInteger numClasses = classValues.getDimension();
      Description desc(numClasses);
      for (UnsignedInteger i = 0; i < numClasses; ++i) {
        desc[i] = OSS() << "P(Y=" << classValues[i] << ")";
      }
      return desc;
    }

    String __repr__() const override {
      return "RandomForestProbabilityEvaluation";
    }

    String __str__(const String &offset = "") const override {
      return OSS() << offset << __repr__();
    }

   private:
    RandomForestProbabilityAlgorithm *algorithm_;
  };

#endif
};

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_RANDOMFORESTPROBABILITYALGORITHM_HXX */