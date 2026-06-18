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
#ifndef OPENTURNS_RANDOMFORESTALGORITHM_HXX
#define OPENTURNS_RANDOMFORESTALGORITHM_HXX

#include "openturns/MetaModelAlgorithm.hxx"
#include "openturns/RandomForestResult.hxx"
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

class OT_API RandomForestRegressionAlgorithm
  : public MetaModelAlgorithm
{
  CLASSNAME

public:

  /** Default constructor */
  RandomForestRegressionAlgorithm();

  /** Parameters constructor */
  RandomForestRegressionAlgorithm(const Sample & inputSample,
                      const Sample & outputSample,
                      const UnsignedInteger importanceMode = 0);

  /** Virtual constructor */
  RandomForestRegressionAlgorithm* clone() const override;

  void run() override;
  Sample predict(const Sample& inputSample) const;
  Function getRandomForestAsFunction();

    
private:
  RandomForestResult result_;
  UnsignedInteger importanceMode_;
#ifdef OPENTURNS_HAVE_RANGER
  // convertToRangerData(const Sample& input, const Sample& output);
  // std::shared_ptr<ranger::ForestRegression> forest_ = 0;
  UnsignedInteger num_trees_;
  PersistentCollection<PersistentCollection<UnsignedInteger> > split_var_ids_;
  PersistentCollection<PersistentCollection<PersistentCollection<UnsignedInteger> > > child_node_ids_;
  PersistentCollection<PersistentCollection<Scalar> > split_values_;
  PersistentCollection<UnsignedInteger> is_ordered_variable_;

  class DataRanger
    : public ranger::Data
    {
      public:
        DataRanger() = default;
        // Avoid having to copy Samples simply to modify their Description
        DataRanger(const Sample& input, const Sample& output, const Description& inputDescription, const Description& outputDescription)
          :ranger::Data()
          // ,num_rows(input.getSize())
          // ,num_rows_rounded(0)
          // ,num_cols(input.getSize() + 1)
          // ,snp_data(0)
          // ,num_cols_no_snp(input.getSize() + 1)
          // ,externalData(true)
          // ,index_data(0)
          // ,max_num_unique_values(0)
          // ,order_snps(false)
          // ,any_na(false) 
          ,x(input)
          ,y(output)
        {
          num_rows = input.getSize();
          num_cols = input.getDimension();
          num_cols_no_snp = num_cols;
          for (size_t i = 0; i < input.getDimension(); ++i) {
              variable_names.push_back("var_" + inputDescription[i]);
          }
          variable_names.push_back("var_" + outputDescription[0]);
        };
        
        DataRanger(const DataRanger&) = delete;
        DataRanger& operator=(const DataRanger&) = delete;

        virtual ~DataRanger() override = default;

        double get_x(size_t row, size_t col) const override {
          // Use permuted data for corrected impurity importance
          //LOGWARN(OSS() << "Dans get_x(" << row << ", "<< col << ")");
          //LOGWARN(OSS() << "valeur = " << x(row, col));
          size_t col_permuted = col;
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

        double get_y(size_t row, size_t col) const override {
          //LOGWARN(OSS() << "Dans get_y(" << row << ", "<< col << ")");
          return y(row, col);
        }

        void reserveMemory(size_t y_cols) override {
          //LOGWARN(OSS() << "Dans reserveMemory avec y_cols = " << y_cols);
          if (y_cols != 1) throw InvalidArgumentException(HERE) << "Only 1 output dimension possible";
        }

        void set_x(size_t col, size_t row, double value, bool& error) override {
          //LOGWARN(OSS() << "Dans set_x(" << row << ", "<< col << ") avec value = " << value);
          x(row, col) = value;
        }

        void set_y(size_t col, size_t row, double value, bool& error) override {
          //LOGWARN(OSS() << "Dans set_y(" << row << ", "<< col << ") avec value = " << value);
          y(row, col) = value;
        }

      private:
        Sample x;
        Sample y;
      };

  // Helper class to evaluate the random forest
  class RandomForestEvaluation: public EvaluationImplementation
  {
  public:
    // Parameter constructor
    RandomForestEvaluation(const RandomForestRegressionAlgorithm & algorithm)
      : EvaluationImplementation()
      , algorithm_(algorithm.clone())
    {
      // Nothing to do
    }

    RandomForestEvaluation * clone() const override
    {
      return new RandomForestEvaluation(*this);
    }

    // It is a simple call to the predict of the algo
    Point operator() (const Point & point) const override
    {
      Sample sample(1, point);
      const Point value(algorithm_->predict(sample)[0]);
      return value;
    }

    // It is a simple call to the predict of the algo
    Sample operator() (const Sample & sample) const override
    {
      const Sample values(algorithm_->predict(sample));
      return values;
    }

    // The following are probably not necessary, and override should be wrong as it is not an accessor of the parent class
    UnsignedInteger getInputDimension() const override
    {
      return algorithm_->getInputSample().getDimension();
    }

    UnsignedInteger getOutputDimension() const override
    {
      return algorithm_->getOutputSample().getDimension();
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
      Description description(getInputDescription());
      description.add(getOutputDescription());
      return description;
    }

    String __repr__() const override
    {
      OSS oss;
      // Don't print algorithm_ here as it will result in an infinite loop!
      oss << "RandomForestEvaluation";
      return oss;
    }

    String __str__(const String & offset = "") const override
    {
      // Don't print algorithm_ here as it will result in an infinite loop!
      return OSS() << offset << __repr__();
    }

  private:
    RandomForestRegressionAlgorithm* algorithm_;
  }; // RandomForestEvaluation

#endif
};

END_NAMESPACE_OPENTURNS

#endif /* OPENTURNS_RANDOMFORESTALGORITHM_HXX */
