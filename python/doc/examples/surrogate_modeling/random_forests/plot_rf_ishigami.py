"""
Quick start:  the Ishigami function
===================================
"""

# %%
#
# In this example, we create a polynomial chaos for the
# :ref:`Ishigami function<use-case-ishigami>`.
# We create a sparse polynomial with maximum total degree equal to 8
# using the :class:`~openturns.FunctionalChaosAlgorithm` class.
#

# Define the model
# ----------------

from openturns.usecases import ishigami_function
import openturns as ot
import math

# We load the Ishigami model :
im = ishigami_function.IshigamiModel()

# The `IshigamiModel` data class contains the input distribution :math:`\vect{X}=(X_1, X_2, X_3)` in `im.distribution` and the Ishigami function in `im.model`.
# We also have access to the input variable names with
input_names = im.distribution.getDescription()


sampleSize = 10000
inputTrain = im.distribution.getSample(sampleSize)
outputTrain = im.model(inputTrain)


rf = ot.RandomForestRegressionAlgorithm(inputTrain, outputTrain, 1)
rf.run()

n_valid = 1000
inputTest = im.distribution.getSample(n_valid)
outputTest = im.model(inputTest)

# Validate random forest

MM = rf.getRandomForestAsFunction()
rf_predictions = MM(inputTest)
val_rf = ot.MetaModelValidation(outputTest, rf_predictions)
r2Score_rf = val_rf.computeR2Score()[0]
print(r2Score_rf)

res = rf.getResult()
MM = res.getMetaModel()
rf_predictions = MM(inputTest)
val_rf = ot.MetaModelValidation(outputTest, rf_predictions)
r2Score_rf = val_rf.computeR2Score()[0]
print(r2Score_rf)
# %%
