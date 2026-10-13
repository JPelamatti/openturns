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

import numpy as np
from openturns.usecases import ishigami_function
import openturns as ot
import math

# We load the Ishigami model :
im = ishigami_function.IshigamiModel()

# The `IshigamiModel` data class contains the input distribution :math:`\vect{X}=(X_1, X_2, X_3)` in `im.distribution` and the Ishigami function in `im.model`.
# We also have access to the input variable names with
input_names = im.distribution.getDescription()


sampleSize = 200
inputTrain = im.distribution.getSample(sampleSize)
outputTrain = im.model(inputTrain)
outputTrain = ot.Sample([[1] if yi[0] > 0 else [0] for yi in outputTrain])

rf = ot.RandomForestClassificationAlgorithm(inputTrain, outputTrain)
rf.run()

n_valid = 10000
inputTest = im.distribution.getSample(n_valid)
outputTest = im.model(inputTest)
outputTest = ot.Sample([[1] if yi[0] > 0 else [0] for yi in outputTest])

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
print(np.sum(np.array(outputTest) == np.array(rf_predictions))/outputTest.getSize())