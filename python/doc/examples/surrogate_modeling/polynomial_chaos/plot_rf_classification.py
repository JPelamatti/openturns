import openturns as ot
from matplotlib import pyplot as plt
import openturns.viewer as otv

ot.Log.Show(ot.Log.ALL)

Id = ot.IdentityMatrix(2)
atoms = [
    ot.Normal([1.0, 2.0], [0.5, 0.8], Id),
    ot.Normal([1.0, -2.0], [0.9, 0.8], Id),
    ot.Normal([-1.0, 0.0], [0.5, 0.6], Id),
]
weights = [0.3, 0.3, 0.4]
mixture = ot.Mixture(atoms, weights)
data = mixture.getSample(1000)
classifier = ot.MixtureClassifier(mixture)
graph = mixture.drawPDF(data.getMin(), data.getMax())
graph.setLegendPosition("")
graph.setTitle("MixtureClassifier example")
classes = classifier.classify(data)
palette = ot.Drawable.BuildDefaultPalette(len(atoms))
symbols = ot.Drawable.GetValidPointStyles()
for i in range(classes.getSize()):
    index = classes[i]
    graph.add(
        ot.Cloud(
            [data[i]], palette[index % len(palette)], symbols[index % len(symbols)]
        )
    )

fig = plt.figure(figsize=(4, 4))
axis = fig.add_subplot(111)
axis.set_xlim(auto=True)
otv.View(graph, figure=fig, axes=[axis], add_legend=False)

# %%
rf = ot.ClassificationRandomForestAlgorithm(
    data, [[classes[i]] for i in range(data.getSize())]
)
# %%
rf.run()
# %%
data_test = mixture.getSample(1000)
pred = rf.predict(data_test)
# %%
graph2 = mixture.drawPDF(data.getMin(), data.getMax())
graph2.setLegendPosition("")
graph2.setTitle("Classification random forest example")
classes2 = [int(c[0]) for c in pred]
palette2 = ot.Drawable.BuildDefaultPalette(len(atoms))
symbols2 = ot.Drawable.GetValidPointStyles()
for i in range(len(classes2)):
    index = classes2[i]
    graph2.add(
        ot.Cloud(
            [data_test[i]],
            palette2[index % len(palette2)],
            symbols2[index % len(symbols2)],
        )
    )
fig2 = plt.figure(figsize=(4, 4))
axis2 = fig2.add_subplot(111)
axis2.set_xlim(auto=True)
otv.View(graph2, figure=fig2, axes=[axis2], add_legend=False)

# %%
pred_all = rf.predictAllTrees(data_test)

# %%
fun = rf.getRandomForestAsFunction()
# %%
