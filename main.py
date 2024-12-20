import graphlearning as gl
import numpy as np
from CutSSL import *

labels = gl.datasets.load('cifar', labels_only=True)
W = gl.weightmatrix.knn('cifar', 100, metric='aet')
k = np.unique(labels).shape[0]
class_priors = gl.utils.class_priors(labels)

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

models = [gl.ssl.laplace(W, mean_shift=True), cut_ssl(W, class_priors, maxiter=[100,400], s=[0.0,0.1])]

for model in models:
  pred = model.fit(train_ind,train_labels,all_labels=labels)
  pred_labels = pred.argmax(1)

  accuracy = gl.ssl.ssl_accuracy(pred_labels,true_labels=labels,train_ind=train_ind)
  print(model.name + ': %.2f%%'%accuracy)
