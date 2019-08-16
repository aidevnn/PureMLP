# PureMLP
Simple Multi Layers Perceptron

### Xor Dataset

```
var net = new Network(new SGD(0.2, 0.2), new CrossEntropyLoss(), new RoundAccuracy());
net.AddLayers(2, (8, Activation.Tanh), (1, Activation.Sigmoid));
net.Summary();
net.Fit(X, y, epochs, displayEpochs);
```

Output:
```
Hello World, MLP on Xor Dataset.
Summary
Network: SGD[lr:0.2 momentum:0.2] / CrossEntropyLoss / RoundAccuracy
Input  Shape:2
Layer: Dense-Tanh       Parameters:    24 Nodes[In:   2 -> Out:   8]
Layer: Dense-Sigmoid    Parameters:     9 Nodes[In:   8 -> Out:   1]
Output Shape:1
Total Parameters:33

Epoch    0/50 loss:0.681172 acc:0.750
Epoch    5/50 loss:0.544583 acc:1.000
Epoch   10/50 loss:0.460464 acc:1.000
Epoch   15/50 loss:0.381765 acc:1.000
Epoch   20/50 loss:0.310326 acc:1.000
Epoch   25/50 loss:0.249365 acc:1.000
Epoch   30/50 loss:0.200355 acc:1.000
Epoch   35/50 loss:0.162452 acc:1.000
Epoch   40/50 loss:0.133602 acc:1.000
Epoch   45/50 loss:0.111634 acc:1.000
Epoch   50/50 loss:0.094741 acc:1.000
Time:0 ms
[0 0] = [0] -> 0.039697
[0 1] = [1] -> 0.901709
[1 0] = [1] -> 0.912038
[1 1] = [0] -> 0.123077
```

### Iris Dataset
This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 

Source : https://archive.ics.uci.edu/ml/datasets/iris

Example of MLP network.
```
(var trainX, var trainY, var testX, var testY) = ImportData.IrisDataset(ratio: 0.8);
var net = new Network(new SGD(0.025, 0.2), new MeanSquaredLoss(), new ArgMaxAccuracy());
net.AddLayers(4, (5, Activation.Tanh), (3, Activation.Sigmoid));
net.Summary();

net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
net.Test(testX, testY);
```

Output:

```
Hello World, MLP on Iris Dataset.
Train on 120 / Test on 30
Summary
Network: SGD[lr:0.025 momentum:0.2] / MeanSquaredLoss / ArgMaxAccuracy
Input  Shape:4
Layer: Dense-Tanh       Parameters:    25 Nodes[In:   4 -> Out:   5]
Layer: Dense-Sigmoid    Parameters:    18 Nodes[In:   5 -> Out:   3]
Output Shape:3
Total Parameters:43

Epoch    0/50 loss:0.136851 acc:0.017
Epoch    5/50 loss:0.098196 acc:0.633
Epoch   10/50 loss:0.073116 acc:0.717
Epoch   15/50 loss:0.061589 acc:0.825
Epoch   20/50 loss:0.056082 acc:0.783
Epoch   25/50 loss:0.052963 acc:0.875
Epoch   30/50 loss:0.050015 acc:0.900
Epoch   35/50 loss:0.048205 acc:0.908
Epoch   40/50 loss:0.044819 acc:0.942
Epoch   45/50 loss:0.042106 acc:0.925
Epoch   50/50 loss:0.039562 acc:0.942
Time:19 ms
Test loss:0.0341 acc:1.0000
```

### Digits Dataset

This dataset is made up of 1797 8x8 images. Each image is of a hand-written digit.

Source : https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html

Example of MLP network.
```
(var trainX, var trainY, var testX, var testY) = ImportData.DigitsDataset(ratio: 0.9);
var net = new Network(new SGD(0.025, 0.2), new CrossEntropyLoss(), new ArgMaxAccuracy());
net.AddLayers(64, (32, Activation.Sigmoid), (10, Activation.Sigmoid));
net.Summary();

net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
net.Test(testX, testY);
```

Output:

```
Hello World, MLP on Digits Dataset.
Train on 1617 / Test on 180
Summary
Network: SGD[lr:0.025 momentum:0.2] / CrossEntropyLoss / ArgMaxAccuracy
Input  Shape:64
Layer: Dense-Sigmoid    Parameters:  2080 Nodes[In:  64 -> Out:  32]
Layer: Dense-Sigmoid    Parameters:   330 Nodes[In:  32 -> Out:  10]
Output Shape:10
Total Parameters:2410

Epoch    0/50 loss:0.363840 acc:0.332
Epoch    5/50 loss:0.060400 acc:0.948
Epoch   10/50 loss:0.033789 acc:0.966
Epoch   15/50 loss:0.024472 acc:0.981
Epoch   20/50 loss:0.020818 acc:0.979
Epoch   25/50 loss:0.016104 acc:0.988
Epoch   30/50 loss:0.014087 acc:0.989
Epoch   35/50 loss:0.011934 acc:0.995
Epoch   40/50 loss:0.010352 acc:0.995
Epoch   45/50 loss:0.008865 acc:0.996
Epoch   50/50 loss:0.007940 acc:0.997
Time:2125 ms
Test loss:0.0388 acc:0.9389
```