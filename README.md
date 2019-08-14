# PureMLP
Simple Multi Layers Perceptron

## Xor Dataset

```
Hello World, MLP on Xor Dataset.
Summary
Network: SGD[lr:0.2 momentum:0.2] / CrossEntropyLoss / RoundAccuracy
Input  Shape:2
Layer: Dense-Tanh       Parameters:    24 Nodes[In:   2 -> Out:8]
Layer: Dense-Sigmoid    Parameters:     9 Nodes[In:   8 -> Out:1]
Output Shape:1
Total Parameters:33

Epoch    0/1000 loss:0.891239 acc:0.500
Epoch  100/1000 loss:0.034214 acc:1.000
Epoch  200/1000 loss:0.014472 acc:1.000
Epoch  300/1000 loss:0.008966 acc:1.000
Epoch  400/1000 loss:0.006440 acc:1.000
Epoch  500/1000 loss:0.005003 acc:1.000
Epoch  600/1000 loss:0.004080 acc:1.000
Epoch  700/1000 loss:0.003439 acc:1.000
Epoch  800/1000 loss:0.002969 acc:1.000
Epoch  900/1000 loss:0.002610 acc:1.000
Epoch 1000/1000 loss:0.002327 acc:1.000
Time:29 ms
[0 0] = [0] -> 0.000659
[0 1] = [1] -> 0.998016
[1 0] = [1] -> 0.996845
[1 1] = [0] -> 0.003487
```

## Iris Dataset

```
Hello World, MLP on Iris Dataset.
Train on 120 / Test on 30
Summary
Network: SGD[lr:0.025 momentum:0.2] / MeanSquaredLoss / ArgMaxAccuracy
Input  Shape:4
Layer: Dense-Tanh       Parameters:    25 Nodes[In:   4 -> Out:5]
Layer: Dense-Sigmoid    Parameters:    18 Nodes[In:   5 -> Out:3]
Output Shape:3
Total Parameters:43

Epoch    0/50 loss:0.135775 acc:0.058
Epoch    5/50 loss:0.110311 acc:0.383
Epoch   10/50 loss:0.101324 acc:0.658
Epoch   15/50 loss:0.087375 acc:0.658
Epoch   20/50 loss:0.072502 acc:0.792
Epoch   25/50 loss:0.064256 acc:0.800
Epoch   30/50 loss:0.059118 acc:0.767
Epoch   35/50 loss:0.056343 acc:0.775
Epoch   40/50 loss:0.054892 acc:0.842
Epoch   45/50 loss:0.052422 acc:0.867
Epoch   50/50 loss:0.050464 acc:0.833
Time:24 ms
Test loss:0.0482 acc:0.9667
```

## Digits Dataset

```
Hello World, MLP on Digits Dataset.
Train on 1617 / Test on 180
Summary
Network: SGD[lr:0.025 momentum:0.2] / CrossEntropyLoss / ArgMaxAccuracy
Input  Shape:64
Layer: Dense-Sigmoid    Parameters:  2080 Nodes[In:  64 -> Out:32]
Layer: Dense-Sigmoid    Parameters:   330 Nodes[In:  32 -> Out:10]
Output Shape:10
Total Parameters:2410

Epoch    0/50 loss:0.363429 acc:0.293
Epoch    5/50 loss:0.059072 acc:0.944
Epoch   10/50 loss:0.033318 acc:0.974
Epoch   15/50 loss:0.023572 acc:0.982
Epoch   20/50 loss:0.019102 acc:0.985
Epoch   25/50 loss:0.015626 acc:0.987
Epoch   30/50 loss:0.013374 acc:0.992
Epoch   35/50 loss:0.011325 acc:0.993
Epoch   40/50 loss:0.009809 acc:0.993
Epoch   45/50 loss:0.008653 acc:0.996
Epoch   50/50 loss:0.007806 acc:0.998
Time:2159 ms
Test loss:0.0383 acc:0.9389
```