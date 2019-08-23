using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace PureMLP
{
    public class Network
    {
        public Network(IOptimizer optimizer,ILosses losses, IAccuracies accuracies)
        {
            this.optimizer = optimizer;
            this.losses = losses;
            this.accuracies = accuracies;
        }

        readonly IOptimizer optimizer;
        readonly ILosses losses;
        readonly IAccuracies accuracies;

        List<Layer> layers = new List<Layer>();

        public void AddLayer(Layer layer)
        {
            if (layers.Count != 0)
                layer.SetInputNodes(layers.Last().OutputNodes);

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public void AddLayers(int inputNodes, params (int, Activation)[] args)
        {
            for (int k = 0; k < args.Length; ++k)
            {
                (int outputNodes, Activation name) = args[k];
                Layer layer;
                if (k == 0)
                    layer = new Layer(outputNodes, inputNodes, Activations.Get(name));
                else
                    layer = new Layer(outputNodes, Activations.Get(name));

                AddLayer(layer);
            }
        }

        public void Training(bool istraining) => layers.ForEach(l => l.IsTraining = istraining);

        public NDarray Forward(NDarray X)
        {
            foreach (var layer in layers)
                X = layer.Forward(X);

            return X;
        }

        public void Backward(NDarray accumGrad)
        {
            foreach (var layer in layers.Reverse<Layer>())
                accumGrad = layer.Backward(accumGrad);
        }

        public (double,double) TestOnBatch(NDarray X, NDarray y)
        {
            var yp = Forward(X);
            var loss = losses.Loss(y, yp);
            var acc = accuracies.Accuracy(y, yp);
            //Console.WriteLine($"Test loss:{loss:F6} acc:{acc:F6}");
            return (loss, acc);
        }

        public (double, double) TrainOnBatch(NDarray X, NDarray y)
        {
            var yp = Forward(X);
            var loss = losses.Loss(y, yp);
            var acc = accuracies.Accuracy(y, yp);
            var grad = losses.Grad(y, yp);
            Backward(grad);

            return (loss, acc);
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Network: {optimizer.Name} / {losses.Name} / {accuracies.Name}");
            Console.WriteLine($"Input  Shape:{layers[0].InputNodes}");
            int tot = 0;
            foreach (var layer in layers)
            {
                Console.WriteLine($"Layer: Dense-{layer.activation.Name,-10} Parameters: {layer.Params,5} Nodes[In:{layer.InputNodes,4} -> Out:{layer.OutputNodes,4}]");
                tot += layer.Params;
            }

            Console.WriteLine($"Output Shape:{layers.Last().OutputNodes}");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public void Fit(NDarray trainX, NDarray trainY, int epochs, int displayEpochs = 10, int batchsize = 50, bool shuffle = true)
        {
            var sw = Stopwatch.StartNew();

            Training(true);
            for(int e = 0; e <= epochs; ++e)
            {
                List<double> ltLoss = new List<double>();
                List<double> ltAcc = new List<double>();

                var batch = NDarray.BatchIterator(trainX, trainY, batchsize, shuffle);
                foreach ((var X, var y) in batch)
                {
                    (double loss, double acc) = TrainOnBatch(X, y);
                    ltLoss.Add(loss);
                    ltAcc.Add(acc);
                }

                if (e % displayEpochs == 0)
                    Console.WriteLine($"Epoch {e,4}/{epochs} loss:{ltLoss.Average():0.000000} acc:{ltAcc.Average():0.000}");
            }

            Console.WriteLine($"Time:{sw.ElapsedMilliseconds} ms");
        }

        public void Test(NDarray testX, NDarray testY)
        {
            (var loss, var acc) = TestOnBatch(testX, testY);
            Console.WriteLine($"Test loss:{loss:0.0000} acc:{acc:0.0000}");
        }
    }
}
