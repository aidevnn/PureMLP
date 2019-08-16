using System;
using System.Diagnostics;
using System.Linq;
namespace PureMLP
{
    public class Program
    {
        static void TestXor(bool summary = false, int epochs = 50, int displayEpochs = 25)
        {
            Console.WriteLine("Hello World, MLP on Xor Dataset.");

            double[,] X0 = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
            double[,] y0 = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
            double[][] X = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 2).Select(j => X0[i, j]).ToArray()).ToArray();
            double[][] y = Enumerable.Range(0, 4).Select(i => Enumerable.Range(0, 1).Select(j => y0[i, j]).ToArray()).ToArray();

            var net = new Network(new SGD(0.2, 0.2), new CrossEntropyLoss(), new RoundAccuracy());
            net.AddLayers(2, (8, Activation.Tanh), (1, Activation.Sigmoid));

            if (summary)
                net.Summary();

            net.Fit(X, y, epochs, displayEpochs);

            if (summary)
            {
                var yp = net.Forward(X);
                for (int k = 0; k < 4; ++k)
                    Console.WriteLine($"[{X[k].Glue()}] = [{y[k][0]}] -> {yp[k][0]:0.000000}");
            }

            Console.WriteLine();
        }

        static void TestIris(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 10)
        {
            Console.WriteLine("Hello World, MLP on Iris Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportData.IrisDataset(ratio: 0.8);
            var net = new Network(new SGD(0.025, 0.2), new MeanSquaredLoss(), new ArgMaxAccuracy());
            net.AddLayers(4, (5, Activation.Tanh), (3, Activation.Sigmoid));

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        static void TestDigits(bool summary = false, int epochs = 50, int displayEpochs = 25, int batchsize = 100)
        {
            Console.WriteLine("Hello World, MLP on Digits Dataset.");

            (var trainX, var trainY, var testX, var testY) = ImportData.DigitsDataset(ratio: 0.9);
            var net = new Network(new SGD(0.025, 0.2), new CrossEntropyLoss(), new ArgMaxAccuracy());
            net.AddLayers(64, (32, Activation.Sigmoid), (10, Activation.Sigmoid));

            if (summary)
                net.Summary();

            net.Fit(trainX, trainY, epochs, displayEpochs, batchsize);
            net.Test(testX, testY);

            Console.WriteLine();
        }

        public static void Main(string[] args)
        {
            TestXor(summary: true, displayEpochs: 5);
            TestIris(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);

            //for (int k = 0; k < 5; ++k) TestIris();
        }
    }
}
