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
            var ndX = new NDarray(X);
            var ndY = new NDarray(y);

            var net = new Network(new SGD(0.2, 0.2), new CrossEntropyLoss(), new RoundAccuracy());
            net.AddLayers(2, (8, Activation.Tanh), (1, Activation.Sigmoid));

            if (summary)
                net.Summary();

            net.Fit(ndX, ndY, epochs, displayEpochs);

            if (summary)
            {
                var yp = net.Forward(ndX);
                for (int k = 0; k < 4; ++k)
                    Console.WriteLine($"[{X[k].Glue()}] = [{y[k][0]}] -> {yp.Data[k]:0.000000}");
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

        static void TestGEMM()
        {
            double[] data0 = Enumerable.Range(0, 4 * 5).Select(k => (double)Utils.Random.Next(0, 10)).ToArray();
            double[] data1 = Enumerable.Range(0, 5 * 3).Select(k => (double)Utils.Random.Next(0, 10)).ToArray();
            NDarray a0 = new NDarray(data0, 4, 5);
            NDarray a1 = new NDarray(data0, 5, 4);
            NDarray b0 = new NDarray(data1, 5, 3);
            NDarray b1 = new NDarray(data1, 3, 5);
            var c00 = NDarray.GemmABC(a0, b0);
            var c01 = NDarray.GemmATBC(a0, b1);
            var c10 = NDarray.GemmTABC(a1, b0);
            var c11 = NDarray.GemmTATBC(a1, b1);

            Console.WriteLine($"a={a0}");
            Console.WriteLine($"b={b0}");
            Console.WriteLine("np.dot(a,b)");
            Console.WriteLine(c00);
            Console.WriteLine();

            Console.WriteLine($"a={a0}");
            Console.WriteLine($"b={b1}");
            Console.WriteLine("np.dot(a,b.T)");
            Console.WriteLine(c01);
            Console.WriteLine();

            Console.WriteLine($"a={a1}");
            Console.WriteLine($"b={b0}");
            Console.WriteLine("np.dot(a.T,b)");
            Console.WriteLine(c10);
            Console.WriteLine();

            Console.WriteLine($"a={a1}");
            Console.WriteLine($"b={b1}");
            Console.WriteLine("np.dot(a.T,b.T)");
            Console.WriteLine(c11);
            Console.WriteLine();

        }

        static void TestBatchIterator()
        {
            double[] data0 = Enumerable.Range(0, 8 * 2 * 5).Select(k => (double)Utils.Random.Next(0, 10)).ToArray();
            double[] data1 = Enumerable.Range(0, 8 * 2 * 2).Select(k => (double)Utils.Random.Next(0, 10)).ToArray();
            NDarray X = new NDarray(data0, 8, 2, 5);
            NDarray y = new NDarray(data1, 8, 2, 2);

            var l = NDarray.BatchIterator(X, y, 2, false);

            Console.WriteLine(X);
            foreach (var b in l) Console.WriteLine(b.Item1);

            Console.WriteLine();

            Console.WriteLine(y);
            foreach (var b in l) Console.WriteLine(b.Item2);
        }

        public static void Main(string[] args)
        {
            //TestXor(summary: true, displayEpochs: 5);
            //TestIris(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);
            TestDigits(summary: true, displayEpochs: 5);

            //for (int k = 0; k < 5; ++k) TestIris();

        }
    }
}
