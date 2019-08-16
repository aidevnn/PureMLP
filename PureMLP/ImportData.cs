using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace PureMLP
{
    public static class ImportData
    {

        public static (double[][], double[][], double[][], double[][]) DigitsDataset(double ratio)
        {
            var raw = File.ReadAllLines("datasets/digits.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            int dim0 = data.Length / 65;

            var array = Enumerable.Range(0, dim0).Select(i => Enumerable.Range(0, 65).Select(j => data[i * 65 + j]).ToArray()).ToArray();

            var idx0 = (int)(dim0 * ratio);
            (var X, var y) = Utils.Split(array, axis: 1, idx: 64);

            X = X.Select(a => a.Select(v => v / 16.0).ToArray()).ToArray();
            y = y.SelectMany(i => i).Select(i => { var d = new double[10]; d[Convert.ToInt32(i)] = 1; return d; }).ToArray();

            (var trainX, var testX) = Utils.Split(X, axis: 0, idx: idx0);
            (var trainY, var testY) = Utils.Split(y, axis: 0, idx: idx0);

            Console.WriteLine($"Train on {trainX.Length} / Test on {testX.Length}");
            return (trainX, trainY, testX, testY);
        }

        public static (double[][], double[][], double[][], double[][]) IrisDataset(double ratio)
        {
            var raw = File.ReadAllLines("datasets/iris.csv").ToArray();
            var data = raw.SelectMany(l => l.Split(',')).Select(double.Parse).ToArray();
            int dim0 = data.Length / 7;

            var array = Enumerable.Range(0, dim0).Select(i => Enumerable.Range(0, 7).Select(j => data[i * 7 + j]).ToArray()).ToArray();

            var idx0 = (int)(dim0 * ratio);
            (var train, var test) = Utils.Split(array, axis: 0, idx: idx0);

            (var trainX, var trainY) = Utils.Split(train, axis: 1, idx: 4);
            (var testX, var testY) = Utils.Split(test, axis: 1, idx: 4);

            var vmax = Enumerable.Range(0, 4).Select(i => array.Max(a => a[i])).ToArray();
            trainX = trainX.Select(a => Enumerable.Range(0, 4).Select(i => a[i] / vmax[i]).ToArray()).ToArray();
            testX = testX.Select(a => Enumerable.Range(0, 4).Select(i => a[i] / vmax[i]).ToArray()).ToArray();

            Console.WriteLine($"Train on {trainX.Length} / Test on {testX.Length}");
            return (trainX, trainY, testX, testY);
        }
    }
}
