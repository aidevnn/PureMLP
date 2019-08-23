using System;
using System.Linq;
namespace PureMLP
{
    public interface ILosses
    {
        string Name { get; }
        double Loss(NDarray y, NDarray p);
        NDarray Grad(NDarray y, NDarray p);
    }

    public interface IAccuracies
    {
        string Name { get; }
        double Accuracy(NDarray y, NDarray p);
    }

    public class MeanSquaredLoss : ILosses
    {
        public string Name => "MeanSquaredLoss";

        double floss(double y, double p) => (y - p) * (y - p) / 2;
        double fgrad(double y, double p) => p - y;

        public NDarray Grad(NDarray y, NDarray p) => NDarray.ApplyFunc(y, p, fgrad);

        public double Loss(NDarray y, NDarray p) => NDarray.ApplyFunc(y, p, floss).Data.Average();
    }

    public class CrossEntropyLoss : ILosses
    {
        public string Name => "CrossEntropyLoss";

        double floss(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-12, Math.Max(1e-12, p));
            return -y * Math.Log(p0) - (1 - y) * Math.Log(1 - p0);
        }

        double fgrad(double y, double p)
        {
            var p0 = Math.Min(1 - 1e-12, Math.Max(1e-12, p));
            return -y / p0 + (1 - y) / (1 - p0);
        }

        public NDarray Grad(NDarray y, NDarray p) => NDarray.ApplyFunc(y, p, fgrad);

        public double Loss(NDarray y, NDarray p) => NDarray.ApplyFunc(y, p, floss).Data.Average();
    }

    public class RoundAccuracy : IAccuracies
    {
        public string Name => "RoundAccuracy";

        double facc(double y, double p) => Math.Abs(Math.Round(y) - Math.Round(p)) < 1e-6 ? 1.0 : 0.0;

        public double Accuracy(NDarray y, NDarray p)
        {
            var c = NDarray.ApplyFunc(y, p, facc);
            return c.Data.Select(i => Math.Abs(i - 1.0) < 1e-6 ? 1.0 : 0.0).Average();
        }
    }

    public class ArgMaxAccuracy : IAccuracies
    {
        public string Name => "ArgMaxAccuracy";

        public double Accuracy(NDarray y, NDarray p)
        {
            if (y.Shape.Length != 2 || p.Shape.Length != 2)
                throw new Exception();

            int n = y.Shape[0];
            int s = y.Shape[1];
            return Enumerable.Range(0, n).Select(i => Argmax(y.Data, i * s, i * s + s) == Argmax(p.Data, i * s, i * s + s) ? 1.0 : 0.0).Average();
        }

        public static int Argmax(double[] arr, int start, int end)
        {
            int bestIdx = 0;
            double bestValue = double.MinValue;
            for (int i = start, j = 0; i < end; ++i, ++j)
            {
                var v = arr[i];
                if (v > bestValue)
                {
                    bestValue = v;
                    bestIdx = j;
                }
            }

            return bestIdx;
        }
    }
}
