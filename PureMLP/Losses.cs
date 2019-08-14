using System;
using System.Linq;
namespace PureMLP
{
    public interface ILosses
    {
        string Name { get; }
        double Loss(double[][] y, double[][] p);
        double[][] Grad(double[][] y, double[][] p);
    }

    public interface IAccuracies
    {
        string Name { get; }
        double Accuracy(double[][] y, double[][] p);
    }

    public class MeanSquaredLoss : ILosses
    {
        public string Name => "MeanSquaredLoss";

        double floss(double y, double p) => (y - p) * (y - p) / 2;
        double fgrad(double y, double p) => p - y;

        public double Loss(double[][] y, double[][] p)
        {
            Utils.ApplyFunc(y, p, out double[][] c, floss);
            return c.SelectMany(i => i).Average();
        }

        public double[][] Grad(double[][] y, double[][] p)
        {
            Utils.ApplyFunc(y, p, out double[][] c, fgrad);
            return c;
        }
    }

    public class CrossEntropyLoss : ILosses
    {
        public string Name => "CrossEntropyLoss";

        double clamp(double p, double min, double max) => Math.Min(max, Math.Max(min, p));
        double floss(double y, double p)
        {
            var p0 = clamp(p, 1e-12, 1 - 1e-12);
            return -y * Math.Log(p0) - (1 - y) * Math.Log(1 - p0);
        }

        double fgrad(double y, double p)
        {
            var p0 = clamp(p, 1e-12, 1 - 1e-12);
            return -y / p0 + (1 - y) / (1 - p0);
        }

        public double[][] Grad(double[][] y, double[][] p)
        {
            Utils.ApplyFunc(y, p, out double[][] c, fgrad);
            return c;
        }

        public double Loss(double[][] y, double[][] p)
        {
            Utils.ApplyFunc(y, p, out double[][] c, floss);
            return c.SelectMany(i => i).Average();
        }
    }

    public class RoundAccuracy : IAccuracies
    {
        public string Name => "RoundAccuracy";

        double facc(double y, double p) => Math.Abs(Math.Round(y) - Math.Round(p)) < 1e-6 ? 1.0 : 0.0;

        public double Accuracy(double[][] y, double[][] p)
        {
            Utils.ApplyFunc(y, p, out double[][] c, facc);
            return c.Select(a => a.All(i => Math.Abs(i - 1.0) < 1e-6) ? 1.0 : 0.0).Average();
        }
    }

    public class ArgMaxAccuracy : IAccuracies
    {
        public string Name => "ArgMaxAccuracy";

        public double Accuracy(double[][] y, double[][] p)
        {
            return Enumerable.Range(0, y.Length).Select(i => Argmax(y[i]) == Argmax(p[i]) ? 1.0 : 0.0).Average();
        }

        public static int Argmax(double[] arr)
        {
            int bestIdx = 0;
            double bestValue = double.MinValue;
            for(int i = 0; i < arr.Length; ++i)
            {
                var v = arr[i];
                if (v > bestValue)
                {
                    bestValue = v;
                    bestIdx = i;
                }
            }

            return bestIdx;
        }
    }
}
