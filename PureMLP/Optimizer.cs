using System;
using System.Linq;
namespace PureMLP
{
    public interface IOptimizer
    {
        string Name { get; set; }
        IOptimizer Clone();
        void Update(double[][] w, double[][] grad);
    }

    public class SGD : IOptimizer
    {
        public string Name { get; set; }
        public SGD(double lr = 0.01, double momentum = 0.0)
        {
            this.lr = lr;
            this.momentum = momentum;
            Name = $"SGD[lr:{lr} momentum:{momentum}]";
        }

        readonly double lr, momentum;

        public IOptimizer Clone() => new SGD(lr, momentum);

        double[][] wUpdt;
        public void Update(double[][] w, double[][] grad)
        {
            if (wUpdt == null)
                wUpdt = w.Select(i => new double[i.Length]).ToArray();

            for (int i = 0; i < wUpdt.Length; ++i)
            {
                for (int j = 0; j < wUpdt[i].Length; ++j)
                    wUpdt[i][j] = momentum * wUpdt[i][j] + (1 - momentum) * grad[i][j];
            }

            for (int i = 0; i < w.Length; ++i)
            {
                for (int j = 0; j < w[i].Length; ++j)
                    w[i][j] = w[i][j] - lr * wUpdt[i][j];
            }
        }
    }
}
