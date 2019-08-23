using System;
using System.Linq;
namespace PureMLP
{
    public interface IOptimizer
    {
        string Name { get; set; }
        IOptimizer Clone();
        void Update(NDarray w, NDarray grad);
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

        NDarray wUpdt;
        public void Update(NDarray w, NDarray grad)
        {
            if (wUpdt == null)
                wUpdt = new NDarray(w.Shape);

            for (int i = 0; i < wUpdt.Count; ++i)
                wUpdt.Data[i] = momentum * wUpdt.Data[i] + (1 - momentum) * grad.Data[i];

            for (int i = 0; i < w.Count; ++i)
                w.Data[i] = w.Data[i] - lr * wUpdt.Data[i];
        }
    }
}
