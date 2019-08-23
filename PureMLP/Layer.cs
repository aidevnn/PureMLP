using System;
using System.Linq;
namespace PureMLP
{
    public class Layer
    {
        public int InputNodes { get; set; }
        public int OutputNodes { get; set; }

        public bool IsTraining { get; set; }

        public int Params => (InputNodes + 1) * OutputNodes;

        public NDarray weights, biases, lastX, lastXact, wTmp;

        public IOptimizer wOptm, bOptm;
        public IActivation activation;

        public Layer(int outNodes, IActivation activation)
        {
            OutputNodes = outNodes;
            this.activation = activation;
        }

        public Layer(int outNodes, int inNodes, IActivation activation)
        {
            InputNodes = inNodes;
            OutputNodes = outNodes;
            this.activation = activation;
        }

        public void SetInputNodes(int nodes) => InputNodes = nodes;

        public void Initialize(IOptimizer optimizer)
        {
            wOptm = optimizer.Clone();
            bOptm = optimizer.Clone();

            double lim = 3.0 / Math.Sqrt(InputNodes);
            weights = new NDarray(InputNodes, OutputNodes);
            biases = new NDarray(1, OutputNodes);
            wTmp = new NDarray(InputNodes, OutputNodes);

            for (int i = 0; i < weights.Count; ++i)
                weights.Data[i] = -lim + 2 * lim * Utils.Random.NextDouble();
        }

        public NDarray Forward(NDarray X)
        {
            lastX = new NDarray(X);
            var res = NDarray.GemmABC(X, weights, biases);
            lastXact = new NDarray(res);
            res.ApplyFuncInplace(activation.Func);

            return res;
        }

        public NDarray Backward(NDarray accumGrad)
        {
            accumGrad.MulFbInplace(lastXact, activation.Deriv);

            for (int i = 0; i < wTmp.Count; ++i)
                wTmp.Data[i] = weights.Data[i];

            if (IsTraining)
            {
                var gw = NDarray.GemmTABC(lastX, accumGrad);
                wOptm.Update(weights, gw);

                var gb = accumGrad.SumAxis0();
                bOptm.Update(biases, gb);
            }

            return NDarray.GemmATBC(accumGrad, wTmp);
        }
    }
}
