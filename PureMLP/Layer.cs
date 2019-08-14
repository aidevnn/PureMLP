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

        public double[][] weights, biases, lastX, lastXact, wTmp;

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
            weights = Utils.Matrix(InputNodes, OutputNodes);
            biases = Utils.Matrix(1, OutputNodes);
            wTmp = Utils.Matrix(InputNodes, OutputNodes);

            for (int i = 0; i < InputNodes; ++i)
                for (int j = 0; j < OutputNodes; ++j)
                    weights[i][j] = -lim + 2 * lim * Utils.Random.NextDouble();
        }

        public double[][] Forward(double[][] X)
        {
            lastX = X.Select(a => a.ToArray()).ToArray();
            Utils.GemmABC(X, weights, out double[][] res, biases[0]);
            lastXact = res.Select(a => a.ToArray()).ToArray();
            Utils.ApplyFunc(res, activation.Func);

            return res;
        }

        public double[][] Backward(double[][] accumGrad)
        {
            Utils.MulApplyFunc(accumGrad, lastXact, activation.Deriv);

            for (int i = 0; i < InputNodes; ++i)
                for (int j = 0; j < OutputNodes; ++j)
                    wTmp[i][j] = weights[i][j];

            if (IsTraining)
            {
                Utils.GemmTAB(lastX, accumGrad, out double[][] gw);
                wOptm.Update(weights, gw);

                Utils.SumAxis0(accumGrad, out double[][] gb);
                bOptm.Update(biases, gb);
            }

            Utils.GemmATB(accumGrad, wTmp, out double[][] res);
            return res;
        }
    }
}
