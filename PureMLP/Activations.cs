using System;
using System.Linq;
namespace PureMLP
{
    public interface IActivation
    {
        string Name { get; }
        double Func(double X);
        double Deriv(double X);
    }

    public class SigmoidActivation : IActivation
    {
        public string Name => "Sigmoid";

        public double Deriv(double X) => Func(X) * (1.0 - Func(X));

        public double Func(double X) => 1.0 / (1.0 + Math.Exp(-X));
    }

    public class TanhActivation : IActivation
    {
        public string Name => "Tanh";

        public double Deriv(double X) => 1.0 - Math.Tanh(X) * Math.Tanh(X);

        public double Func(double X) => Math.Tanh(X);
    }

    public enum Activation { Tanh, Sigmoid } 

    public static class Activations
    {
        public static IActivation Get(Activation name)
        {
            if (name == Activation.Tanh)
                return new TanhActivation();

            return new SigmoidActivation();
        }
    }
}
