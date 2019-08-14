using System;
using System.Collections.Generic;
using System.Linq;

namespace PureMLP
{
    public static class Utils
    {
        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string fmt = "{0}") => string.Join(sep, ts.Select(t => string.Format(fmt, t)));

        public static Random Random = new Random();

        public static double[][] Matrix(int w, int h) => Enumerable.Range(0, w).Select(i => new double[h]).ToArray();

        public static (int, int) Dim(double[][] A) => (A.Length, A[0].Length);

        public static void ApplyFunc(double[][] A, Func<double, double> func)
        {
            (int wa, int ha) = Dim(A);
            for (int i = 0; i < wa; ++i)
                for (int j = 0; j < ha; ++j)
                    A[i][j] = func(A[i][j]);
        }

        public static void ApplyFunc(double[][] A, double[][] B, out double[][] C, Func<double, double, double> func)
        {
            (int wa, int ha) = Dim(A);
            C = Matrix(wa, ha);
            for (int i = 0; i < wa; ++i)
                for (int j = 0; j < ha; ++j)
                    C[i][j] = func(A[i][j], B[i][j]);
        }

        public static void SumAxis0(double[][] A, out double[][] R)
        {
            (int wa, int ha) = Dim(A);
            R = Matrix(1, ha);
            for (int j = 0; j < ha; ++j)
            {
                double sum = 0;
                for (int i = 0; i < wa; ++i)
                    sum += A[i][j];

                R[0][j] = sum;
            }
        }

        public static void MulApplyFunc(double[][] A, double[][] B, Func<double, double> func)
        {
            (int wa, int ha) = Dim(A);
            for (int i = 0; i < wa; ++i)
                for (int j = 0; j < ha; ++j)
                    A[i][j] = A[i][j] * func(B[i][j]);

        }

        public static void GemmAB(double[][] A, double[][] B, out double[][] R) => GemmABC(A, B, out R, null);
        public static void GemmABC(double[][] A, double[][] B, out double[][] R, double[] C = null)
        {
            (int wa, int ha) = Dim(A);
            (int wb, int hb) = Dim(B);
            R = Matrix(wa, hb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = C != null ? C[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += A[i][k] * B[k][j];

                    R[i][j] = sum;
                }
            }
        }

        public static void GemmATB(double[][] A, double[][] B, out double[][] R) => GemmATBC(A, B, out R, null);
        public static void GemmATBC(double[][] A, double[][] B, out double[][] R, double[] C)
        {
            (int wa, int ha) = Dim(A);
            (int wb, int hb) = Dim(B);
            R = Matrix(wa, wb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = C != null ? C[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += A[i][k] * B[j][k];

                    R[i][j] = sum;
                }
            }
        }

        public static void GemmTAB(double[][] A, double[][] B, out double[][] R) => GemmTABC(A, B, out R, null);
        public static void GemmTABC(double[][] A, double[][] B, out double[][] R, double[] C)
        {
            (int wa, int ha) = Dim(A);
            (int wb, int hb) = Dim(B);
            R = Matrix(ha, hb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = C != null ? C[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += A[k][i] * B[k][j];

                    R[i][j] = sum;
                }
            }
        }

        public static void GemmTATB(double[][] A, double[][] B, out double[][] R) => GemmTATBC(A, B, out R, null);
        public static void GemmTATBC(double[][] A, double[][] B, out double[][] R, double[] C)
        {
            (int wa, int ha) = Dim(A);
            (int wb, int hb) = Dim(B);
            R = Matrix(ha, wb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = C != null ? C[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += A[k][i] * B[j][k];

                    R[i][j] = sum;
                }
            }
        }

        public static List<(double[][], double[][])> BatchIterator(double[][] X, double[][] y, int batchsize, bool shuffle)
        {
            int dim0 = X.Length;
            batchsize = dim0 < batchsize ? dim0 : batchsize;
            int nb = dim0 / batchsize;

            Queue<int> q = new Queue<int>(Enumerable.Range(0, dim0).OrderBy(c => shuffle ? Random.NextDouble() : 0));
            List<(double[][], double[][])> batch = new List<(double[][], double[][])>();
            for (int k = 0; k < nb; ++k)
            {
                List<double[]> lx = new List<double[]>();
                List<double[]> ly = new List<double[]>();
                for (int i = 0; i < batchsize; ++i)
                {
                    int idx = q.Dequeue();
                    lx.Add(X[idx].ToArray());
                    ly.Add(y[idx].ToArray());
                }
                batch.Add((lx.ToArray(), ly.ToArray()));
            }

            return batch;
        }

        public static (double[][], double[][]) Split(double[][] data, int axis, int idx)
        {
            double[][] left, right;
            int dim0 = data.Length, dim1 = data[0].Length;
            int ldim0 = 0, ldim1 = 0, rdim0 = 0, rdim1 = 0;
            if (axis == 0)
            {
                ldim0 = idx;
                rdim0 = dim0 - idx;
                ldim1 = rdim1 = dim1;
            }
            else
            {
                ldim0 = rdim0 = dim0;
                ldim1 = idx;
                rdim1 = dim1 - idx;
            }

            left = Matrix(ldim0, ldim1);
            right = Matrix(rdim0, rdim1);

            for(int i = 0; i < dim0; ++i)
            {
                for(int j = 0; j < dim1; ++j)
                {
                    var v = data[i][j];
                    if (i < ldim0)
                    {
                        if (j < ldim1) left[i][j] = v;
                        else right[i][j - ldim1] = v;
                    }
                    else
                        right[i - ldim0][j] = v;
                }
            }


            return (left, right);
        }
    }
}