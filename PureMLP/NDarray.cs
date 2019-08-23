using System;
using System.Collections.Generic;
using System.Linq;

namespace PureMLP
{
    public class NDarray
    {
        public NDarray(params int[] shape)
        {
            Shape = shape.ToArray();
            Count = ArrMul(Shape);
            Data = new double[Count];
        }

        public NDarray(double[] data, params int[] shape)
        {
            Shape = shape.ToArray();
            Count = ArrMul(Shape);
            Data = data.ToArray();
            if (Data.Length != Count)
                throw new Exception();
        }

        public NDarray(double[][] data)
        {
            Data = data.SelectMany(i => i).ToArray();
            Count = Data.Length;
            Shape = new int[] { data.Length, Count / data.Length };
        }

        public NDarray(NDarray nd)
        {
            Shape = nd.Shape.ToArray();
            Count = nd.Count;
            Data = nd.Data.ToArray();
        }

        public double[] Data;
        public int[] Shape;
        public int Count;

        public override string ToString()
        {
            var reshape = $"reshape({Shape.Glue(",")})";
            var ndarray = $"np.array([{Data.Glue(",")}], dtype=np.float64)";
            return $"{ndarray}.{reshape}";
        }

        public double[] GetAtIndex(int idx)
        {
            var s = ArrMul(Shape, 1);
            return Data.Skip(idx * s).Take(s).ToArray();
        }

        public NDarray Reshape(params int[] shape)
        {
            var nshape = PrepareReshape(Count, shape);
            return new NDarray(Data, nshape);
        }

        public NDarray Transpose(params int[] table)
        {
            if (table == null || table.Length == 0)
                table = PrepareTranspose(Shape.Length);

            if (table.Length != Shape.Length)
                throw new Exception();

            var strides = Shape2Strides(Shape);
            var nshape = DoTranspose(Shape, table);
            var nstrides = DoTranspose(strides, table);

            NDarray nd0 = new NDarray(nshape);
            for(int idx = 0; idx < Count; ++idx)
            {

                int idx1 = Int2IntIndex(idx, nshape, nstrides);
                nd0.Data[idx] = Data[idx1];
            }

            return nd0;
        }

        public NDarray T => Transpose();

        public void ApplyFuncInplace(Func<double, double> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(Data[idx]);
        }

        public void ApplyFuncInplace(Func<int, double, double> func)
        {
            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = func(idx, Data[idx]);
        }

        public void MulFbInplace(NDarray b, Func<double, double> func)
        {
            if (!Shape.SequenceEqual(b.Shape))
                throw new Exception();

            for (int idx = 0; idx < Count; ++idx)
                Data[idx] = Data[idx] * func(b.Data[idx]);
        }

        public NDarray ApplyFunc(Func<double, double> func)
        {
            NDarray nd = new NDarray(Shape);
            for (int idx = 0; idx < Count; ++idx)
                nd.Data[idx] = func(Data[idx]);

            return nd;
        }

        public NDarray MulFb(NDarray b, Func<double, double> func)
        {
            if (!Shape.SequenceEqual(b.Shape))
                throw new Exception();

            NDarray nd = new NDarray(Shape);
            for (int idx = 0; idx < Count; ++idx)
                nd.Data[idx] = Data[idx] * func(b.Data[idx]);

            return nd;
        }

        public NDarray SumAxis0()
        {
            var nshape = Shape.ToArray();
            nshape[0] = 1;
            var nd = new NDarray(nshape);

            for(int idx = 0; idx < nd.Count; ++idx)
            {
                double sum = 0;
                for(int k = 0; k < Shape[0]; ++k)
                    sum += Data[idx + k * nd.Count];

                nd.Data[idx] = sum;
            }

            return nd;
        }

        public (NDarray, NDarray) Split(int axis, int idx)
        {
            (var shape0, var shape1) = PrepareSplit(Shape, axis, idx);
            NDarray nd0 = new NDarray(shape0);
            NDarray nd1 = new NDarray(shape1);

            int[] indices = new int[Shape.Length];
            int[] strides = Shape2Strides(Shape);

            for (int idx0 = 0; idx0 < nd0.Count; ++idx0)
            {
                Int2ArrayIndex(idx0, nd0.Shape, indices);
                var idx2 = Array2IntIndex(indices, Shape, strides);
                nd0.Data[idx0] = Data[idx2];
            }

            for (int idx1 = 0; idx1 < nd1.Count; ++idx1)
            {
                Int2ArrayIndex(idx1, nd1.Shape, indices);
                indices[axis] += idx;
                var idx2 = Array2IntIndex(indices, Shape, strides);
                nd1.Data[idx1] = Data[idx2];
            }

            return (nd0, nd1);
        }

        public NDarray ReshapeTranspose(int[] shape, int[] table)
        {
            var nshape0 = PrepareReshape(Count, shape);
            if (table == null || table.Length == 0)
                table = PrepareTranspose(nshape0.Length);

            if (table.Length != nshape0.Length)
                throw new Exception();

            var strides = Shape2Strides(nshape0);
            var nshape = DoTranspose(nshape0, table);
            var nstrides = DoTranspose(strides, table);

            NDarray nd0 = new NDarray(nshape);
            for (int idx = 0; idx < Count; ++idx)
            {

                int idx1 = Int2IntIndex(idx, nshape, nstrides);
                nd0.Data[idx] = Data[idx1];
            }

            return nd0;
        }

        public NDarray TransposeReshape(int[] table,int[] shape)
        {
            if (table == null || table.Length == 0)
                table = PrepareTranspose(Shape.Length);

            if (table.Length != Shape.Length)
                throw new Exception();

            var strides = Shape2Strides(Shape);
            var nshape = DoTranspose(Shape, table);
            var nstrides = DoTranspose(strides, table);

            NDarray nd0 = new NDarray(nshape);
            for (int idx = 0; idx < Count; ++idx)
            {

                int idx1 = Int2IntIndex(idx, nshape, nstrides);
                nd0.Data[idx] = Data[idx1];
            }

            var nshape1 = PrepareReshape(Count, shape);
            nd0.Shape = nshape1;

            return nd0;
        }

        public static NDarray GemmABC(NDarray a, NDarray b, NDarray c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != wb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != hb)))
                throw new Exception();

            var nd = new NDarray(wa, hb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray GemmATBC(NDarray a, NDarray b, NDarray c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (ha != hb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != wb)))
                throw new Exception();

            var nd = new NDarray(wa, wb);
            for (int i = 0; i < wa; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < ha; ++k)
                        sum += a.Data[i * ha + k] * b.Data[j * ha + k];

                    nd.Data[i * wb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray GemmTABC(NDarray a, NDarray b, NDarray c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != wb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != hb)))
                throw new Exception();

            var nd = new NDarray(ha, hb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < hb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[k * hb + j];

                    nd.Data[i * hb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray GemmTATBC(NDarray a, NDarray b, NDarray c = null)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2 || (c != null && c.Shape.Length != 2))
                throw new Exception();

            (int wa, int ha) = (a.Shape[0], a.Shape[1]);
            (int wb, int hb) = (b.Shape[0], b.Shape[1]);

            if (wa != hb || (c != null && (c.Shape[0] != 1 || c.Shape[1] != wb)))
                throw new Exception();

            var nd = new NDarray(ha, wb);
            for (int i = 0; i < ha; ++i)
            {
                for (int j = 0; j < wb; ++j)
                {
                    double sum = c != null ? c.Data[j] : 0;
                    for (int k = 0; k < wa; ++k)
                        sum += a.Data[k * ha + i] * b.Data[j * wa + k];

                    nd.Data[i * wb + j] = sum;
                }
            }

            return nd;
        }

        public static NDarray ApplyFunc(NDarray a, NDarray b, Func<double, double, double> func)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new Exception();

            NDarray nd = new NDarray(a.Shape);
            for (int idx = 0; idx < a.Count; ++idx)
                nd.Data[idx] = func(a.Data[idx], b.Data[idx]);

            return nd;
        }


        public static List<(NDarray, NDarray)> BatchIterator(NDarray X, NDarray y, int batchsize, bool shuffle = true)
        {
            int dim0 = X.Shape[0];
            batchsize = dim0 < batchsize ? dim0 : batchsize;
            int nb = dim0 / batchsize;

            Queue<int> q = new Queue<int>(Enumerable.Range(0, dim0).OrderBy(c => shuffle ? Utils.Random.NextDouble() : 0));
            List<(NDarray, NDarray)> batch = new List<(NDarray, NDarray)>();
            var xshape = X.Shape.ToArray();
            var yshape = y.Shape.ToArray();
            xshape[0] = batchsize;
            yshape[0] = batchsize;

            int xs = ArrMul(xshape, 1);
            int ys = ArrMul(yshape, 1);

            for (int k = 0; k < nb; ++k)
            {
                var lx = new NDarray(xshape);
                var ly = new NDarray(yshape);
                for (int i = 0; i < batchsize; ++i)
                {
                    int idx = q.Dequeue();
                    X.GetAtIndex(idx).CopyTo(lx.Data, i * xs);
                    y.GetAtIndex(idx).CopyTo(ly.Data, i * ys);
                }
                batch.Add((lx, ly));
            }

            return batch;
        }


        public static int ArrMul(int[] shape, int start = 0) => shape.Skip(start).Aggregate(1, (a, i) => a * i);
        public static int[] Shape2Strides(int[] shape) => Enumerable.Range(0, shape.Length).Select(i => ArrMul(shape, i + 1)).ToArray();

        public static void Int2ArrayIndex(int idx, int[] shape, int[] indices)
        {

            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                indices[k] = idx % sk;
                idx = idx / sk;
            }
        }

        public static int Array2IntIndex(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                idx += v * strides[k];
            }

            return idx;
        }

        public static int Int2IntIndex(int idx0, int[] shape, int[] strides)
        {
            int idx1 = 0;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                idx1 += strides[k] * (idx0 % sk);
                idx0 = idx0 / sk;
            }

            return idx1;
        }

        public static int[] PrepareReshape(int dim0, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("Can only specify one unknown dimension");

            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"cannot reshape array of size {dim0} into shape ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank) => Enumerable.Range(0, rank).Reverse().ToArray();
        public static int[] DoTranspose(int[] arr, int[] table) => Enumerable.Range(0, arr.Length).Select(i => arr[table[i]]).ToArray();

        public static (int[], int[]) PrepareSplit(int[] shape, int axis, int idx)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("Bad Split axis");

            int dim = shape[axis];
            if (idx < 0 || idx >= dim)
                throw new ArgumentException("Bad Split index");

            int[] shape0 = shape.ToArray();
            int[] shape1 = shape.ToArray();
            shape0[axis] = idx;
            shape1[axis] -= idx;

            return (shape0, shape1);
        }
    }
}
