using System;
using System.Collections.Generic;
using System.Linq;

namespace PureMLP
{
    public static class Utils
    {
        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string fmt = "{0}") => string.Join(sep, ts.Select(t => string.Format(fmt, t)));

        public static Random Random = new Random();


    }
}