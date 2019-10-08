using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Digits
{
    public static class Reader
    {
        private static string LabelPath = @"H:\Documents\train-labels-idx1-ubyte";
        private static string ImagePath = @"H:\Documents\train-images-idx3-ubyte";
        static int LabelOffset = 8;
        static int ImageOffset = 16;
        static int Resolution = 28;
        public static int ReadNextLabel()
        {
            FileStream fs = File.OpenRead(LabelPath);
            //Ensure Labeloffset is always under the file size
            if (!(LabelOffset < fs.Length)) { LabelOffset = 8; }

            fs.Position = LabelOffset;
            byte[] b = new byte[1];
            try
            {
                fs.Read(b, 0, 1);
            }
            catch { Console.WriteLine("Reset; ImageOffset = " + LabelOffset.ToString()); LabelOffset = 0; }
            int[] result = Array.ConvertAll(b, Convert.ToInt32);
            LabelOffset++;
            fs.Close();
            foreach (int i in result) { return i; }
            return -1;
        }
        public static double[,] ReadNextImage()
        {
            //Read image
            FileStream fs = File.OpenRead(ImagePath);
            //Ensure Labeloffset is always under the file size
            if (!(ImageOffset < fs.Length)) { ImageOffset = 16; }
            fs.Position = ImageOffset;
            byte[] b = new byte[Resolution * Resolution];
            try
            {
                fs.Read(b, 0, Resolution * Resolution);
            }
            catch { Console.WriteLine("Reset; ImageOffset = " + ImageOffset.ToString()); ImageOffset = 0; }
            int[] array = Array.ConvertAll(b, Convert.ToInt32);
            ImageOffset += Resolution * Resolution;
            //Convert to 2d array
            double[,] result = new double[Resolution, Resolution];
            //Convert array to doubles in result
            for (int i = 0; i < Resolution; i++)
            {
                for (int ii = 0; ii < Resolution; ii++)
                {
                    result[i, ii] = (double)array[(Resolution * i) + ii];
                }
            }
            ActivationFunctions.Normalize(result, Resolution, Resolution);

            fs.Close();
            return result;
        }
        public static void PrintArray(int[,] a)
        {
            for (int i = 0; i < a.Length; i++)
            {
                for (int ii = 0; ii < a.Length; ii++)
                {
                    Console.Write(a[i, ii].ToString().PadRight(5));
                }
                Console.WriteLine();
            }
        }
    }
}
