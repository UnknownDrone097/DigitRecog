using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Digits
{
    public static class Reader
    {
        private static string LabelPath = @"C:\Users\gwflu\Desktop\Test\train-labels-idx1-ubyte\train-labels.idx1-ubyte";
        private static string ImagePath = @"C:\Users\gwflu\Desktop\Test\train-images-idx3-ubyte\train-images.idx3-ubyte";
        static int LabelOffset = 8;
        static int ImageOffset = 16;
        static int Resolution = 28;
        public static int ReadNextLabel()
        {
            FileStream fs = File.OpenRead(LabelPath);
            //Reset parameters and decrement NN hyperparameters upon new epoch
            if (!(LabelOffset < fs.Length)) { LabelOffset = 8; ImageOffset = 16; NN.LearningRate *= .6666; NN.Momentum *= .6666; }

            fs.Position = LabelOffset;
            byte[] b = new byte[1];
            try
            {
                fs.Read(b, 0, 1);
            }
            catch (Exception ex) { Console.WriteLine("Reader exception: " + ex.ToString()); Console.ReadLine(); }
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
            //Reset parameters and decrement NN hyperparameters upon new epoch
            if (!(ImageOffset < fs.Length)) { ImageOffset = 16; LabelOffset = 8; NN.LearningRate *= .6666; NN.Momentum *= .6666; }
            fs.Position = ImageOffset;
            byte[] b = new byte[Resolution * Resolution];
            try
            {
                fs.Read(b, 0, Resolution * Resolution);
            }
            catch (Exception ex) { Console.WriteLine("Reader exception: " + ex.ToString()); Console.ReadLine(); }
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
