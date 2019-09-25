using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Digits
{
    class Reader
    {
        //Need to check for end of file
        private readonly string LabelPath = @"H:\Documents\\train-labels-idx1-ubyte";
        private readonly string ImagePath = @"H:\Documents\train-images-idx3-ubyte";
        int LabelOffset = 8;
        int ImageOffset = 16;
        int Resolution = 28;
        public int ReadNextLabel()
        {
            FileStream fs = File.OpenRead(LabelPath);
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
        public int[,] ReadNextImage()
        {
            //Read image
            FileStream fs = File.OpenRead(ImagePath);
            fs.Position = ImageOffset;
            byte[] b = new byte[Resolution * Resolution];
            try
            {
                fs.Read(b, 0, Resolution * Resolution);
            }
            catch { Console.WriteLine("Reset; ImageOffset = " + ImageOffset.ToString()); ImageOffset = 0; }
            int[] array = Array.ConvertAll(b, Convert.ToInt32);
            ImageOffset += Resolution * Resolution;
            fs.Close();
            //Convert to 2d array
            int[,] result = new int[Resolution, Resolution];
            for (int i = 0; i < Resolution; i++)
            {
                for (int ii = 0; ii < Resolution; ii++)
                {
                    result[i, ii] = array[(Resolution * i) + ii];
                }
            }
            fs.Close();
            return result;
        }
        public void PrintArray(int[,] a)
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
