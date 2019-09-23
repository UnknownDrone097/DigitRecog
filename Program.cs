using System;
using System.IO;

namespace Digits
{
    class Program
    {
        static void Main(string[] args)
        {
            Reader r = new Reader();
            int[] num = r.ReadNextLabel();
            int[] array = r.ReadNextImage();
            foreach (int i in num) { Console.WriteLine(i); }
            r.PrintArray(array);
            num = r.ReadNextLabel();
            array = r.ReadNextImage();
            foreach (int i in num) { Console.WriteLine("\n" + i); }
            r.PrintArray(array);

            NeuralNet nn = new NeuralNet();
            
            Data.ReadNs(nn, true, @"H:\Documents\wets.txt", 28);
            Data.ReadNs(nn, false, @"H:\Documents\bias.txt", 28);
            /*
            nn.initNN();
            Data.WriteNs(nn, true, @"H:\Documents\wets.txt", 28);
            Data.WriteNs(nn, false, @"H:\Documents\bias.txt", 28);
            */
        }
    }

}
