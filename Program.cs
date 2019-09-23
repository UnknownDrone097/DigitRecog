using System;

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
        }
    }

}
