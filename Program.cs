using System;
using System.IO;

namespace Digits
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                if (!active.isrunning)
                {
                    active.program();
                }
            }
        }

    }
    class active
    {
        public static bool isrunning = false;
        static float avg = 1;
        static double avgerror = 0;
        static int batchsize = 1;
        public static void program()
        {
            isrunning = true;
            NN nn = new NN();
            //nn.initialize();
            //D.WriteWeightBias(nn);
            D.ReadWeightBias(nn);
            for (int i = 0; i < batchsize - 1; i++)
            {
                nn.backprop(Reader.ReadNextImage(), Reader.ReadNextLabel());
                nn.Descend();
            }
            double[,] image = Reader.ReadNextImage();
            int correct = Reader.ReadNextLabel();
            nn.backprop(image, correct);
            nn.Descend(batchsize);
            D.WriteWeightBias(nn);
            int guess = 0; double certainty = 0;
            //yeah, yeah, this SHOULD be a NN method, not public and used here but I'm tired right now
            for (int i = 0; i < 10; i++) { if (nn.Outputs[i].value > certainty) { certainty = nn.Outputs[i].value; guess = i; } }
            double error = 0;
            for (int i = 0; i < NN.Count; i++)
            {
                error += ((i == correct ? 1 : 0) - nn.Outputs[i].value) * ((i == correct ? 1 : 0) - nn.Outputs[i].value);
            }
            avgerror = (.99 * avgerror) + (.01 * error);
            avg = (float)(avg * .99) + ((guess == correct) ? (float).01 : 0);
            Console.WriteLine("Correct: " + correct + " Guess: " + guess + " Correct? " + (guess == correct ? "1 " : "0 ") + "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)
                + " % Correct: " + Math.Round(avg, 5).ToString().PadRight(7) + " Avg error: " + Math.Round(avgerror, 5) + " Avg gradient: " + Math.Round(nn.AvgGradient, 15));
            nn.Dispose();
            isrunning = false;
        }
    }

}
