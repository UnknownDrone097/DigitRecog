using System;
using System.IO;

namespace Digits
{
    class Program
    {
        static void Main(string[] args)
        {
            while (!active.finished)
            {
                if (!active.isrunning)
                {
                    active.isrunning = true;
                    //active.reset();
                    active.program();
                }
            }
            Console.WriteLine("Finished");
        }
    }
    class active
    {
        public static bool isrunning = false;
        public static bool finished = false;
        static double avg = 1;
        static double maxavg = 0;
        static double avgerror = 1;
        static int batchsize = 10;
        static double iterator = 1;
        public static void reset()
        {
            NN nn = new NN();
            nn.initialize();
            D.WriteWeightBias(nn);
        }
        public static void program()
        {
            NN nn = new NN();
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
                error += ((i == correct ? 1d : 0d) - nn.Outputs[i].value) * ((i == correct ? 1d : 0d) - nn.Outputs[i].value);
            }
            avgerror = ((iterator / (iterator + 1)) * avgerror) + ((1 / iterator) * error);
            avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0d);
            if (avgerror > maxavg && iterator > 300) { maxavg = avgerror; }
            if (avgerror > maxavg * 10 && iterator > 300) { finished = true; }
            Console.WriteLine("Correct: " + correct + " Guess: " + guess + " Correct? " + (guess == correct ? "1 " : "0 ") + "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)
                + " %Correct: " + Math.Round(avg, 5).ToString().PadRight(7) + " Avg error: " + Math.Round(avgerror, 5).ToString().PadRight(8) + " Avg gradient: " + Math.Round(nn.AvgGradient, 15));
            nn.Dispose();
            iterator++;
            if (iterator > 1000) { iterator = 100; }
            isrunning = false;
        }
    }

}
