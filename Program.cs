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
        static double SaveState = 100;
        static double avg = 1;
        static double maxavg = 0;
        static double avgerror = 1;
        static int batchsize = 5;
        static double iterator = 1;
        public static void reset()
        {
            NN nn = new NN();
            nn.initialize();
            D.WriteWeightBias(nn);
        }
        public static void program()
        {
            NN nn = new NN(); double[,] image; int correct = -1;
            //Run the program SaveState times, then save and print out values for that point
            for (int j = 0; j < SaveState; j++)
            {
                D.ReadWeightBias(nn);
                for (int i = 0; i < batchsize - 1; i++)
                {
                    nn.backprop(Reader.ReadNextImage(), Reader.ReadNextLabel());
                    nn.Descend();
                }
                image = Reader.ReadNextImage();
                correct = Reader.ReadNextLabel();
                nn.backprop(image, correct);
                nn.Descend(batchsize);
                //Print out various things
                int guess = 0; double certainty = 0;
                //yeah, yeah, this SHOULD be a NN method, not public and used here but I'm tired right now
                for (int i = 0; i < 10; i++) { if (nn.OutputValues[i] > certainty) { certainty = nn.OutputValues[i]; guess = i; } }
                double error = 0;
                for (int i = 0; i < NN.OutputCount; i++)
                {
                    error += ((i == correct ? 1d : 0d) - nn.OutputValues[i]) * ((i == correct ? 1d : 0d) - nn.OutputValues[i]);
                }
                avgerror = ((iterator / (iterator + 1)) * avgerror) + ((1 / iterator) * error);
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0d);
                iterator++;
                //Some safety code which is currently disabled
                //if (avgerror > maxavg && iterator > 300) { maxavg = avgerror; }
                //if (avgerror > maxavg * 10 && iterator > 300) { finished = true; }
                Console.WriteLine("Correct: " + correct + " Guess: " + guess + " Correct? " + (guess == correct ? "1 " : "0 ") + "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)
                    + " %Correct: " + Math.Round(avg, 5).ToString().PadRight(7) + " Avg error: " + Math.Round(avgerror, 5).ToString().PadRight(8) + " Avg gradient: " + nn.AvgGradient, 15);
            }
            //Save weights and biases
            D.WriteWeightBias(nn);
            nn.Dispose();
            if (iterator > 1000) { iterator = 100; }
            isrunning = false;
        }
    }

}
