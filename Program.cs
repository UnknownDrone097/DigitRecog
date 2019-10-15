using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Digits
{
    class Program
    {
        public static bool TrainorTest = true;
        static void Main(string[] args)
        {
            //Active.reset();
            D.ReadWeightBias();
            if (TrainorTest)
            {
                while (!Active.finished)
                {
                    Active.Training();
                }
            }
            else { Active.Testing(); }
            
            //When things go wrong (or testing finishes), notify the user via console
            Console.WriteLine("Finished");
        }
    }
    class Active
    {
        public static bool isrunning = false;
        public static bool finished = false;
        static double SaveState = 100;
        static double avg = 0;
        static double maxavg = 0;
        static double avgerror = 0;
        static int batchsize = 5;
        static double iterator = 0;
        //Reset (re-initialize) weights and biases of the neural network
        public static void reset()
        {
            NN nn = new NN();
            nn.initialize();
            D.WriteWeightBias();
        }
        public static void Testing()
        {
            NN nn = new NN();
            while(iterator < 9000)
            {
                iterator++;
                nn.Calculate(Reader.ReadNextImage());
                int correct = Reader.ReadNextLabel();
                double certainty = -99d; int guess = -1;
                for (int i = 0; i < 10; i++) { if (nn.OutputValues[i] > certainty) { certainty = nn.OutputValues[i]; guess = i; } }
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0d);
                Console.WriteLine("Correct: " + correct + " Correct? " + (guess == correct ? "1 " : "0 ") + " %Correct: " + Math.Round(avg, 10).ToString().PadRight(12) + " Certainty " + Math.Round(certainty, 10));
                nn.Dispose();
            }           
        }
        public static void Training()
        {
            for (int j = 0; j < SaveState; j++)
            {
                List<NN> __ = new List<NN>();
                var tasks = new Task[batchsize];
                for (int ii = 0; ii < batchsize; ii++)
                {
                    //B/c ii may change and this code can't let that happen
                    int iterator = ii;
                    double[,] image = Reader.ReadNextImage();
                    int correct = Reader.ReadNextLabel();
                    __.Add(new NN());
                    tasks[iterator] = Task.Run(() => __[iterator].Backprop(image, correct));
                }
                Task.WaitAll(tasks);
                //Syncronously descend
                foreach (NN nn in __)
                {
                    nn.Descend();
                    nn.Dispose();
                }
                //Updating the weights with the avg gradients
                NN.Descend(batchsize);
                UserValidation();
            }
            //Save weights and biases
            D.WriteWeightBias();
        }
        public static void UserValidation()
        {
            NN nn = new NN(); double[,] image; int correct;
          
            //Some user validation code

            image = Reader.ReadNextImage();
            correct = Reader.ReadNextLabel();
            //Backprop again for averaging
            nn.Calculate(image);

            //Print out various things
            int guess = 0; double certainty = 0;
            //yeah, yeah, this SHOULD be a NN method, not public and used here but I'm tired right now
            //Is a calculation of r^2 error
            for (int i = 0; i < 10; i++) { if (nn.OutputValues[i] > certainty) { certainty = nn.OutputValues[i]; guess = i; } }
            //Calculate the moving average of the percentage of trials correct of those written to console
            double error = 0;
            for (int i = 0; i < NN.OutputCount; i++)
            {
                error += ((i == correct ? 1d : 0d) - nn.OutputValues[i]) * ((i == correct ? 1d : 0d) - nn.OutputValues[i]);
            }
            iterator++;
            avgerror = ((iterator / (iterator + 1)) * avgerror) + ((1 / iterator) * error);
            avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0d);

            //Some safety code which is currently disabled
            //if (avgerror > maxavg && iterator > 300) { maxavg = avgerror; }
            //if (avgerror > maxavg * 10 && iterator > 300) { finished = true; }

            //Print various things to the console for verification that things are nominal
            Console.WriteLine("Correct: " + correct + " Guess: " + guess + " Correct? " + (guess == correct ? "1 " : "0 ") + "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)
                + " %Correct: " + Math.Round(avg, 5).ToString().PadRight(7) + " Avg error: " + Math.Round(avgerror, 5).ToString().PadRight(8) + " Avg gradient: " + NN.AvgGradient, 15);

            //Dispose of the neural network (may not be necessary)
            nn.Dispose();
            //Reset the console data every few iterations to ensure up to date numbers
            if (iterator > 1000) { iterator = 100; NN.Epoch++; }
        }
    }

}
