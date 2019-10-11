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
                //Program is a singleton
                if (!active.isrunning)
                {
                    active.isrunning = true;
                    //active.reset();
                    active.program();
                }
            }
            //If things go wrong, notify the user via console
            Console.WriteLine("Finished");
        }
    }
    class active
    {
        public static bool isrunning = false;
        public static bool finished = false;
        static double SaveState = 100;
        static double avg = 0;
        static double maxavg = 0;
        static double avgerror = 0;
        static int batchsize = 16;
        static double iterator = 1;
        //Reset (re-initialize) weights and biases of the neural network
        public static void reset()
        {
            NN nn = new NN();
            nn.initialize();
            D.WriteWeightBias(nn);
        }
        public static void program()
        {
            NN nn = new NN(); double[,] image; int correct;
            D.ReadWeightBias(nn);
            //Run the program SaveState times, then save and print out values for that point
            for (int j = 0; j < SaveState; j++)
            {
                for (int i = 0; i < batchsize - 1; i++)
                {
                    //Backward propagation
                    nn.backprop(Reader.ReadNextImage(), Reader.ReadNextLabel());
                    //Storage of the gradient into an average
                    nn.Descend();
                }
                image = Reader.ReadNextImage();
                correct = Reader.ReadNextLabel();
                //Backprop again for averaging
                nn.backprop(image, correct);
                //Updating the weights with the avg gradients
                nn.Descend(batchsize);
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
                avgerror = ((iterator / (iterator + 1)) * avgerror) + ((1 / iterator) * error);
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0d);
                iterator++;
                //Some safety code which is currently disabled
                //if (avgerror > maxavg && iterator > 300) { maxavg = avgerror; }
                //if (avgerror > maxavg * 10 && iterator > 300) { finished = true; }

                //Print various things to the console for verification that things are nominal
                Console.WriteLine("Correct: " + correct + " Guess: " + guess + " Correct? " + (guess == correct ? "1 " : "0 ") + "Certainty: " + Math.Round(certainty, 5).ToString().PadRight(7)
                    + " %Correct: " + Math.Round(avg, 5).ToString().PadRight(7) + " Avg error: " + Math.Round(avgerror, 5).ToString().PadRight(8) + " Avg gradient: " + nn.AvgGradient, 15);
            }
            //Save weights and biases
            D.WriteWeightBias(nn);
            //Dispose of the neural network (may not be necessary)
            nn.Dispose();
            //Reset the console data every few iterations to ensure up to date numbers
            if (iterator > 1000) { iterator = 100; }
            isrunning = false;
        }
    }

}
