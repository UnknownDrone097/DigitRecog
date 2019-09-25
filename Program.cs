using System;
using System.IO;

namespace Digits
{
    class Program
    {
        static void Main(string[] args)
        {
            NN nn = new NN();
            //nn.initialize();
            //D.WriteWeightBias(nn);
            D.ReadWeightBias(nn);
            Reader r = new Reader();
            while (true) { program(nn, r); }                 
        }
        static void program(NN nn, Reader r)
        {           
            int[,] image = r.ReadNextImage();
            int correct = r.ReadNextLabel();
            nn.backprop(image , correct);
            nn.stochasticdescent();
            D.WriteWeightBias(nn);
            //Testing to ensure full copy
            D.ReadWeightBias(nn);
            int guess = 0; double certainty = 0;
            //yeah, yeah, this SHOULD be a NN method, not public and used here but I'm tired right now
            for (int i = 0; i < 10; i++) { if (nn.Outputs[i].value > certainty) { certainty = nn.Outputs[i].value; guess = i; } }
            Console.WriteLine("Correct: " + correct.ToString() + " Guess: " + guess.ToString() + " Certainty: " + certainty);
        }
    }

}
