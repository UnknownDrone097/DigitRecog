using System;
using System.Collections.Generic;
using System.Text;

namespace Digits
{
    class NN
    {
        public static int Depth = 3;
        //Happens to work out that Count == OutputCount, but if this changes code will need to be refactored to account
        public static int Count = 10;
        public static int OutputCount = 10;
        public static int Resolution = 28;
        //Weights/biases
        public double[,] InputWeights = new double[Count, Resolution * Resolution];
        public double[,] InputBiases = new double[Count, Resolution * Resolution];
        public double[,,] HiddenWeights = new double[Depth - 1, Count, Count];
        public double[,,] HiddenBiases = new double[Depth - 1, Count, Count];
        //Gradients
        public double[,] InputWeightGradient = new double[Count, Resolution * Resolution];
        public double[,] InputBiasGradient = new double[Count, Resolution * Resolution];
        public double[,,] HiddenWeightGradient = new double[Depth - 1, Count, Count];
        public double[,,] HiddenBiasGradient = new double[Depth - 1, Count, Count];

        public N[] Outputs = new N[OutputCount];
        //Layer, n
        N[,] Neurons = new N[Depth, Count];
        public NN()
        {
            //Initialize neurons
            for (int i = 0; i < Depth; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    N n = new N();
                    Neurons[i, ii] = n;
                    if (i == Depth - 1) { Outputs[ii] = n; }
                }
            }
        }
        public void stochasticdescent()
        {
            //Input layer
            for (int i = 0; i < Count; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    InputWeights[i, ii] -= InputWeightGradient[i, ii];
                    InputBiases[i, ii] -= InputBiasGradient[i, ii];
                }
            }
            //Hidden layers
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        HiddenWeights[i, ii, iii] -= HiddenWeightGradient[i, ii, iii];
                        HiddenBiases[i, ii, iii] -= HiddenBiasGradient[i, ii, iii];
                    }
                }
            }
        }
        public void backprop(int[,] image, int correct)
        {
            //Run NN on image
            calculate(image);

            //Foreach weight/bias
            for (int l = Depth - 2; l > -1; l--)
            {
                //If an input layer
                if (l == -1)
                {
                    for (int k = 0; k < Count; k++)
                    {
                        for (int j = 0; j < Resolution * Resolution; j++)
                        {
                            double upperlayerderiv = 0;
                            for (int i = 0; i < Count; i++)
                            {
                                upperlayerderiv += HiddenWeights[0, i, k] * Sigmoid.sigmoidderiv((InputWeights[i, k] * Neurons[1, i].value) + InputBiases[i, k]) * HiddenWeightGradient[0, i, k];
                            }
                            double zval = (InputWeights[k, j] * image[k, j]) + InputBiases[k, j];
                            InputWeightGradient[k, j] = image[k, j] * Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                            InputBiasGradient[k, j] = Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                        }
                    }
                    continue;
                }
                for (int k = 0; k < Count - 1; k++)
                {

                    //If an output layer
                    if (l == Depth - 2)
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            double zval = (HiddenWeights[l, k, j] * Neurons[l - 1, j].value) + HiddenBiases[l, k, j];

                            int value = 0;
                            if (j == correct) { value = 1; }
                            //Formulas
                            HiddenWeightGradient[l, k, j] = Neurons[l - 1, k].value * Sigmoid.sigmoidderiv(zval) * 2 * (Outputs[j].value - value);
                            HiddenBiasGradient[l, k, j] = Sigmoid.sigmoidderiv(zval) * 2 * (Outputs[j].value - value);
                        }
                        continue;
                    }
                    //If a hidden layer
                    for (int j = 0; j < Count; j++)
                    {
                        double upperlayerderiv = 0;
                        for (int i = 0; i < Count; i++)
                        {
                            upperlayerderiv += HiddenWeights[l + 1, i, k] * Sigmoid.sigmoidderiv((HiddenWeights[l + 1, i, k] * Neurons[l + 1, i].value) + HiddenBiases[l + 1, i, k]) * HiddenWeightGradient[l + 1, i, k];
                        }
                        double zval = (HiddenWeights[l, k, j] * image[k, j]) + HiddenBiases[l, k, j];
                        HiddenWeightGradient[l, k, j] = image[k, j] * Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                        HiddenBiasGradient[l, k, j] = Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                    }
                }
            }
        }
        public void calculate(int[,] image)
        {
            for (int l = 0; l < Depth - 1; l++)
            {
                for (int k = 0; k < Count; k++)
                {
                    //Calc input layer
                    if (l == 0)
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            for (int jj = 0; jj < (Resolution * Resolution); jj++)
                            {
                                Neurons[0, k].value += (InputWeights[j, jj] * image[jj / Resolution, jj - ((jj / Resolution) * Resolution)]) + InputBiases[j, jj];                               
                            }                           
                        }                     
                    }
                    //Calc hidden layers
                    else
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            Neurons[l, k].value += (HiddenWeights[l, k, j] * Neurons[l - 1, j].value) + HiddenBiases[l, k, j];                            
                        }
                    }
                    Neurons[0, k].value = Sigmoid.sigmoid(Neurons[0, k].value);
                }
            }
        }
        public void initialize()
        {
            Random r = new Random();
            //Initialize input weights/biases
            for (int i = 0; i < Count; i++)
            {
                for (int ii = 0; ii < (Resolution * Resolution); ii++)
                {
                    InputWeights[i, ii] = (double)r.Next(-99, 99)/10000;
                    InputBiases[i, ii] = (double)r.Next(-99, 99)/10000;
                }
            }
            //Initialize hidden weights/biases
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        HiddenWeights[i, ii, iii] = (double)r.Next(-99, 99)/10000;
                        HiddenBiases[i, ii, iii] = (double)r.Next(-99, 99)/10000;
                    }
                }
            }
            //Initialize neurons
            for (int i = 0; i < Depth; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    N n = new N();
                    Neurons[i, ii] = n;
                    if (i == Depth - 1) { Outputs[ii] = n; }
                }
            }
        }
    }
    class N
    {
        public double value { get; set; }
    }
    class Sigmoid
    {
        public static double sigmoid(double number)
        {
            return 1 / (1 + Math.Pow(Math.E, -number));
        }
        public static double sigmoidderiv(double number)
        {
            return (sigmoid(number) * (1 - sigmoid(number)));
        }
    }
}
