using System;
using System.Collections.Generic;
using System.Text;

namespace Digits
{
    class NN : IDisposable
    {
        public static int Depth = 3;
        //Happens to work out that Count == OutputCount, but if this changes code will need to be refactored to account
        public static int Count = 10;
        public static int OutputCount = 10;
        public static int Resolution = 28;
        static int LearningRate = 1;
        static int CurrentIteration = 0;
        static int BatchSize = 100;
        public double AvgGradient = 0;
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
        public void Descend()
        {
            //Input layer
            for (int i = 0; i < Count; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    InputWeights[i, ii] -= LearningRate * InputWeightGradient[i, ii];
                    AvgGradient += LearningRate * InputWeightGradient[i, ii];
                    InputBiases[i, ii] -= LearningRate * InputBiasGradient[i, ii];
                }
            }
            //Hidden layers
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        HiddenWeights[i, ii, iii] -= LearningRate * HiddenWeightGradient[i, ii, iii];
                        AvgGradient += LearningRate * HiddenWeightGradient[i, ii, iii];
                        HiddenBiases[i, ii, iii] -= LearningRate * HiddenBiasGradient[i, ii, iii];
                    }
                }
            }
            AvgGradient /= HiddenWeightGradient.Length + InputWeightGradient.Length;
        }
        public void backprop(double[,] image, int correct)
        {
            //Run NN on image
            calculate(image);

            //Foreach weight/bias
            for (int l = Depth - 1; l >= 0; l--)
            {
                //If an input layer
                if (l == 0)
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
                            double zval = (InputWeights[k, j] * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + InputBiases[k, j];                        
                            InputWeightGradient[k, j] = Neurons[l, k].value * Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                            InputBiasGradient[k, j] = Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                        }                       
                    }
                    continue;
                }
                for (int k = 0; k < Count; k++)
                {
                    
                    //If an output layer
                    if (l == Depth - 1)
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            double zval = (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value) + HiddenBiases[l - 1, k, j];

                            int value = 0;
                            if (j == correct) { value = 1; }
                            //Formulas
                            HiddenWeightGradient[l - 1, k, j] = Neurons[l, k].value * Sigmoid.sigmoidderiv(zval) * 2 * (Outputs[j].value - value);
                            HiddenBiasGradient[l - 1, k, j] = Sigmoid.sigmoidderiv(zval) * 2 * (Outputs[j].value - value);
                        }
                        continue;
                    }
                    //If a hidden layer
                    for (int j = 0; j < Count; j++)
                    {
                        double upperlayerderiv = 0;
                        for (int i = 0; i < Count; i++)
                        {
                            upperlayerderiv += HiddenWeights[l, i, k] * Sigmoid.sigmoidderiv((HiddenWeights[l, i, k] * Neurons[l, i].value) + HiddenBiases[l, i, k]) * HiddenWeightGradient[l, i, k];
                        }
                        double zval = (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value) + HiddenBiases[l - 1, k, j];
                        HiddenWeightGradient[l - 1, k, j] = Neurons[l, k].value * Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                        HiddenBiasGradient[l - 1, k, j] = Sigmoid.sigmoidderiv(zval) * upperlayerderiv;
                    }
                }
            }
        }
        public void calculate(double[,] image)
        {
            for (int l = 0; l < Depth; l++)
            {
                for (int k = 0; k < Count; k++)
                {
                    //Calc input layer
                    if (l == 0)
                    {
                        for (int j = 0; j < (Resolution * Resolution); j++)
                        {
                            Neurons[0, k].value += (InputWeights[k, j] * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + InputBiases[k, j];
                        }
                    }
                    //Calc hidden layers
                    else
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            Neurons[l, k].value += (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value) + HiddenBiases[l - 1, k, j];
                        }
                    }
                }
            }
            //Normalize these values by layer
            for (int i = 0; i < Depth; i++)
            {
                double[] array = new double[Count];
                for (int ii = 0; ii < Count; ii++) { array[ii] = Neurons[i, ii].value; }
                Normalize(array);
                //This may be redundant b/c of pass by reference
                for (int ii = 0; ii < Count; ii++) { Neurons[i, ii].value = array[ii]; }
            }
        }
        public void Normalize(double[] array)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in array) { mean += d; }
            mean /= array.Length;
            //Calc std dev of data
            foreach (double d in array) { stddev += (d - mean) * (d - mean); }
            stddev /= Neurons.Length;
            stddev = Math.Sqrt(stddev);
            //Standardize each value with sigmoid of z-score
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = Sigmoid.sigmoid((array[i] - mean) / stddev);
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
                    //Normalize to [-5, 5]
                    InputWeights[i, ii] = (double)r.Next(-99, 99) / 20;
                    InputBiases[i, ii] = (double)r.Next(-99, 99) / 20;
                }
            }
            //Initialize hidden weights/biases
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        //Normalize to [-5, 5]
                        HiddenWeights[i, ii, iii] = (double)r.Next(-99, 99) / 15;
                        HiddenBiases[i, ii, iii] = (double)r.Next(-99, 99) / 15;
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

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        // ~NN()
        // {
        //   // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
        //   Dispose(false);
        // }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            // GC.SuppressFinalize(this);
        }
        #endregion
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
