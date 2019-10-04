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
        static double LearningRate = 0.00146;
        public double AvgGradient = 0;
        //Overall gradients
        public double[,] AvgInputWeightGradient = new double[Count, Resolution * Resolution];
        public double[,,] AvgHiddenWeightGradient = new double[Depth - 1, Count, Count];
        //For the biases
        public double[,] AvgErrorSignal = new double[Depth - 1, Count];
        //Error signal
        public double[,] ErrorSignals = new double[Depth, Count];
        //Weights/biases
        public double[,] InputWeights = new double[Count, Resolution * Resolution];
        public double[,,] HiddenWeights = new double[Depth - 1, Count, Count];
        public double[,] Biases = new double[Depth - 1, Count];
        //Gradients
        public double[,] InputWeightGradient = new double[Count, Resolution * Resolution];
        public double[,,] HiddenWeightGradient = new double[Depth - 1, Count, Count];

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
        //Batch descent
        public void Descend(int batchsize)
        {
            //Input layer
            for (int i = 0; i < Count; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    InputWeights[i, ii] -= LearningRate * InputWeightGradient[i, ii] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * InputWeightGradient[i, ii] * (-2 / (double)batchsize);
                }
            }
            //Hidden layers
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        HiddenWeights[i, ii, iii] -= LearningRate * HiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                        AvgGradient -= LearningRate * HiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                    }
                    Biases[i, ii] -= AvgErrorSignal[i, ii] * (-2 / (double)batchsize);
                }
            }
            AvgGradient /= HiddenWeightGradient.Length + InputWeightGradient.Length;
        }
        public void Descend()
        {
            //Input layer
            for (int i = 0; i < Count; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    AvgInputWeightGradient[i, ii] += InputWeightGradient[i, ii];
                }
            }
            //Hidden layers
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        AvgHiddenWeightGradient[i, ii, iii] += HiddenWeightGradient[i, ii, iii];
                    }
                    AvgErrorSignal[i, ii] += ErrorSignals[i, ii];
                }
            }
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
                                upperlayerderiv += HiddenWeights[0, i, k] * Sigmoid.sigmoidderiv((InputWeights[i, k] * Neurons[1, i].value) + Biases[l + 1, k]) * ErrorSignals[l + 1, k];
                            }
                            ErrorSignals[l, k] = upperlayerderiv;
                            double zval = (InputWeights[k, j] * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + Biases[l, k];

                            InputWeightGradient[k, j] = image[j / Resolution, j - ((j / Resolution) * Resolution)] * Sigmoid.sigmoidderiv(zval) * ErrorSignals[l, k];
                        }
                    }
                    continue;
                }
                for (int k = 0; k < Count; k++)
                {
                    //If an output layer
                    if (l == Depth - 1)
                    {
                        double value = 0;
                        if (k == correct) { value = 1d; }
                        ErrorSignals[l, k] = 2 * (value - Outputs[k].value);
                        for (int j = 0; j < Count; j++)
                        {
                            double zval = (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value);
                            //Formulas
                            HiddenWeightGradient[l - 1, k, j] = Neurons[l - 1, j].value * Sigmoid.sigmoidderiv(zval) * ErrorSignals[l, k];
                        }
                        continue;
                    }
                    //If a hidden layer
                    for (int j = 0; j < Count; j++)
                    {
                        double upperlayerderiv = 0;
                        for (int i = 0; i < Count; i++)
                        {
                            upperlayerderiv += HiddenWeights[l, i, k] * Sigmoid.sigmoidderiv(HiddenWeights[l, i, k] * Neurons[l, i].value) * ErrorSignals[l + 1, k];
                        }
                        ErrorSignals[l, k] = upperlayerderiv;
                        double zval = (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value) + Biases[l, k];
                        HiddenWeightGradient[l - 1, k, j] = Neurons[l - 1, j].value * Sigmoid.sigmoidderiv(zval) * ErrorSignals[l, k];
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
                            Neurons[0, k].value += (InputWeights[k, j] * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + Biases[l, k];
                        }
                    }
                    //Calc hidden layers
                    else
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            double bias;
                            if (l == Depth - 1) { bias = 0; } else { bias = Biases[l, k]; }
                            Neurons[l, k].value += (HiddenWeights[l - 1, k, j] * Neurons[l - 1, j].value) + bias;
                        }
                    }
                }
            }
            /*
            for (int i = 0; i < Depth; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    Neurons[i, ii].value = Sigmoid.sigmoid(Neurons[i, ii].value);
                }
            }
            */
            //Normalize these values by layer
            for (int i = 0; i < Depth; i++)
            {
                
                double[] array = new double[Count];
                for (int ii = 0; ii < Count; ii++) { array[ii] = Neurons[i, ii].value; }
                Normalize(array);
                //This may be redundant b/c of pass by reference
                for (int ii = 0; ii < Count; ii++) { Neurons[i, ii].value = Sigmoid.sigmoid(array[ii]); }
                
                //Sigmoid
                //for (int ii = 0; ii < Count; ii++) { Neurons[i, ii].value = Sigmoid.sigmoid(Neurons[i, ii].value); }
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
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
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
                    InputWeights[i, ii] = r.NextDouble() * Math.Sqrt(2 / (double)(Resolution * Resolution));
                }
            }
            //Initialize hidden weights and all biases
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    Biases[i, ii] = r.NextDouble() * Math.Sqrt(2 / NN.Count);
                    for (int iii = 0; iii < Count; iii++)
                    {
                        //Normalize to [-5, 5]
                        HiddenWeights[i, ii, iii] = r.NextDouble() * Math.Sqrt(2 / NN.Count);
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
