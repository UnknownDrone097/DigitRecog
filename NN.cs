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
        public double AvgGradient = 0;
        //Hyperparameters
        static double LearningRate = 0.00008;
        static double DropoutRate = .2;
        //Overall gradients
        public double[,] AvgInputWeightGradient = new double[Count, Resolution * Resolution];
        public double[,,] AvgHiddenWeightGradient = new double[Depth - 1, Count, Count];
        //For the biases
        public double[,] AvgBiasGradient = new double[Depth - 1, Count];
        //Error signal
        public double[,] ErrorSignals = new double[Depth, Count];
        //Weights/biases
        public double[,] InputWeights = new double[Count, Resolution * Resolution];
        public double[,,] HiddenWeights = new double[Depth - 1, Count, Count];
        public double[,] Biases = new double[Depth - 1, Count];
        public double[,] Zvals = new double[Depth, Count];
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
                    InputWeights[i, ii] -= LearningRate * AvgInputWeightGradient[i, ii] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * AvgInputWeightGradient[i, ii] * (-2 / (double)batchsize);
                }
            }
            //Hidden layers
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    for (int iii = 0; iii < Count; iii++)
                    {
                        HiddenWeights[i, ii, iii] -= LearningRate * AvgHiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                        AvgGradient -= LearningRate * AvgHiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                    }
                    Biases[i, ii] -= LearningRate * AvgBiasGradient[i, ii] * (-2 / (double)batchsize);
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
                    AvgBiasGradient[i, ii] += ErrorSignals[i, ii] * ActivationFunctions.Sigmoid(Zvals[i, ii]);
                }
            }
        }
        public void backprop(double[,] image, int correct)
        {
            //Run NN on image
            calculate(image);
            //Reset temp gradients
            ErrorSignals = new double[Depth, Count];
            HiddenWeightGradient = new double[Depth - 1, Count, Count];
            InputWeightGradient = new double[Count, Resolution * Resolution];

            //Foreach layer
            for (int l = Depth - 1; l >= 0; l--)
            {
                //Calculate error signals
                //Foreach ending neuron
                for (int k = 0; k < Count; k++)
                {
                    double upperlayerderiv = 0;
                    //Input neurons
                    if (l == 0)
                    {
                        //Foreach starting neuron
                        for (int j = 0; j < Count; j++)
                        {
                            //How does this equal zero?
                            upperlayerderiv += HiddenWeights[1, j, k] * ActivationFunctions.Sigmoid(Zvals[l + 1, j]) * ErrorSignals[1, j];
                        }
                    }
                    //Hidden neurons
                    if (l != 0 && l < Depth - 1)
                    {
                        //Foreach starting neuron
                        for (int j = 0; j < Count; j++)
                        {
                            //Hiddenweights uses l because the formula's l + 1 is l due to a lack of input layer in this array
                            upperlayerderiv += HiddenWeights[l, j, k] * ActivationFunctions.Sigmoid(Zvals[l + 1, j]) * ErrorSignals[l + 1, j];
                        }
                    }
                    //Output neurons
                    if (l == Depth - 1)
                    {
                        upperlayerderiv = 2d * ((k == correct ? 1d : 0d) - Outputs[k].value);
                    }
                    ErrorSignals[l, k] = upperlayerderiv;
                }

                //If an input layer
                if (l == 0)
                {
                    
                    //foreach starting neuron
                    for (int k = 0; k < Count; k++)
                    {
                        //foreach ending neuron
                        for (int j = 0; j < Resolution * Resolution; j++)
                        {
                            InputWeightGradient[k, j] = image[j / Resolution, j - ((j / Resolution) * Resolution)] * ActivationFunctions.Sigmoid(Zvals[l, k]) * ErrorSignals[l, k];
                        }
                    }
                    continue;
                }
                //foreach starting neuron
                for (int k = 0; k < Count; k++)
                {
                    //If an output layer
                    //Foreach ending neuron
                    if (l == Depth - 1)
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            HiddenWeightGradient[l - 1, k, j] = Neurons[l - 1, j].value * ActivationFunctions.Sigmoid(Zvals[l, k]) * ErrorSignals[l, k];
                        }
                        continue;
                    }
                    //If hidden layer
                    //Foreach ending neuron
                    for (int j = 0; j < Count; j++)
                    {
                        HiddenWeightGradient[l - 1, k, j] = Neurons[l - 1, j].value * ActivationFunctions.Sigmoid(Zvals[l, k]) * ErrorSignals[l, k];
                    }
                }
            }
        }
        public void calculate(double[,] image)
        {
            Zvals = new double[Depth, Count];
            Random r = new Random();
            for (int l = 0; l < Depth; l++)
            {
                for (int k = 0; k < Count; k++)
                {
                    //Calc input layer
                    if (l == 0)
                    {
                        for (int j = 0; j < (Resolution * Resolution); j++)
                        {
                            Zvals[0, k] += (InputWeights[k, j] * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + Biases[l, k];
                        }
                    }
                    //Calc hidden layers
                    else
                    {
                        for (int j = 0; j < Count; j++)
                        {
                            double bias; 
                            if (l == Depth - 1) { bias = 0; } else { bias = Biases[l, k]; }
                            //if (l == Depth - 2) { dropout = (r.NextDouble() <= DropoutRate ? 0 : 1); } else { dropout = 1; }
                            Zvals[l, k] += ((HiddenWeights[l - 1, k, j] * Zvals[l - 1, j]) + bias);
                        }
                    }
                }
            }
            //Normalize zvals
            //Zvals = ActivationFunctions.Normalize(Zvals, Depth, Count);
            //Standardize and set values
            for (int l = 0; l < Depth; l++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    if (l < Depth - 1) { Neurons[l, ii].value = ActivationFunctions.Softplus(Zvals[l, ii]); continue; }
                    Neurons[l, ii].value = Zvals[l, ii];
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
                    //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                    InputWeights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (double)(Resolution * Resolution));
                }
            }
            //Initialize hidden weights and all biases
            for (int i = 0; i < Depth - 1; i++)
            {
                for (int ii = 0; ii < Count; ii++)
                {
                    Biases[i, ii] = 0;
                    for (int iii = 0; iii < Count; iii++)
                    {
                        //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                        HiddenWeights[i, ii, iii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (NN.Count * NN.Count));
                    }
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
    class ActivationFunctions
    {
        public static double Softplus(double number)
        {
            double num = Math.Log(1 + Math.Pow(Math.E, number));
            if (num < 0) { num = 0; }
            if (double.IsNaN(num)) { Console.WriteLine("nan"); }
            return num;
        }
        //Derrivative of the softplus
        public static double Sigmoid(double number)
        {
            return 1 / (1 + Math.Pow(Math.E, -number));
        }

        public static double[] Normalize(double[] array)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in array) { mean += d; }
            mean /= array.Length;
            //Calc std dev of data
            foreach (double d in array) { stddev += (d - mean) * (d - mean); }
            stddev /= array.Length;
            stddev = Math.Sqrt(stddev);
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
            //Calc zscore
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (array[i] - mean) / stddev;
            }
            return array;
        }
        public static double[,] Normalize(double[,] array, int depth, int count)
        {
            for (int i = 0; i < depth; i++)
            {
                double[] smallarray = new double[count];
                for (int ii = 0; ii < count; ii++)
                {
                    smallarray[ii] = array[i, ii];
                }
                smallarray = Normalize(smallarray);
                for (int ii = 0; ii < count; ii++)
                {
                    array[i, ii] = smallarray[ii];
                }
            }
            return array;
        }
    }
}
