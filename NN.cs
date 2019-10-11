using System;

namespace Digits
{
    /// <summary>
    /// Prepare to cry!
    /// Abandon all hope, ye who enter here, for the depths of despair lurk within.
    /// 
    /// In all seriousness, if you understand gradients its not that bad, but the math is very obtuse,
    /// so don't screw with it if you don't know what you're doing.
    /// </summary>
    class NN : IDisposable
    {
        //May want to add batch normalization

        //Depth of the entire network
        public static int Depth = 3;
        //Depth of the hidden layers
        public static int HiddenDepth = 1;
        //Count of neurons per layer
        public static int HiddenCount = 19;
        public static int OutputCount = 10;
        public static int Resolution = 28;
        public static int InputCount = Resolution;
        //An output parameter to be written to console
        public double AvgGradient = 0;
        //Hyperparameters
        public static double LearningRate = 0.00000146;
        public static double Momentum = .4;
        //Overall gradients
        double[,] AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
        double[,,] AvgHiddenWeightGradient = new double[HiddenDepth, HiddenCount, InputCount];
        double[,] AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
        double[,] AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
        double[] AvgInputBiasGradient = new double[InputCount];
        //Error signals
        double[] InputErrorSignals = new double[InputCount];
        double[,] HiddenErrorSignals = new double[HiddenDepth, HiddenCount];
        double[] OutputErrorSignals = new double[OutputCount];
        //Weights (public to allow reading and writing without needing methods for it 
        //[yes, this is bad practice, but the point of this code is the maths, not the practice])
        public double[,] InputWeights = new double[InputCount, Resolution * Resolution];
        public double[,,] HiddenWeights = new double[HiddenDepth, HiddenCount, InputCount];
        public double[,] OutputWeights = new double[OutputCount, HiddenCount];
        //Biases (same thing with the publicity)
        public double[] InputBiases = new double[InputCount];
        public double[,] HiddenBiases = new double[HiddenDepth, HiddenCount];
        //Zvals
        double[] InputZVals = new double[InputCount];
        double[,] HiddenZVals = new double[HiddenDepth, HiddenCount];
        double[] OutputZVals = new double[OutputCount];
        //Gradients
        double[,] InputWeightGradient = new double[InputCount, Resolution * Resolution];
        double[,,] HiddenWeightGradient = new double[Depth - 1, HiddenCount, InputCount];
        double[,] OutputWeightGradient = new double[OutputCount, HiddenCount];
        //Momentums
        double[,] InputWeightMomentum = new double[InputCount, Resolution * Resolution];
        double[,,] HiddenWeightMomentum = new double[Depth - 1, HiddenCount, InputCount];
        double[,] OutputWeightMomentum = new double[OutputCount, HiddenCount];
        //Values
        double[] InputValues = new double[InputCount];
        double[,] HiddenValues = new double[HiddenDepth, HiddenCount];
        //Public to allow reading for validation
        public double[] OutputValues = new double[OutputCount];

        //Batch descent
        public void Descend(int batchsize)
        {
            //Reset avg gradient
            AvgGradient = 0;

            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    InputWeights[i, ii] -= LearningRate * AvgInputWeightGradient[i, ii] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * AvgInputWeightGradient[i, ii] * (-2 / (double)batchsize);
                }
                InputBiases[i] -= LearningRate * AvgInputBiasGradient[i] * (-2 / (double)batchsize);
            }
            //Hidden
            for (int i = 0; i < HiddenDepth; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    for (int iii = 0; iii < InputCount; iii++)
                    {
                        HiddenWeights[i, ii, iii] -= LearningRate * AvgHiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                        AvgGradient -= LearningRate * AvgHiddenWeightGradient[i, ii, iii] * (-2 / (double)batchsize);
                    }
                    HiddenBiases[i, ii] -= LearningRate * AvgHiddenBiasGradient[i, ii] * (-2 / (double)batchsize);
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    OutputWeights[i, ii] -= LearningRate * AvgOutputWeightGradient[i, ii] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * AvgOutputWeightGradient[i, ii] * (-2 / (double)batchsize);
                }
            }
            AvgGradient /= HiddenWeightGradient.Length + InputWeightGradient.Length + OutputWeightGradient.Length;

            //Reset averages
            AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
            AvgHiddenWeightGradient = new double[HiddenDepth, HiddenCount, InputCount];
            AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
            AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
            AvgInputBiasGradient = new double[InputCount];
        }
        //Stochastic descent (all code below is done according to formulas)
        public void Descend()
        {
            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int ii = 0; ii < Resolution * Resolution; ii++)
                {
                    //Nesterov momentum
                    InputWeightMomentum[i, ii] = (InputWeightMomentum[i, ii] * Momentum) - (LearningRate * InputWeightGradient[i, ii]);
                    AvgInputWeightGradient[i, ii] += InputWeightGradient[i, ii] + InputWeightMomentum[i, ii];
                }
                AvgInputBiasGradient[i] += InputErrorSignals[i] * ActivationFunctions.TanhDerriv(InputZVals[i]);
            }
            //Hidden
            for (int i = 0; i < HiddenDepth; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    for (int iii = 0; iii < InputCount; iii++)
                    {
                        //Nesterov momentum
                        HiddenWeightMomentum[i, ii, iii] = (HiddenWeightMomentum[i, ii, iii] * Momentum) - (LearningRate * HiddenWeightGradient[i, ii, iii]);
                        AvgHiddenWeightGradient[i, ii, iii] += HiddenWeightGradient[i, ii, iii] + HiddenWeightMomentum[i, ii, iii];
                    }
                    AvgHiddenBiasGradient[i, ii] += HiddenErrorSignals[i, ii] * ActivationFunctions.TanhDerriv(HiddenZVals[i, ii]);
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    //Nesterov momentum
                    OutputWeightMomentum[i, ii] = (OutputWeightMomentum[i, ii] * Momentum) - (LearningRate * OutputWeightGradient[i, ii]);
                    AvgOutputWeightGradient[i, ii] += OutputWeightGradient[i, ii] + OutputWeightMomentum[i, ii];
                }
            }
        }
        /// <summary>
        /// Backpropagation of error (formulas)
        /// </summary>
        /// <param name="image">The matrix (image) to be forward propagated from</param>
        /// <param name="correct">The number shown in the image</param>
        public void backprop(double[,] image, int correct)
        {
            //Forward propagation of data
            calculate(image);

            //Reset things about to be calculated
            InputErrorSignals = new double[InputCount];
            HiddenErrorSignals = new double[HiddenDepth, HiddenCount];
            OutputErrorSignals = new double[OutputCount];
            InputWeightGradient = new double[InputCount, Resolution * Resolution];
            HiddenWeightGradient = new double[Depth - 1, HiddenCount, InputCount];
            OutputWeightGradient = new double[OutputCount, HiddenCount];

            //Output
            //Foreach ending neuron
            for (int k = 0; k < OutputCount; k++)
            {
                double upperlayerderiv = 2d * ((k == correct ? 1d : 0d) - OutputValues[k]);
                OutputErrorSignals[k] = upperlayerderiv;

                //Calculate gradient
                //This works b/c of only 1 hidden layer, will need to be changed if HiddenDepth is modified
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputWeightGradient[k, j] = HiddenValues[HiddenDepth - 1, j] * ActivationFunctions.TanhDerriv(OutputZVals[k]) * OutputErrorSignals[k];
                }
            }
            //Hidden
            //Foreach layer of hidden 'neurons'
            for (int l = 0; l < HiddenDepth; l++)
            {
                //Hidden upper layer derrivative calculation
                //Foreach starting neuron
                for (int k = 0; k < HiddenCount; k++)
                {
                    double upperlayerderiv = 0;
                    //Foreach ending neuron
                    for (int j = 0; j < OutputCount; j++)
                    {
                        //Hiddenweights uses l because the formula's l + 1 is l due to a lack of input layer in this array
                        upperlayerderiv += OutputWeights[j, k] * ActivationFunctions.TanhDerriv(OutputZVals[j]) * OutputErrorSignals[j];
                    }
                    HiddenErrorSignals[l, k] = upperlayerderiv;
                }
                //Foreach starting neuron
                for (int k = 0; k < HiddenCount; k++)
                {
                    //Foreach ending neuron neuron
                    for (int j = 0; j < InputCount; j++)
                    {
                        HiddenWeightGradient[l, k, j] = InputValues[j] * ActivationFunctions.TanhDerriv(HiddenZVals[l, k]) * HiddenErrorSignals[l, k];
                    }
                }
            }
            //Input
            //Foreach starting neuron
            for (int k = 0; k < InputCount; k++)
            {
                double upperlayerderiv = 0;

                //Calculate error signal
                //Foreach ending neuron
                for (int j = 0; j < HiddenCount; j++)
                {
                    upperlayerderiv += HiddenWeights[0, j, k] * ActivationFunctions.TanhDerriv(HiddenZVals[0, j]) * HiddenErrorSignals[0, j];
                }
                InputErrorSignals[k] = upperlayerderiv;

                //Calculate gradient
                for (int j = 0; j < Resolution * Resolution; j++)
                {
                    InputWeightGradient[k, j] = image[j / Resolution, j - ((j / Resolution) * Resolution)] * ActivationFunctions.TanhDerriv(InputZVals[k]) * InputErrorSignals[k];
                }
            }
            //Normalize gradients (currently disabled as is obvious)
            /*
            InputWeightGradient = ActivationFunctions.Normalize(InputWeightGradient, InputCount, Resolution * Resolution);
            HiddenWeightGradient = ActivationFunctions.Normalize(HiddenWeightGradient, HiddenDepth, HiddenCount, InputCount);
            OutputWeightGradient = ActivationFunctions.Normalize(OutputWeightGradient, OutputCount, HiddenCount);
            //Normalize error signals (biases)
            HiddenErrorSignals = ActivationFunctions.Normalize(HiddenErrorSignals, HiddenDepth, HiddenCount);
            InputErrorSignals = ActivationFunctions.Normalize(InputErrorSignals);
            */
        }
        /// <summary>
        /// Forward propagation of values
        /// </summary>
        /// <param name="image">The matrix (image) to be forward propagated from</param>
        public void calculate(double[,] image)
        {
            //Reset ZVals (raw values untouched by the activation function)
            InputZVals = new double[InputCount];
            HiddenZVals = new double[HiddenDepth, HiddenCount];
            OutputZVals = new double[OutputCount];

            //Random r = new Random();
            //Random is used for dropout of neurons, but said feature is currently disabled for efficiency reasons

            //Input
            for (int k = 0; k < InputCount; k++)
            {
                for (int j = 0; j < (Resolution * Resolution); j++)
                {
                    InputZVals[k] += ((InputWeights[k, j] + InputWeightMomentum[k, j]) * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + InputBiases[k];
                }
                InputValues[k] = ActivationFunctions.Tanh(InputZVals[k]);
            }
            //Hidden
            for (int l = 0; l < HiddenDepth; l++)
            {
                for (int k = 0; k < HiddenCount; k++)
                {
                    for (int j = 0; j < InputCount; j++)
                    {
                        //Former dropout code, if desired must be added to input and output as well
                        //if (l == Depth - 2) { dropout = (r.NextDouble() <= DropoutRate ? 0 : 1); } else { dropout = 1; }
                        HiddenZVals[l, k] += (((HiddenWeights[l, k, j] + HiddenWeightMomentum[l, k, j]) * InputZVals[j]) + HiddenBiases[l, k]);
                    }
                    HiddenValues[l, k] = ActivationFunctions.Tanh(HiddenZVals[l, k]);
                }
            }
            //Output
            for (int k = 0; k < OutputCount; k++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputZVals[k] += ((OutputWeights[k, j] + OutputWeightMomentum[k, j]) * HiddenZVals[HiddenDepth - 1, j]);
                }
                //No activation function on outputs
                OutputValues[k] = OutputZVals[k];
                //OutputValues[k] = ActivationFunctions.Tanh(OutputZVals[k]);
            }
        }
        /// <summary>
        /// Reset the weights and biases of the neuron (randomly and to zero, respectively)
        /// </summary>
        public void initialize()
        {
            Random r = new Random();
            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int ii = 0; ii < (Resolution * Resolution); ii++)
                {
                    //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                    InputWeights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (double)(Resolution * Resolution));
                }
            }
            //Hidden
            for (int l = 0; l < HiddenDepth; l++)
            {
                for (int i = 0; i < HiddenCount; i++)
                {
                    for (int ii = 0; ii < InputCount; ii++)
                    {
                        //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                        HiddenWeights[l, i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputCount * InputCount));
                    }
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                    OutputWeights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (double)(HiddenCount * HiddenCount));
                }
            }
        }

        //Probably useless?
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
    /// <summary>
    /// Mapping the raw data onto a neuron-like firing state (on or off, ish)
    /// </summary>
    class ActivationFunctions
    {
        //Hyperbolic tangent
        public static double Tanh(double number)
        {
            return (Math.Pow(Math.E, 2 * number) - 1) / (Math.Pow(Math.E, 2 * number) + 1);
            //The following is unused Smooth ReLU code
            /*
            double num = Math.Log(1 + Math.Pow(Math.E, number));
            if (num < 0) { num = 0; }
            //Ensure program breaks if a NAN is detected
            try { if (double.IsNaN(num)) { throw new Exception("Nan found"); } }
            catch (Exception ex) { Console.WriteLine(ex); Console.ReadLine(); }
            return num;
            */
        }
        //Derrivative of the activation function
        public static double TanhDerriv(double number)
        {
            return (1 - Math.Pow(Tanh(number), 2));
        }
        //Currently no clipping, may want to add eventually?
        /// <summary>
        /// Normalize the data contained in the array
        /// </summary>
        /// <param name="array">Array to be normalized</param>
        /// <returns></returns>
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
        /// <summary>
        /// Normalize the data contained in the array
        /// </summary>
        /// <param name="array">Array to be normalized</param>
        /// <param name="depth">Row count of array</param>
        /// <param name="count">Column count of array</param>
        /// <returns></returns>
        public static double[,] Normalize(double[,] array, int depth, int count)
        {
            double[] smallarray = new double[depth * count];
            int iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    smallarray[iterator] = array[i, ii];
                    iterator++;
                }
            }
            smallarray = Normalize(smallarray);
            iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    array[i, ii] = smallarray[iterator];
                    iterator++;
                }
            }
            return array;
        }
        /// <summary>
        /// Normalize the data contained in the array
        /// </summary>
        /// <param name="array">Array to be normalized</param>
        /// <param name="depth">Row count of array</param>
        /// <param name="count1">Row count of each column of the sub-array</param>
        /// <param name="count2">Column count of the sub-array</param>
        /// <returns></returns>
        public static double[,,] Normalize(double[,,] array, int depth, int count1, int count2)
        {
            double[] workingvalues = new double[depth * count1 * count2];
            int iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count1; ii++)
                {
                    for (int iii = 0; iii < count2; iii++)
                    {
                        workingvalues[iterator] = array[i, ii, iii];
                        iterator++;
                    }
                }
            }
            workingvalues = Normalize(workingvalues);
            iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count1; ii++)
                {
                    for (int iii = 0; iii < count2; iii++)
                    {
                        array[i, ii, iii] = workingvalues[iterator];
                        iterator++;
                    }
                }
            }
            return array;
        }
    }
}
