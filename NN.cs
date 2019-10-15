using System;
using System.Threading.Tasks;

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

        public static int Epoch = 0;
        //Depth of the hidden layers
        public static int HiddenDepth = 2;
        //Depth of the entire network (Hiddendepth + input [1] + output [1])
        public static int Depth = HiddenDepth + 2;
        //Count of neurons per layer
        public static int HiddenCount = 19;
        public static int OutputCount = 10;
        public static int Resolution = 28;
        public static int InputCount = Resolution;
        //An output parameter to be written to console
        public static double AvgGradient = 0;
        //Hyperparameters
        private static double LearningRateDecay = (.000146 / 5d) * (double)Epoch;
        private static double LearningRate = 0.0000146 - LearningRateDecay;
        public static double Momentum = .9;
        //Overall gradients
        static double[,] AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
        static double[,] AvgFirstHiddenWeightGradient = new double[HiddenCount, InputCount];
        static double[,,] AvgHiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
        static double[,] AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
        static double[,] AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
        static double[] AvgInputBiasGradient = new double[InputCount];
        //Error signals
        double[] InputErrorSignals { get; set; }
        double[,] HiddenErrorSignals { get; set; }
        double[] OutputErrorSignals { get; set; }
        //Weights (public to allow reading and writing without needing methods for it 
        //[yes, this is bad practice, but the point of this code is the maths, not the practice])
        static public double[,] InputWeights = new double[InputCount, Resolution * Resolution];
        static public double[,] FirstHiddenWeights = new double[HiddenCount, InputCount];
        static public double[,,] HiddenWeights = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
        static public double[,] OutputWeights = new double[OutputCount, HiddenCount];
        //Biases (same thing with the publicity)
        public static double[] InputBiases = new double[InputCount];
        public static double[,] HiddenBiases = new double[HiddenDepth, HiddenCount];
        //Zvals
        double[] InputZVals { get; set; }
        double[,] HiddenZVals { get; set; }
        double[] OutputZVals { get; set; }
        //Gradients
        double[,] InputWeightGradient { get; set; }
        double[,] FirstHiddenWeightGradient { get; set; }
        double[,,] HiddenWeightGradient { get; set; }
        double[,] OutputWeightGradient { get; set; }
        //Momentums
        static double[,] InputWeightMomentum = new double[InputCount, Resolution * Resolution];
        static double[,] FirstHiddenWeightMomentum = new double[HiddenCount, InputCount];
        static double[,,] HiddenWeightMomentum = new double[HiddenDepth, HiddenCount, HiddenCount];
        static double[,] OutputWeightMomentum = new double[OutputCount, HiddenCount];
        static double[] InputBiasMomentum = new double[InputCount];
        static double[,] HiddenBiasMomentum = new double[HiddenDepth, HiddenCount];
        //Values
        double[] InputValues { get; set; }
        double[,] HiddenValues { get; set; }
        //Public to allow reading for validation
        public double[] OutputValues = new double[OutputCount];
        //Batch descent
        public static void Descend(int batchsize)
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
                    if (i == 0)
                    {
                        for (int iii = 0; iii < InputCount; iii++)
                        {
                            FirstHiddenWeights[ii, iii] -= LearningRate * AvgFirstHiddenWeightGradient[ii, iii] * (-2 / (double)batchsize);
                            AvgGradient -= LearningRate * AvgFirstHiddenWeightGradient[ii, iii] * (-2 / (double)batchsize);
                        }
                    }
                    else
                    {
                        for (int iii = 0; iii < HiddenCount; iii++)
                        {
                            HiddenWeights[i - 1, ii, iii] -= LearningRate * AvgHiddenWeightGradient[i - 1, ii, iii] * (-2 / (double)batchsize);
                            AvgGradient -= LearningRate * AvgHiddenWeightGradient[i - 1, ii, iii] * (-2 / (double)batchsize);
                        }
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
            AvgGradient /= AvgHiddenWeightGradient.Length + AvgInputWeightGradient.Length + AvgOutputWeightGradient.Length;

            //Reset averages
            AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
            AvgFirstHiddenWeightGradient = new double[HiddenCount, InputCount];
            AvgHiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
            AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
            AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
            AvgInputBiasGradient = new double[InputCount];
        }
        //Stochastic descent (all code below is done according to formulas)
        //This adds each NN's gradients to the avg
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
                double tempbias = InputErrorSignals[i] * ActivationFunctions.TanhDerriv(InputZVals[i]);
                InputBiasMomentum[i] = (InputBiasMomentum[i] * Momentum) - (LearningRate * tempbias);
                AvgInputBiasGradient[i] += tempbias + InputBiasMomentum[i];
            }
            //Hidden
            for (int i = 0; i < HiddenDepth; i++)
            {
                for (int ii = 0; ii < HiddenCount; ii++)
                {
                    if (i == 0)
                    {
                        for (int iii = 0; iii < InputCount; iii++)
                        {
                            //Nesterov momentum
                            FirstHiddenWeightMomentum[ii, iii] = (FirstHiddenWeightMomentum[ii, iii] * Momentum) - (LearningRate * FirstHiddenWeightGradient[ii, iii]);
                            AvgFirstHiddenWeightGradient[ii, iii] += FirstHiddenWeightGradient[ii, iii] + FirstHiddenWeightMomentum[ii, iii];
                        }
                    }
                    else
                    {
                        for (int iii = 0; iii < HiddenCount; iii++)
                        {
                            //Nesterov momentum
                            HiddenWeightMomentum[i - 1, ii, iii] = (HiddenWeightMomentum[i - 1, ii, iii] * Momentum) - (LearningRate * HiddenWeightGradient[i - 1, ii, iii]);
                            AvgHiddenWeightGradient[i - 1, ii, iii] += HiddenWeightGradient[i - 1, ii, iii] + HiddenWeightMomentum[i - 1, ii, iii];
                        }
                    }
                    double tempbias = HiddenErrorSignals[i, ii] * ActivationFunctions.TanhDerriv(HiddenZVals[i, ii]);
                    HiddenBiasMomentum[i, ii] = (HiddenBiasMomentum[i, ii] * Momentum) - (LearningRate * tempbias);
                    AvgHiddenBiasGradient[i, ii] += tempbias + HiddenBiasMomentum[i, ii];
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
        public void Backprop(double[,] image, int correct)
        {
            //Forward propagation of data
            Calculate(image);

            //Reset things about to be calculated
            InputErrorSignals = new double[InputCount];
            HiddenErrorSignals = new double[HiddenDepth, HiddenCount];
            OutputErrorSignals = new double[OutputCount];
            InputWeightGradient = new double[InputCount, Resolution * Resolution];
            FirstHiddenWeightGradient = new double[HiddenCount, InputCount];
            HiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
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
            //Calc errors
            for (int l = HiddenDepth - 1; l >= 0; l--)
            {
                //Hidden upper layer derrivative calculation
                //Foreach starting neuron
                if (l == HiddenDepth - 1)
                {
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
                }
                else
                {
                    for (int k = 0; k < HiddenCount; k++)
                    {
                        double upperlayerderiv = 0;
                        //Foreach ending neuron
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            //Hiddenweights uses l instead of l + 1 because firsthiddenweights is a different array
                            upperlayerderiv += HiddenWeights[l, j, k] * ActivationFunctions.TanhDerriv(HiddenZVals[l + 1, j]) * HiddenErrorSignals[l + 1, j];
                        }
                        HiddenErrorSignals[l, k] = upperlayerderiv;
                    }
                }
            }
            //Calc values
            for (int l = 0; l < HiddenDepth; l++)
            {
                //Foreach starting neuron
                for (int k = 0; k < HiddenCount; k++)
                {
                    if (l == 0)
                    {
                        //Foreach ending neuron neuron
                        for (int j = 0; j < InputCount; j++)
                        {
                            FirstHiddenWeightGradient[k, j] = InputValues[j] * ActivationFunctions.TanhDerriv(HiddenZVals[l, k]) * HiddenErrorSignals[l, k];
                        }
                    }
                    else
                    {
                        //Foreach ending neuron neuron
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            HiddenWeightGradient[l - 1, k, j] = HiddenValues[l - 1, j] * ActivationFunctions.TanhDerriv(HiddenZVals[l, k]) * HiddenErrorSignals[l, k];
                        }
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
                    upperlayerderiv += FirstHiddenWeights[j, k] * ActivationFunctions.TanhDerriv(HiddenZVals[0, j]) * HiddenErrorSignals[0, j];
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
        public void Calculate(double[,] image)
        {
            //Reset ZVals (raw values untouched by the activation function), vals, and momentums
            InputZVals = new double[InputCount]; InputValues = new double[InputCount];
            HiddenZVals = new double[HiddenDepth, HiddenCount]; HiddenValues = new double[HiddenDepth, HiddenCount];
            OutputZVals = new double[OutputCount]; OutputValues = new double[OutputCount];

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
                    if (l == 0)
                    {
                        for (int j = 0; j < InputCount; j++)
                        {
                            //Former dropout code, if desired must be added to input and output as well
                            //if (l == Depth - 2) { dropout = (r.NextDouble() <= DropoutRate ? 0 : 1); } else { dropout = 1; }
                            HiddenZVals[l, k] += (((FirstHiddenWeights[k, j] + FirstHiddenWeightMomentum[k, j]) * InputValues[j]) + HiddenBiases[l, k]);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            //Former dropout code, if desired must be added to input and output as well
                            //if (l == Depth - 2) { dropout = (r.NextDouble() <= DropoutRate ? 0 : 1); } else { dropout = 1; }
                            //Hiddenweights and momentum use l - 1 because the first layer is under firsthidden and firstmomentum respectively
                            HiddenZVals[l, k] += (((HiddenWeights[l - 1, k, j] + HiddenWeightMomentum[l - 1, k, j]) * HiddenValues[l - 1, j]) + HiddenBiases[l, k]);
                        }
                    }
                    HiddenValues[l, k] = ActivationFunctions.Tanh(HiddenZVals[l, k]);
                }
            }
            //Output
            for (int k = 0; k < OutputCount; k++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputZVals[k] += ((OutputWeights[k, j] + OutputWeightMomentum[k, j]) * HiddenValues[HiddenDepth - 1, j]);
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
                    if (l == 0)
                    {
                        for (int ii = 0; ii < InputCount; ii++)
                        {
                            //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                            FirstHiddenWeights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputCount * InputCount));
                        }
                    }
                    else
                    {
                        for (int ii = 0; ii < HiddenCount; ii++)
                        {
                            //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                            HiddenWeights[l - 1, i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (HiddenCount * HiddenCount));
                        }
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
        public static double[] ApplyActFunc(double[] matrix, int p)
        {
            double[] result = new double[p];
            for (int i = 0; i < p; i++) { result[i] = Tanh(matrix[i]); }
            return result;
        }
        public static double[,] ApplyActFunc(double[,] matrix, int p, int m)
        {
            double[,] result = new double[p, m];
            for (int i = 0; i < p; i++) for (int ii = 0; ii < m; ii++) { { result[i, ii] = Tanh(matrix[i, ii]); } }
            return result;
        }
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
        public static double[] Normalize(double[] array, bool image)
        {
            if (!image)
            {
                double max = 0; double min = 0;
                double a = 0; double b = .0001; 
                foreach (double d in array) { if (d > max) { max = d; } if (d < min) { min = d; } }
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] = (array[i] < 0 ? -1 : 1) * (a + ((array[i] - min) * (b - a) / (max - min)));
                }
            }
            else
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
        public static double[,] Normalize(double[,] array, bool image, int depth, int count)
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
            smallarray = Normalize(smallarray, image);
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
        public static double[,,] Normalize(double[,,] array, bool image, int depth, int count1, int count2)
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
            workingvalues = Normalize(workingvalues, image);
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
