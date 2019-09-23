using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Digits
{
    public class NeuralNet
    {
        public List<Neuron> Neurons = new List<Neuron>();
        public Neuron[] Outputs = new Neuron[10];
        public static int depth = 3;
        public static int count = 7;
        public void Learn()
        {
            Reader r = new Reader();
            int correct = r.ReadNextLabel()[0];
            int[] image = r.ReadNextImage();
            
        }
        public void LossFunction(int[] image, int correct)
        {
            //use https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931 to continue
        }
        public void computecvals(int[] image)
        {
            foreach (Neuron n in Neurons)
            {
                n.computeCVal(image);
            }
        }
        public void initNN()
        {
            try
            {
                //Randomize the weights
                Random r = new Random();
                for (int i = 0; i <= depth; i++)
                {
                    if (i == 0)
                    {
                        for (int ii = 0; ii <= count - 1; ii++)
                        {
                            //Foreach space in the l1 weight array, randomize it
                            double[] temps = new double[28 * 28];
                            for (int j = 0; j < temps.Length; j++) { temps[j] = Sigmoid.sigmoid(r.Next(-9, 9)); }
                            double[] temps2 = new double[28 * 28];
                            for (int j = 0; j < temps.Length; j++) { temps2[j] = Sigmoid.sigmoid(r.Next(-9, 9)); }
                            Neuron n = new Neuron(this, temps, temps2, 0, 0);
                        }
                    }
                    if (i >= 1 && i <= depth - 1)
                    {
                        for (int ii = 0; ii <= count - 1; ii++)
                        {
                            Neuron n = new Neuron(this, new Dictionary<Neuron, double[]>(), 0, i);
                            n.layWeightBias.Clear();
                            foreach (Neuron neu in Neurons)
                            {
                                if (neu.layer == n.layer - 1)
                                {
                                    if (!n.layWeightBias.ContainsKey(neu))
                                    {
                                        //Make a connection of random weight/bias to each neuron with a layer n - 1 lower
                                        n.layWeightBias.Add(neu, /* Weight/bias for the neuron */ new double[2] { Sigmoid.sigmoid(r.Next(-9, 9)), Sigmoid.sigmoid(r.Next(-9, 9)) });
                                    }
                                }
                            }
                        }
                    }
                    if (i == depth)
                    {
                        //Make outputs coorisponding to digits 0-9
                        for (int j = 0; j <= 9; j++)
                        {
                            //Make the final neuron (output)
                            Neuron n = new Neuron(this, new Dictionary<Neuron, double[]>(), 0, i);
                            n.layWeightBias.Clear();
                            foreach (Neuron neu in Neurons)
                            {
                                if (neu.layer == n.layer - 1)
                                {
                                    if (!n.layWeightBias.ContainsKey(neu))
                                    {
                                        //Make a connection of random weight/bias to each neuron with a layer n - 1 lower
                                        n.layWeightBias.Add(neu, /* Weight/bias for the neuron */ new double[2] { Sigmoid.sigmoid(r.Next(-9, 9)), Sigmoid.sigmoid(r.Next(-9, 9)) });
                                    }
                                }
                            }
                            //Output of the NN is this neuron
                            Outputs[j] = n;
                        }                   
                    }
                }
            }
            //If it fails, print the error out
            catch (Exception ex) { Console.WriteLine(ex); }
        }
    }
    public class Neuron
    {
        public NeuralNet NN { get; set; }
        public int layer { get; set; }
        public double[] weights { get; set; }
        public double[] biases { get; set; }
        public Dictionary<Neuron, double[]> layWeightBias = new Dictionary<Neuron, double[]>();
        public double currentVal;

        public Neuron(NeuralNet nn, double[] ws, double[] bs, double cval, int lay)
        {
            currentVal = cval; layer = lay;
            NN = nn; nn.Neurons.Add(this);
            weights = ws;
        }
        /// <summary>
        /// Make sure to specify the wets array later! If not, DO NOT use this factory
        /// </summary>
        public Neuron(NeuralNet nn, Dictionary<Neuron, double[]> vals, double cval, int lay)
        {
            layWeightBias = vals; currentVal = cval; layer = lay; NN = nn;
            nn.Neurons.Add(this);
        }
        /// <summary>
        /// Make sure to specify the vals/weights dict/array later! If not, DO NOT use this factory.
        /// </summary>
        public Neuron(NeuralNet nn, double cval, int lay, int resolution)
        {
            NN = nn; currentVal = cval; layer = lay; NN.Neurons.Add(this);
            weights = new double[resolution]; layWeightBias = new Dictionary<Neuron, double[]>();
        }
        public void computeCVal(int[] array)
        {
            if (layer == 0)
            {
                currentVal = 0;
                for (int i = 0; i < array.Length; i++)
                {
                    currentVal += (array[i] * weights[i]) + biases[i];
                }
            }
            if (layer >= 1)
            {
                currentVal = 0;
                foreach (KeyValuePair<Neuron, double[]> kvp in layWeightBias)
                {
                    currentVal += (kvp.Key.currentVal * kvp.Value[0]) + kvp.Value[1];
                }
            }
            //Ensure that the value of the neuron is a percent activation
            currentVal = Sigmoid.sigmoid(currentVal);
        }
    }
    class Sigmoid
    {
        public static double sigmoid(double number)
        {
            return 1 / (1 + Math.Pow(Math.E, -number));
        }
    }
}
