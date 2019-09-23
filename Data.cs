using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;

namespace Digits
{
    class Data
    {
        public const string weightPath = @"C:\Users\gwflu\Desktop\Test\Digits.txt";
        public const string biasPath = @"";
        public static void ReadNs(NeuralNet NN, string path, bool weightorbias, int resolution)
        {
            NN.Neurons = new List<Neuron>();
            FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
            StreamReader sr = new StreamReader(fs);
            while (!sr.EndOfStream)
            {
                string line = sr.ReadLine();
                string[] splitLine = line.Split(' ');

                if (splitLine[0] == "Neuron" && splitLine[1] == "0")
                {
                    int.TryParse(splitLine[1], out int result);
                    Neuron n = new Neuron(NN, 0, result, resolution);
                    for (int i = 0; i < resolution; i++)
                    {
                        line = sr.ReadLine();
                        splitLine = line.Split(' ');
                        try
                        {
                            double.TryParse(splitLine[i], out double result2);
                            if (weightorbias) { n.weights[i] = result2; }
                            else { n.biases[i] = result2; }
                            
                        }
                        catch (Exception ex) { Console.WriteLine(ex); }
                    }
                }
                if (splitLine[0] == "Neuron" && splitLine[1] != "0")
                {
                    Dictionary<Neuron, double> layerwets = new Dictionary<Neuron, double>();
                    int.TryParse(splitLine[1], out int result);
                    Neuron n = new Neuron(NN, 0, result, resolution);
                    if (result == 3) { int.TryParse(splitLine[2], out int result2); NN.Outputs[result2] = n; }
                    line = sr.ReadLine();
                    splitLine = line.Split(' ');

                    int iterator = 0;
                    foreach (Neuron neuron in NN.Neurons)
                    {
                        if (neuron.layer == result - 1)
                        {
                            double.TryParse(splitLine[iterator], out double result2);
                            iterator++;
                            layerwets.Add(neuron, result2);
                        }
                    }    
                    foreach (KeyValuePair<Neuron, double> kvp in layerwets)
                    {
                        //May be broken
                        if (n.layWeightBias == null) { n.layWeightBias[kvp.Key] = new double[2] { 0, 0 }; }
                        double[] array = n.layWeightBias[kvp.Key];
                        if (weightorbias) { array[0] = kvp.Value; }
                        else { array[1] = kvp.Value; }
                        n.layWeightBias.Add(kvp.Key, array);
                    }
                }
            }
            sr.Close(); fs.Close();
        }
        public static void WriteNs(NeuralNet NN, bool weightorbias, string path, int resolution)
        {
            FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);
            foreach (Neuron n in NN.Neurons)
            {
                sw.WriteLine("Neuron" + ' ' + n.layer);
                if (NN.Outputs.Contains(n)) { sw.WriteLine(' ' + Array.IndexOf(NN.Outputs, n)); }
                if (n.layer == 0)
                {
                    for (int i = 0; i < (resolution * resolution); i++)
                    {
                        if (weightorbias) { sw.Write((n.weights[i]).ToString()); }
                        else { sw.Write((n.biases[i].ToString())); }
                        sw.WriteLine();
                    }
                }
                else
                {
                    int count = 1;
                    foreach (KeyValuePair<Neuron, double[]> kvp in n.layWeightBias)
                    {
                        if (weightorbias) { sw.Write((kvp.Value[0]).ToString()); }
                        else { sw.Write(kvp.Value[1].ToString()); }  
                        count++;
                    }
                    sw.WriteLine();
                }
            }
            sw.Close(); fs.Close();
        }
    }
}
