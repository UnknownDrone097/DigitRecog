using System;
using System.Collections.Generic;
using System.IO;

namespace Digits
{
    class D
    {
        const string Path = @"H:\documents\wbs.txt";

        public static void ReadWeightBias(NN nn)
        {
            FileStream fs = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.None);
            StreamReader sr = new StreamReader(fs);
            string all = sr.ReadToEnd();
            string[] splitline = all.Split(' ');
            int iterator = 0;
            //Read input weights/biases
            for (int i = 0; i < NN.Count; i++)
            {
                for (int ii = 0; ii < NN.Resolution * NN.Resolution; ii++)
                {
                    string[] split = splitline[iterator].Split(',');
                    double.TryParse(split[0], out double weight);
                    double.TryParse(split[1], out double bias);
                    nn.InputWeights[i, ii] = weight;
                    nn.InputBiases[i, ii] = bias;
                    iterator++;
                }
            }
            //Read hidden weights/biases
            for (int i = 0; i < NN.Depth - 1; i++)
            {
                for (int ii = 0; ii < NN.Count; ii++)
                {
                    for (int iii = 0; iii < NN.Count; iii++)
                    {
                        string[] split = splitline[iterator].Split(',');
                        double.TryParse(split[0], out double weight);
                        double.TryParse(split[1], out double bias);
                        nn.HiddenWeights[i, ii, iii] = weight;
                        nn.HiddenBiases[i, ii, iii] = bias;
                        iterator++;
                    }
                }
            }
            sr.Close(); fs.Close();
        }
        public static void WriteWeightBias(NN nn)
        {
            FileStream fs = new FileStream(Path, FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);
            //Write input weights/biases
            for (int i = 0; i < NN.Count; i++)
            {
                for (int ii = 0; ii < NN.Resolution * NN.Resolution; ii++)
                {
                    sw.Write(nn.InputWeights[i, ii].ToString() + "," + nn.InputBiases[i, ii].ToString() + " ");
                }
            }
            //Write hidden weights/biases
            for (int i = 0; i < NN.Depth - 1; i++)
            {
                for (int ii = 0; ii < NN.Count; ii++)
                {
                    for (int iii = 0; iii < NN.Count; iii++)
                    {
                        sw.Write(nn.HiddenWeights[i, ii, iii].ToString() + "," + nn.HiddenBiases[i, ii, iii].ToString() + " ");
                    }
                }
            }
            sw.Close(); fs.Close();
        }
    }
}
