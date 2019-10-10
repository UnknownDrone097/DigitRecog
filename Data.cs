using System;
using System.Collections.Generic;
using System.IO;

namespace Digits
{
    class D
    {
        const string Path = @"C:\Users\gwflu\Desktop\Test\DataBackup.txt";
        static bool Running = false;
        public static void ReadWeightBias(NN nn)
        {
            if (Running == true) { throw new Exception("Already accessing file"); }
            Running = true;
            FileStream fs = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.None);
            StreamReader sr = new StreamReader(fs);
            string all = sr.ReadToEnd();
            string[] splitline = all.Split(' ');
            int iterator = 0;
            //Read input weights
            for (int i = 0; i < NN.InputCount; i++)
            {
                for (int ii = 0; ii < NN.Resolution * NN.Resolution; ii++)
                {
                    double.TryParse(splitline[iterator], out double weight);
                    nn.InputWeights[i, ii] = weight;
                    iterator++;
                }
            }
            //Read hidden weights
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    for (int iii = 0; iii < NN.InputCount; iii++)
                    {
                        double.TryParse(splitline[iterator], out double weight);
                        nn.HiddenWeights[i, ii, iii] = weight;
                        iterator++;
                    }
                }
            }
            //Read output weights
            for (int i = 0; i < NN.OutputCount; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    double.TryParse(splitline[iterator], out double weight);
                    nn.OutputWeights[i, ii] = weight;
                    iterator++;
                }
            }
            //Read input biases
            for (int i = 0; i < NN.InputCount; i++)
            {
                double.TryParse(splitline[iterator], out double bias);
                nn.InputBiases[i] = bias;
                iterator++;
            }
            //Read hidden biases
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    double.TryParse(splitline[iterator], out double bias);
                    nn.HiddenBiases[i, ii] = bias;
                    iterator++;
                }
            }
            sr.Close(); fs.Close();
            Running = false;
        }
        public static void WriteWeightBias(NN nn)
        {
            if (Running == true) { throw new Exception("Already accessing file"); }
            Running = true;
            FileStream fs = new FileStream(Path, FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);
            //Write input weights
            for (int i = 0; i < NN.InputCount; i++)
            {
                for (int ii = 0; ii < NN.Resolution * NN.Resolution; ii++)
                {
                    sw.Write(nn.InputWeights[i, ii].ToString() + " ");
                }
            }
            //Write hidden weights
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    for (int iii = 0; iii < NN.InputCount; iii++)
                    {
                        sw.Write(nn.HiddenWeights[i, ii, iii].ToString() + " ");
                    }
                }
            }
            //Write output weights
            for (int i = 0; i < NN.OutputCount; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    sw.Write(nn.OutputWeights[i, ii].ToString() + " ");
                }
            }
            //Write input biases
            for (int i = 0; i < NN.InputCount; i++)
            {
                sw.Write(nn.InputBiases[i].ToString() + " ");
            }
            //Write hidden biases
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    sw.Write(nn.HiddenBiases[i, ii].ToString() + " ");
                }
            }
            sw.Close(); fs.Close();
            Running = false;
        }
    }
}
