using System;
using System.Collections.Generic;
using System.IO;

namespace Digits
{
    class D
    {
        const string Path = @"C:\Users\gwflu\Desktop\Test\DataBackup.txt";
        static bool Running = false;
        //Read weight and bias data from a file created by the writer method
        public static void ReadWeightBias()
        {
            //This is a singleton process
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
                    NN.InputWeights[i, ii] = weight;
                    iterator++;
                }
            }
            //Read hidden weights
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    if (i == 0)
                    {
                        for (int iii = 0; iii < NN.InputCount; iii++)
                        {
                            double.TryParse(splitline[iterator], out double weight);
                            NN.FirstHiddenWeights[ii, iii] = weight;
                            iterator++;
                        }
                    }
                    else
                    {
                        for (int iii = 0; iii < NN.HiddenCount; iii++)
                        {
                            double.TryParse(splitline[iterator], out double weight);
                            NN.HiddenWeights[i - 1, ii, iii] = weight;
                            iterator++;
                        }
                    }
                }
            }
            //Read output weights
            for (int i = 0; i < NN.OutputCount; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    double.TryParse(splitline[iterator], out double weight);
                    NN.OutputWeights[i, ii] = weight;
                    iterator++;
                }
            }
            //Read input biases
            for (int i = 0; i < NN.InputCount; i++)
            {
                double.TryParse(splitline[iterator], out double bias);
                NN.InputBiases[i] = bias;
                iterator++;
            }
            //Read hidden biases
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    double.TryParse(splitline[iterator], out double bias);
                    NN.HiddenBiases[i, ii] = bias;
                    iterator++;
                }
            }
            sr.Close(); fs.Close();
            Running = false;
        }
        //Write weight and bias data to a file
        public static void WriteWeightBias()
        {
            //This is a singleton process
            if (Running == true) { throw new Exception("Already accessing file"); }
            Running = true;
            FileStream fs = new FileStream(Path, FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);
            //Write input weights
            for (int i = 0; i < NN.InputCount; i++)
            {
                for (int ii = 0; ii < NN.Resolution * NN.Resolution; ii++)
                {
                    sw.Write(NN.InputWeights[i, ii].ToString() + " ");
                }
            }
            //Write hidden weights
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    if (i == 0)
                    {
                        for (int iii = 0; iii < NN.InputCount; iii++)
                        {
                            sw.Write(NN.FirstHiddenWeights[ii, iii].ToString() + " ");
                        }
                    }
                    else
                    {
                        for (int iii = 0; iii < NN.HiddenCount; iii++)
                        {
                            sw.Write(NN.HiddenWeights[i - 1, ii, iii].ToString() + " ");
                        }
                    }
                }
            }
            //Write output weights
            for (int i = 0; i < NN.OutputCount; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    sw.Write(NN.OutputWeights[i, ii].ToString() + " ");
                }
            }
            //Write input biases
            for (int i = 0; i < NN.InputCount; i++)
            {
                sw.Write(NN.InputBiases[i].ToString() + " ");
            }
            //Write hidden biases
            for (int i = 0; i < NN.HiddenDepth; i++)
            {
                for (int ii = 0; ii < NN.HiddenCount; ii++)
                {
                    sw.Write(NN.HiddenBiases[i, ii].ToString() + " ");
                }
            }
            sw.Close(); fs.Close();
            Running = false;
        }
    }
}
