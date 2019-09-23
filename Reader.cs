namespace Digits
{
    class Reader
    {
        //Need to check for end of file
        private readonly string LabelPath = @"H:\Documents\train-labels-idx1-ubyte";
        private readonly string ImagePath = @"H:\Documents\train-images-idx3-ubyte";
        int LabelOffset = 8;
        int ImageOffset = 16;
        int Resolution = 28;
        public int[] ReadNextLabel()
        {
            FileStream fs = File.OpenRead(LabelPath);
            fs.Position = LabelOffset;
            byte[] b = new byte[1];
            try
            {
                fs.Read(b, 0, 1);
            }
            catch { Console.WriteLine("Reset; ImageOffset = " + LabelOffset.ToString()); LabelOffset = 0; }
            int[] result = Array.ConvertAll(b, Convert.ToInt32);
            LabelOffset++;
            fs.Close();
            return result;
        }
        public int[] ReadNextImage()
        {
            FileStream fs = File.OpenRead(ImagePath);
            fs.Position = ImageOffset;
            byte[] b = new byte[Resolution * Resolution];
            try
            {
                fs.Read(b, 0, Resolution * Resolution);
            }
            catch { Console.WriteLine("Reset; ImageOffset = " + ImageOffset.ToString()); ImageOffset = 0; }
            int[] result = Array.ConvertAll(b, Convert.ToInt32);
            ImageOffset += Resolution * Resolution;
            fs.Close();
            return result;
        }
        public void PrintArray(int[] a)
        {
            int iterator = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (iterator == Resolution) { iterator = 0; Console.WriteLine(); }
                Console.Write(a[i] + " ");
                iterator++;
            }
        }
    }
}
