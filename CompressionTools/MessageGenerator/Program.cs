using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MessageGenerator
{
    class Program
    {
        static void Main(string[] args)
        {
            int length = 0, type = 0;
            double percent = 0;
            string str = "", outFile = "";

            for (int i = 0; i < args.Length; i++)
            {
                string option = args[i];
                switch (option)
                {
                    case "-s":
                        i++;
                        length = int.Parse(args[i]);
                        break;
                    case "-m":
                        i++;
                        str = args[i];
                        break;
                    case "-o":
                        i++;
                        outFile = args[i];
                        break;
                    case "-t":
                        i++;
                        type = int.Parse(args[i]);
                        break;
                    case "-p":
                        i++;
                        percent = double.Parse(args[i]);
                        break;
                    default:
                        break;
                }
            }

            if(type == 0)
            {
                generateMessage(length, str, outFile);
            }
            else if(type == 1)
            {
                Console.WriteLine(percent);

                generateMessageBinary(length, percent / 100.0, outFile);
            }
        }

        public static void generateMessage(int length, string str, string outFile)
        {
            int count = length / str.Length;
            string message = string.Concat(Enumerable.Repeat(str, count));
            message += str.Substring(0, length % str.Length);

            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message);
        }

        public static void generateMessageBinary(int length, double percent0, string outFile)
        {
            StringBuilder message = new StringBuilder(length);
            float compteur = 0;
            Random rand = new Random();
            for(int i = 0; i < length; i++)
            {
                double num = rand.NextDouble();
                if(num < percent0)
                {
                    message.Append(0);
                    compteur++;
                }
                else
                {
                    message.Append(1);
                }

            }

            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message.ToString());
            Console.WriteLine("Pourcentage d'occurence des symboles binaires:");
            Console.WriteLine($"0: {compteur/length * 100}%, 1: {(length - compteur)/length * 100}%");
        }


    }
}
