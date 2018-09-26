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
            int length = 0, type = 0, symbolCount = 0;
            double percent = 0;
            string str = "", outFile = "";

            for (int i = 0; i < args.Length; i++)
            {
                string option = args[i];
                switch (option)
                {
                    case "-l":
                        i++;
                        length = int.Parse(args[i]);
                        break;
                    case "-s":
                        i++;
                        symbolCount = int.Parse(args[i]);
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
                generateMessageFromStr(length, str, outFile);
            }
            else if(type == 1)
            {
                generateMessageEquiprobable(length, symbolCount, outFile);
            }
            else if (type == 2)
            {
                generateMessagePercent(length, percent / 100.0, outFile);
            }
        }

        public static void generateMessageFromStr(int length, string str, string outFile)
        {
            int count = length / str.Length;
            string message = string.Concat(Enumerable.Repeat(str, count));
            message += str.Substring(0, length % str.Length);
            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message);
        }

        public static void generateMessageEquiprobable(int length, int symbolCount, string outFile)
        {
            StringBuilder message = new StringBuilder(length);
            float[] a = new float[symbolCount];
            Random rand = new Random();
            for (int i = 0; i < length; i++)
            {
                int num = rand.Next(symbolCount);
                message.Append("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".ElementAt(num));
                a[num]++;
            }

            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message.ToString());
            Console.WriteLine("Pourcentage d'occurence des symboles :");
            for(int i = 0; i < symbolCount; i++)
            {
                Console.WriteLine($"{"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".ElementAt(i)}: {a[i] / length * 100}%");
            }
        }

        public static void generateMessagePercent(int length, double percent0, string outFile)
        {
            StringBuilder message = new StringBuilder(length);
            float compteurA = 0, compteurB = 0, compteurC = 0, compteurD = 0;
            Random rand = new Random();
            for(int i = 0; i < length; i++)
            {
                double num = rand.NextDouble();
                if(num < percent0)
                {
                    message.Append("A");
                    compteurA++;
                }
                else if(rand.Next(2) == 0)
                {
                    message.Append("B");
                    compteurB++;
                }
                else if (rand.Next(2) == 0)
                {
                    message.Append("C");
                    compteurC++;
                }
                else
                {
                    message.Append("D");
                    compteurD++;
                }
            }

            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message.ToString());
            Console.WriteLine("Pourcentage d'occurence des symboles :");
            Console.WriteLine($"A: {compteurA/length * 100}%, B: {compteurB / length * 100}%, C: {compteurC / length * 100}%, D: {compteurD / length * 100}%");
        }


    }
}
