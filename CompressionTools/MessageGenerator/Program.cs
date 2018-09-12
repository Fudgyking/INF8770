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
            int length = 0, ghetto = 0, percent = 0;
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
                    case "-ghetto":
                        i++;
                        ghetto = int.Parse(args[i]);
                        break;
                    case "p":
                        i++;
                        percent = int.Parse(args[i]);
                        break;
                    default:
                        break;
                }
            }

            if(ghetto == 0)
            {
                generateMessage(length, str, outFile);
            }
            else if(ghetto == 1)
            {
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
                int num = rand.Next();
                if(num == 0)
                {
                    compteur++;
                }
                message.Append(num);
            }

            File.WriteAllText(Directory.GetCurrentDirectory() + "\\" + outFile, message.ToString());
            Console.WriteLine("Pourcentage d'occurence des symboles binaires:");
            Console.WriteLine($"0: {compteur/length * 100}%, 1: {(length - compteur)/length * 100}%");
        }


    }
}
