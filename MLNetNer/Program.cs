namespace MLNetNer
{
   internal class Program
   {
      static void Main()
      {
         var ml = new TrainerPredictor(mock: false);

         // ml.Train();
         ml.Predict();

         Console.ReadKey();
      }

   }
}
