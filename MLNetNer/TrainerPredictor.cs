using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using MLNetNer.Models;
using static Microsoft.ML.Transforms.Text.TextNormalizingEstimator;

namespace MLNetNer
{
   internal class TrainerPredictor
   {
      private readonly bool _mock = true;
      private readonly MLContext _mlContext;
      private readonly Label[] _labels;
      private readonly int BATCH_SIZE = 32;
      private readonly int MAX_EPOCHS = 2;

      public TrainerPredictor(bool mock)
      {
         _mock = mock;
         _mlContext = new MLContext();

         if (_mock)
         {
            _labels = GetMockLabels();
         }
         else
         {
            _labels = GetRealLabels();
         }

      }

      public void Train()
      {
         List<ModelInput> trainingData = [];

         if (_mock)
         {
            trainingData = GetMockData();
         }
         else
         {
            trainingData = GetRealData();
         }

         var trainingDataView = _mlContext.Data.LoadFromEnumerable(trainingData);

         var dataSplit = _mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

         var pipeline = BuildPipeline();

         // var estimatorSchema = pipeline.GetOutputSchema(SchemaShape.Create(trainingDataView.Schema));

         var trainData = dataSplit.TrainSet;
         var testData = dataSplit.TestSet;

         // Training the model
         ITransformer trainedModel = pipeline.Fit(trainData);

         // Saving the trained model
         _mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Path.Combine(AppContext.BaseDirectory, "Data", @"ner-model.zip"));
      }

      public void Predict()
      {
         var trainedModel = _mlContext.Model.Load(Path.Combine(AppContext.BaseDirectory, "Data", @"ner-model.zip"), out _);

         var engine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

         ModelInput sampleData = new()
         {
            Sentence = @"Alice and Bob live in London Uk",
         };

         var predictionResult = engine.Predict(sampleData);

         Console.WriteLine($"\n\nPredicted labels Label: {String.Join(", ", predictionResult.PredictedLabel)}\n\n");

         var values = sampleData.Sentence.Split(' ').Select((val, index) =>
         {
            return $"{val} => {predictionResult.PredictedLabel[index]}";
         })
         .ToList();

         values.ForEach(x =>
         {
            Console.WriteLine("{0}", x);
         });
      }

      private IEstimator<ITransformer> BuildPipeline()
      {
         var labels = _mlContext.Data.LoadFromEnumerable(_labels);


         //var chain = new EstimatorChain<ITransformer>();
         //var pipeline = chain.Append(mlContext.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
         //   .Append(mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(labelColumnName: @"Label", outputColumnName: @"predicted_label", sentence1ColumnName: @"Sentence"))
         //   .Append(mlContext.Transforms.Conversion.MapKeyToValue("predicted_label"));

         //return pipeline;

         //// Data process configuration with pipeline data transformations
         var pipeline = _mlContext.Transforms.Text.NormalizeText(inputColumnName: @"Sentence", outputColumnName: @"Sentence", caseMode: CaseMode.None, keepDiacritics: false, keepPunctuations: false, keepNumbers: true)
                                 .Append(_mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label", addKeyValueAnnotationsAsText: false, keyData: labels))
                                 .Append(_mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(labelColumnName: @"Label", outputColumnName: @"PredictedLabel", sentence1ColumnName: @"Sentence", batchSize: BATCH_SIZE, maxEpochs: MAX_EPOCHS))
                                 .Append(_mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

         return pipeline;
      }

      private static Label[] GetMockLabels()
      {
         return [
             new Label { Key = "PERSON" },
                new Label { Key = "CITY" },
                new Label { Key = "COUNTRY"  }
             ];
      }

      private static Label[] GetRealLabels()
      {
         var lines = File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Data", @"ner-key-info.txt"));

         return lines.Where(line => !string.IsNullOrWhiteSpace(line)).Select(line => new Label
         {
            Key = line
         })
         .ToArray();
      }

      private static List<ModelInput> GetMockData()
      {
         var trainingData = new List<ModelInput>();

         for (int i = 0; i < 1500; i++)
         {
            trainingData.Add(new ModelInput
            {
               Sentence = "Alice and Bob live in London with a dog",
               Label = ["PERSON", "0", "PERSON", "0", "0", "CITY", "0", "0", "0"]
            });
         }

         return trainingData;
      }

      private static List<ModelInput> GetRealData()
      {
         var lines = File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Data", @"ner-conll2012_english_v4_train.txt"));

         return lines.Where(line => !string.IsNullOrWhiteSpace(line)).Select(line =>
         {
            var values = line.Split('\t');

            return new ModelInput
            {
               Sentence = values[0],
               Label = values.Skip(1).ToArray()
            };
         })
         .ToList();
      }
   }
}
