using Microsoft.ML.Data;

namespace MLNetNer.Models
{
   internal class ModelOutput
   {
      [ColumnName(@"Sentence")]
      public string Sentence { get; set; } = string.Empty;

      [ColumnName(@"Label")]
      public uint[] Label { get; set; } = [];

      [ColumnName(@"PredictedLabel")]
      public string[] PredictedLabel { get; set; } = [];
   }
}
