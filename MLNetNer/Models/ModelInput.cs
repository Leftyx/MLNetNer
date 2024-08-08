using Microsoft.ML.Data;

namespace MLNetNer.Models
{
   internal class ModelInput
   {
      [LoadColumn(0)]
      [ColumnName(@"Sentence")]
      public string Sentence { get; set; } = string.Empty;

      [LoadColumn(1)]
      [ColumnName(@"Label")]
      public string[] Label { get; set; } = [];
   }
}
