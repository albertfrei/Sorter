using System.IO;
using Sorter.Core;

namespace Sorter.App;

// Display paths are relative; file operations keep using the original Sample.
public sealed class SampleRow
{
    private readonly Sample sample;
    public SampleRow(Sample sample, string library)
    {
        this.sample = sample;
        RelativePath = Path.GetRelativePath(library, sample.Path);
    }
    public string FileName => Path.GetFileName(sample.Path);
    public string RelativePath { get; }
    public string Folder => Path.GetDirectoryName(RelativePath) is { Length: > 0 } folder ? folder : "Library root";
    public string Category => sample.Category;
    public string Reason => sample.Reason;
    public string MatchSource => Category == "Uncategorized" ? "Needs review" : Reason.Split(':')[0];
    public string BadgeBackground => Category switch
    {
        "Kicks" or "Bass" => "#352A48",
        "Snares" or "Claps" => "#3D2938",
        "Hi Hats" or "Cymbals" => "#3B3323",
        "Percussion" or "Loops" => "#203A36",
        "Vocals" or "Effects" => "#24354A",
        _ => "#30343D"
    };
    public string BadgeForeground => Category switch
    {
        "Kicks" or "Bass" => "#CEB1FF",
        "Snares" or "Claps" => "#F3A7CD",
        "Hi Hats" or "Cymbals" => "#EBCB85",
        "Percussion" or "Loops" => "#89DCC8",
        "Vocals" or "Effects" => "#9CC9FF",
        _ => "#B8C0CF"
    };
}
