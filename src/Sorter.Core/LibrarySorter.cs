using System.Text.RegularExpressions;

namespace Sorter.Core;

public record Sample(string Path, string Category, string Reason);

public interface ISampleClassifier
{
    Sample Classify(string path);
}

// This baseline uses RIFF INFO metadata and names, not audio inference.
public sealed class MetadataClassifier : ISampleClassifier
{
    private static readonly (string Category, string Pattern)[] Rules =
    [
        ("Hi Hats", @"hi[ _-]?hat|hihat|\bhh\b"),
        ("Kicks", @"\bkicks?\b|\bbd\b"),
        ("Snares", @"\bsnares?\b|\bsd\b"),
        ("Claps", @"\bclaps?\b"),
        ("Cymbals", @"\b(crash|ride|cymbal)s?\b"),
        ("Percussion", @"\b(perc|percussion|tom|shaker|tambourine)s?\b"),
        ("Bass", @"\b(bass|808|sub)\b"),
        ("Vocals", @"\b(vocal|voice|vox|acapella)s?\b"),
        ("Loops", @"\bloops?\b"),
        ("Effects", @"\b(fx|effect|riser|impact|sweep)s?\b")
    ];

    public Sample Classify(string path)
    {
        var labels = WaveMetadata.ReadLabels(path).Select(label => (Text: label, Source: "WAV metadata"))
            .Concat(new[] { (Text: System.IO.Path.GetFileNameWithoutExtension(path), Source: "Filename"),
                (Text: System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(path)) ?? "", Source: "Folder") });
        foreach (var label in labels)
        {
            var normalized = Regex.Replace(label.Text, @"[_\-\d]+", " ");
            // Keep 808 available as a meaningful label.
            if (label.Text.Contains("808")) normalized += " 808";
            foreach (var rule in Rules)
                if (Regex.IsMatch(normalized, rule.Pattern, RegexOptions.IgnoreCase | RegexOptions.CultureInvariant))
                    return new(path, rule.Category, $"{label.Source}: {rule.Category}");
        }
        return new(path, "Uncategorized", "No matching label; needs review or audio classification");
    }
}

public static class LibrarySorter
{
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
        { ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a" };

    public static IReadOnlyList<Sample> Scan(string root, ISampleClassifier classifier, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(root)) throw new DirectoryNotFoundException(root);
        var options = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = false, AttributesToSkip = FileAttributes.ReparsePoint };
        var result = new List<Sample>();
        foreach (var file in Directory.EnumerateFiles(root, "*", options))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Extensions.Contains(Path.GetExtension(file))) result.Add(classifier.Classify(file));
        }
        return result;
    }

    public static int Copy(string source, string destination, IEnumerable<Sample> samples)
    {
        var sourceRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(source));
        var targetRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(destination));
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (targetRoot.Equals(sourceRoot, comparison) || targetRoot.StartsWith(sourceRoot + Path.DirectorySeparatorChar, comparison))
            throw new ArgumentException("Choose an output folder outside the source library.");
        var count = 0;
        foreach (var sample in samples)
        {
            if (sample.Category is "." or ".." || sample.Category.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || sample.Category.Contains('/') || sample.Category.Contains('\\'))
                throw new ArgumentException("Invalid category.");
            var relative = Path.GetRelativePath(sourceRoot, Path.GetFullPath(sample.Path));
            if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar))
                throw new ArgumentException("Sample is outside the source library.");
            // Retain source structure to preserve identically named samples.
            var target = Path.Combine(targetRoot, sample.Category, relative);
            var operation = "create output folder";
            var blockedPath = Path.GetDirectoryName(target)!;
            try
            {
                Directory.CreateDirectory(blockedPath);
                operation = "read source file";
                blockedPath = sample.Path;
                using var input = new FileStream(sample.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
                operation = "create output file";
                blockedPath = target;
                using var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                operation = "copy file data";
                input.CopyTo(output);
            }
            catch (UnauthorizedAccessException ex)
            {
                throw new UnauthorizedAccessException(
                    $"Access denied while trying to {operation}:\n{blockedPath}\n\n" +
                    $"{count} files copied before stopping. Originals are unchanged.\n" +
                    "Choose a fresh output folder you can write to, outside the source library. " +
                    "If the source is blocked, check that you can open it in Windows Explorer. " +
                    "If Windows Security reports a Controlled folder access block, allow Sorter only if you trust it, or choose another output location.\n\n" +
                    $"Windows details: {ex.Message}", ex);
            }
            count++;
        }
        return count;
    }
}
