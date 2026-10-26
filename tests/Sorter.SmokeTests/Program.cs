using Sorter.Core;
var root = Path.Combine(Path.GetTempPath(), "sorter-" + Guid.NewGuid());
var source = Path.Combine(root, "source");
var output = Path.Combine(root, "output");
int assertions = 0;
void Check(bool value) { if (!value) throw new Exception("Assertion failed: " + assertions); assertions++; }
try {
 Directory.CreateDirectory(Path.Combine(source, "a"));
 Directory.CreateDirectory(Path.Combine(source, "b"));
 foreach (var name in new[] { "a/kick_01.wav", "b/kick_01.wav", "808.wav", "closed_hh.wav", "unknown.flac", "notes.txt" })
  File.WriteAllText(Path.Combine(source, name), "fixture");
 var samples = LibrarySorter.Scan(source, new MetadataClassifier());
 Check(samples.Count == 5);
 Check(samples.Count(s => s.Category == "Kicks") == 2);
 Check(samples.Count(s => s.Category == "Bass") == 1);
 Check(samples.Count(s => s.Category == "Hi Hats") == 1);
 Check(samples.Count(s => s.Category == "Uncategorized") == 1);
 var taggedPath = Path.Combine(root, "tagged.wav");
 using (var stream = File.Create(taggedPath))
 using (var writer = new BinaryWriter(stream)) {
  writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(30u);
  writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVELIST")); writer.Write(18u);
  writer.Write(System.Text.Encoding.ASCII.GetBytes("INFOINAM")); writer.Write(6u);
  writer.Write(System.Text.Encoding.ASCII.GetBytes("snare\0"));
 }
 var tagged = new MetadataClassifier().Classify(taggedPath);
 Check(tagged.Category == "Snares" && tagged.Reason.StartsWith("WAV metadata"));
 File.WriteAllBytes(taggedPath, [82, 73, 70, 70]);
 Check(new MetadataClassifier().Classify(taggedPath).Category == "Uncategorized");
 Check(LibrarySorter.Copy(source, output, samples) == 5);
 Check(File.ReadAllText(Path.Combine(output, "Kicks", "a", "kick_01.wav")) == "fixture");
 Check(File.Exists(Path.Combine(output, "Kicks", "b", "kick_01.wav")));
 Check(Directory.GetFiles(source, "*", SearchOption.AllDirectories).Length == 6);
 try { LibrarySorter.Copy(source, output, samples); throw new Exception("Overwrite accepted"); } catch (IOException) { assertions++; }
 try { LibrarySorter.Copy(source, Path.Combine(source, "sorted"), samples); throw new Exception("Nested output accepted"); } catch (ArgumentException) { assertions++; }
 Console.WriteLine($"PASS: {assertions} assertions: scanning, labels, copy, originals, collisions, nested output.");
} finally { Directory.Delete(root, true); }
