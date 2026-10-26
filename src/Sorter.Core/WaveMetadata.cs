using System.Text;

namespace Sorter.Core;

internal static class WaveMetadata
{
    // Read bounded RIFF INFO text chunks without decoding or loading the audio data.
    public static IEnumerable<string> ReadLabels(string path)
    {
        if (!Path.GetExtension(path).Equals(".wav", StringComparison.OrdinalIgnoreCase)) yield break;
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8);
        if (stream.Length < 12 || new string(reader.ReadChars(4)) != "RIFF") yield break;
        reader.ReadUInt32();
        if (new string(reader.ReadChars(4)) != "WAVE") yield break;
        while (stream.Position + 8 <= stream.Length)
        {
            var id = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var size = reader.ReadUInt32();
            var end = stream.Position + size;
            if (end > stream.Length) yield break;
            if (id == "LIST" && size >= 4 && Encoding.ASCII.GetString(reader.ReadBytes(4)) == "INFO")
            {
                while (stream.Position + 8 <= end)
                {
                    var tag = Encoding.ASCII.GetString(reader.ReadBytes(4));
                    var length = reader.ReadUInt32();
                    var next = stream.Position + length;
                    if (next > end) break;
                    if (length <= 65536 && tag is "INAM" or "ICMT" or "IGNR" or "IKEY")
                        yield return Encoding.UTF8.GetString(reader.ReadBytes((int)length)).TrimEnd('\0');
                    stream.Position = next + (length % 2);
                }
            }
            stream.Position = end + (size % 2);
        }
    }
}
