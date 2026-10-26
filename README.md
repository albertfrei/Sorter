# Sorter

Windows sample-library organizer using .NET 10 and WPF.

## Run on Windows

Install the .NET 10 SDK, then run from the repository root:

```powershell
dotnet run --project src/Sorter.App
```

Choose a library, scan to preview categories, choose an output folder outside the library, and copy. Originals stay in place. Output uses `category/original-relative-path`, preserving duplicate filenames across packs. Existing output files are never overwritten: a collision stops the operation and earlier copies remain. Use a fresh output folder for each export. Avoid symlink/junction output directories.

Discovery supports WAV, AIFF, FLAC, MP3, OGG and M4A. Classification uses WAV RIFF INFO title, comment, genre and keyword tags first, then filenames and immediate parent-folder labels for kicks, snares, hats, claps, cymbals, percussion, bass/808, vocals, loops and effects. Unknown labels remain Uncategorized. Other embedded tag formats are not read yet. No audio is uploaded.

Playback, manual category correction, and machine learning inference are not implemented yet. `ISampleClassifier` provides an extension point for an audio model; the current classifier is a heuristic baseline.

## Validate

```sh
dotnet run --project tests/Sorter.SmokeTests
dotnet build src/Sorter.App
```

The dependency-free smoke runner checks scanning, classification, copy contents, original preservation, duplicate names, overwrite rejection and nested output rejection. Fixtures test file handling, not audio decoding. Core tests run on Linux or Windows. Linux can cross-build WPF; UI execution and manual validation require Windows.
