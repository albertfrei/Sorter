# Sorter

Windows sample-library organizer using .NET 10 and WPF.

## Run on Windows

Open `Sorter.slnx` in a Visual Studio version that supports .NET 10 and `.slnx`, with the **.NET desktop development** workload installed. Set **Sorter.App** as the startup project and press F5.

Alternatively, install the .NET 10 SDK, then run from the repository root:

```powershell
dotnet run --project src/Sorter.App
```

Choose a library, scan to preview categories, choose an output folder outside the library, and copy. Originals stay in place. Output uses `category/original-relative-path`, preserving duplicate filenames across packs. Existing output files are never overwritten: a collision stops the operation and earlier copies remain. Use a fresh output folder for each export. Avoid symlink/junction output directories.

Discovery supports WAV, AIFF, FLAC, MP3, OGG and M4A. Classification uses WAV RIFF INFO title, comment, genre and keyword tags first, then filenames and immediate parent-folder labels for kicks, snares, hats, claps, cymbals, percussion, bass/808, vocals, loops and effects. Unknown labels remain Uncategorized. Other embedded tag formats are not read yet. No audio is uploaded.

Playback, manual category correction, and machine learning inference are not implemented yet. `ISampleClassifier` provides an extension point for an audio model; the current classifier is a heuristic baseline.

## Copy access errors

Choose a fresh output folder outside the source library that your Windows account can write to (for example, a new folder under your Music folder). Avoid Windows and Program Files directories. The access-denied dialog identifies the operation and blocked path, and reports how many files finished copying. Check source files can be opened in Explorer and output folders can be written to. If Windows Security reports a Controlled folder access block, allow the trusted Sorter executable or use another output location; do not disable protection globally. Some copies can remain after a failure, including an incomplete file if writing was interrupted. Retry into a fresh folder because existing files are never overwritten.

## Validate

```sh
dotnet run --project tests/Sorter.SmokeTests
dotnet build src/Sorter.App
```

The dependency-free smoke runner checks scanning, classification, copy contents, original preservation, duplicate names, overwrite rejection and nested output rejection. Fixtures test file handling, not audio decoding. Core tests run on Linux or Windows. Linux can cross-build WPF; UI execution and manual validation require Windows.
