using System.Windows;
using Microsoft.Win32;
using Sorter.Core;

namespace Sorter.App;

public partial class MainWindow : Window
{
    private IReadOnlyList<Sample> samples = [];
    public MainWindow() => InitializeComponent();
    private void ChooseSource(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "Choose sample library" };
        if (dialog.ShowDialog() != true) return;
        SourcePath.Text = dialog.FolderName;
        samples = [];
        ShowSamples();
        EmptyTitle.Text = "Ready to discover your samples";
        EmptyDescription.Text = "Scan this library to preview categories before copying.";
        CopyButton.IsEnabled = false;
        Status.Text = "Library selected. Scan to preview categories.";
    }
    private void ChooseDestination(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "Choose output outside the source library" };
        if (dialog.ShowDialog() == true) DestinationPath.Text = dialog.FolderName;
    }
    private void Busy(bool busy)
    {
        SourceButton.IsEnabled = DestinationButton.IsEnabled = ScanButton.IsEnabled = !busy;
        CopyButton.IsEnabled = !busy && samples.Count > 0;
    }
    private void ShowSamples()
    {
        SamplesGrid.ItemsSource = samples.Select(sample => new SampleRow(sample, SourcePath.Text)).ToArray();
        SampleCount.Text = $"{samples.Count:N0} files";
        EmptyState.Visibility = samples.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }
    private async void Scan(object sender, RoutedEventArgs e)
    {
        if (!System.IO.Directory.Exists(SourcePath.Text)) { Status.Text = "Choose a sample library first."; return; }
        Busy(true);
        samples = [];
        ShowSamples();
        EmptyTitle.Text = "Finding your sounds…";
        EmptyDescription.Text = "Reading metadata and matching sample names.";
        Status.Text = "Scanning…";
        try
        {
            var source = SourcePath.Text;
            samples = await Task.Run(() => LibrarySorter.Scan(source, new MetadataClassifier()));
            ShowSamples();
            EmptyTitle.Text = "No samples found";
            EmptyDescription.Text = "Try a folder containing WAV, AIFF, FLAC, MP3, OGG or M4A files.";
            Status.Text = $"{samples.Count} samples · {samples.Count(s => s.Category == "Uncategorized")} uncategorized";
        }
        catch (Exception ex)
        {
            Status.Text = ex.Message;
            EmptyTitle.Text = "Scan could not finish";
            EmptyDescription.Text = "Check the folder is accessible and try again.";
        }
        finally { Busy(false); }
    }
    private async void Copy(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(DestinationPath.Text)) { Status.Text = "Choose an output folder first."; return; }
        Busy(true);
        Status.Text = "Copying…";
        try
        {
            var source = SourcePath.Text;
            var destination = DestinationPath.Text;
            var count = await Task.Run(() => LibrarySorter.Copy(source, destination, samples));
            Status.Text = $"Copied {count} samples. Originals preserved.";
        }
        catch (UnauthorizedAccessException ex)
        {
            Status.Text = "Copy stopped: access denied. See the details for the blocked path.";
            MessageBox.Show(this, ex.Message, "Copy access denied", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        catch (Exception ex)
        {
            Status.Text = "Copy stopped. Some files may already be copied.";
            MessageBox.Show(this, $"{ex.Message}\n\nSome files may already be copied. Originals are unchanged.",
                "Copy could not finish", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        finally { Busy(false); }
    }
}
