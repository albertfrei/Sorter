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
        SamplesGrid.ItemsSource = samples;
        CopyButton.IsEnabled = false;
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
    private async void Scan(object sender, RoutedEventArgs e)
    {
        Busy(true);
        Status.Text = "Scanning…";
        try
        {
            var source = SourcePath.Text;
            samples = await Task.Run(() => LibrarySorter.Scan(source, new MetadataClassifier()));
            SamplesGrid.ItemsSource = samples;
            Status.Text = $"{samples.Count} samples · {samples.Count(s => s.Category == "Uncategorized")} uncategorized";
        }
        catch (Exception ex) { Status.Text = ex.Message; }
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
        catch (Exception ex) { Status.Text = $"Copy stopped; some files may already be copied. {ex.Message}"; }
        finally { Busy(false); }
    }
}
