using System;
using System.IO;
using System.Threading.Tasks;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace FasterRcnnSample.Forms
{
    enum ImageAcquisitionMode
    {
        Sample,
        Capture,
        Pick
    }

    public partial class MainPage : ContentPage
    {
        FasterRcnnObjectDetector _objectDetector;
        FasterRcnnObjectDetector ObjectDetector => _objectDetector ??= new FasterRcnnObjectDetector();

        public MainPage()
        {
            InitializeComponent();

            ImageProcessorOptions.Items.Add(nameof(ImageProcessor.ImageSharp));
            ImageProcessorOptions.Items.Add(nameof(ImageProcessor.SkiaSharp));
            ImageProcessorOptions.SelectedIndex = 1;

            SessionOptionModes.Items.Add(nameof(SessionOptionMode.Default));
            SessionOptionModes.Items.Add(nameof(SessionOptionMode.Platform));
            SessionOptionModes.SelectedIndex = 1;
        }

        async Task AcquireAndAnalyzeImageAsync(ImageAcquisitionMode acquisitionMode = ImageAcquisitionMode.Sample)
        {
            byte[] outputImage = null;

            try
            {
                SetBusyState(true);

                var imageData = acquisitionMode switch
                {
                    ImageAcquisitionMode.Capture => await TakePhotoAsync(),
                    ImageAcquisitionMode.Pick => await PickPhotoAsync(),
                    _ => await GetSampleImageAsync()
                };

                if (imageData == null)
                {
                    SetBusyState(false);
                    return;
                }

                ClearResult();

                var imageProcessor = ImageProcessorOptions.SelectedItem switch
                {
                    nameof(ImageProcessor.ImageSharp) => ImageProcessor.ImageSharp,
                    _ => ImageProcessor.SkiaSharp
                };

                var sessionOptionMode = SessionOptionModes.SelectedItem switch
                {
                    nameof(SessionOptionMode.Default) => SessionOptionMode.Default,
                    _ => SessionOptionMode.Platform
                };

                outputImage = await ObjectDetector.GetImageWithObjectsAsync(imageData, imageProcessor, sessionOptionMode);
            }
            finally
            {
                SetBusyState(false);
            }

            if (outputImage != null)
                ShowResult(outputImage);
        }

        Task<byte[]> GetSampleImageAsync() => Task.Run(() =>
        {
            var assembly = GetType().Assembly;

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.demo.jpg");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            var sampleImage = memoryStream.ToArray();

            return sampleImage;
        });

        async Task<byte[]> PickPhotoAsync()
        {
            FileResult photo;

            try
            {
                photo = await MediaPicker.PickPhotoAsync(new MediaPickerOptions { Title = "Choose photo" });
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(PickPhotoAsync)} method throw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);

            return bytes;
        }

        async Task<byte[]> TakePhotoAsync()
        {
            FileResult photo;

            try
            {
                photo = await MediaPicker.CapturePhotoAsync(new MediaPickerOptions { Title = "Take photo" });
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(TakePhotoAsync)} method throw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);

            return bytes;
        }

        async Task<byte[]> GetBytesFromPhotoFile(FileResult fileResult)
        {
            byte[] bytes;

            using Stream stream = await fileResult.OpenReadAsync();
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            bytes = ms.ToArray();

            return bytes;
        }

        void ClearResult()
            => MainThread.BeginInvokeOnMainThread(() => OutputImage.Source = null);

        void ShowResult(byte[] image)
            => MainThread.BeginInvokeOnMainThread(() => OutputImage.Source = ImageSource.FromStream(() => new MemoryStream(image)));

        void SetBusyState(bool busy)
        {
            ImageProcessorOptions.IsEnabled = !busy;
            SessionOptionModes.IsEnabled = !busy;
            SamplePhotoButton.IsEnabled = !busy;
            PickPhotoButton.IsEnabled = !busy;
            TakePhotoButton.IsEnabled = !busy;
            BusyIndicator.IsEnabled = busy;
            BusyIndicator.IsRunning = busy;
        }

        ImageAcquisitionMode GetAcquisitioinModeFromText(string tag) => tag switch
        {
            nameof(ImageAcquisitionMode.Capture) => ImageAcquisitionMode.Capture,
            nameof(ImageAcquisitionMode.Pick) => ImageAcquisitionMode.Pick,
            _ => ImageAcquisitionMode.Sample
        };

        void AcquireButton_Clicked(object sender, EventArgs e)
            => AcquireAndAnalyzeImageAsync(GetAcquisitioinModeFromText((sender as Button).Text)).ContinueWith((task)
                => { if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(()
                  => DisplayAlert("Error", task.Exception.Message, "OK")); });
    }
}