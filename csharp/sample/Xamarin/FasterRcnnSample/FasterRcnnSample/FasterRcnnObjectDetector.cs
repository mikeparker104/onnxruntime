using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FasterRcnnSample
{
    public enum ImageProcessor
    {
        ImageSharp,
        SkiaSharp
    }

    public enum SessionOptionMode
    {
        Default,
        Platform
    }

    public class FasterRcnnObjectDetector
    {
        byte[] _model;
        Task _initializeTask;
        ImageSharpImageProcessor _imageSharpProcessor;
        SkiaSharpImageProcessor _skiaSharpProcessor;

        ImageSharpImageProcessor ImageSharpProcessor => _imageSharpProcessor ??= new ImageSharpImageProcessor();
        SkiaSharpImageProcessor SkiaSharpProcessor => _skiaSharpProcessor ??= new SkiaSharpImageProcessor();

        public async Task<byte[]> GetImageWithObjectsAsync(byte[] sourceImage, ImageProcessor processor = ImageProcessor.ImageSharp, SessionOptionMode sessionOptionMode = SessionOptionMode.Default)
        {
            byte[] outputImage = null;

            await InitializeAsync().ConfigureAwait(false);
            // Make output image zoomable: https://github.com/xamarin/xamarin-forms-samples/blob/main/WorkingWithGestures/PinchGesture/PinchGesture/HomePage.xaml

            if (processor == ImageProcessor.ImageSharp)
            {
                using var preprocessedImage = await Task.Run(() => ImageSharpProcessor.PreprocessSourceImage(sourceImage)).ConfigureAwait(false);
                var tensor = await Task.Run(() => ImageSharpProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
                var predictions = await Task.Run(() => GetPredictions(tensor)).ConfigureAwait(false);
                outputImage = await Task.Run(() => ImageSharpProcessor.ApplyPredictionsToImage(predictions, preprocessedImage)).ConfigureAwait(false);
            }
            else
            {
                using var preprocessedImage = await Task.Run(() => SkiaSharpProcessor.PreprocessSourceImage(sourceImage)).ConfigureAwait(false);
                var tensor = await Task.Run(() => SkiaSharpProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
                var predictions = await Task.Run(() => GetPredictions(tensor)).ConfigureAwait(false);
                outputImage = await Task.Run(() => SkiaSharpProcessor.ApplyPredictionsToImage(predictions, preprocessedImage)).ConfigureAwait(false);
            }

            return outputImage;
        }

        List<Prediction> GetPredictions(Tensor<float> input, SessionOptionMode sessionOptionsMode = SessionOptionMode.Default)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", input) };

            // Run inference
            using var session = new InferenceSession(_model, PlatformSessionOptions.Create(sessionOptionsMode));
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
            float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
            var predictions = new List<Prediction>();
            var minConfidence = 0.7f;

            for (int i = 0; i < boxes.Length - 4; i += 4)
            {
                var index = i / 4;

                if (confidences[index] >= minConfidence)
                {
                    predictions.Add(new Prediction
                    {
                        Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
                        Label = LabelMap.Labels[labels[index]],
                        Confidence = confidences[index]
                    });
                }
            }

            return predictions;
        }

        Task InitializeAsync()
        {
            if (_initializeTask == null || _initializeTask.IsFaulted)
                _initializeTask = Task.Run(() => Initialize());

            return _initializeTask;
        }

        void Initialize()
        {
            var assembly = GetType().Assembly;

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.faster_rcnn.onnx");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            _model = memoryStream.ToArray();
        }
    }
}