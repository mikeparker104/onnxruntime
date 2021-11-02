using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FasterRcnnSample
{
    public class SkiaSharpImageProcessor : IImageProcessor<SKBitmap>
    {
        public byte[] ApplyPredictionsToImage(IList<Prediction> predictions, SKBitmap image)
        {
            // Put boxes, labels and confidence on image and save for viewing
            using SKSurface surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            using SKCanvas canvas = surface.Canvas;
            using SKPaint textPaint = new SKPaint { TextSize = 32, Color = SKColors.White };
            using SKPaint rectPaint = new SKPaint { StrokeWidth = 2f, IsStroke = true, Color = SKColors.Red };

            canvas.DrawBitmap(image, 0, 0);

            foreach (var p in predictions)
            {
                var text = $"{p.Label}, {p.Confidence:0.00}";
                var textBounds = new SKRect();
                textPaint.MeasureText(text, ref textBounds);

                canvas.DrawRect(p.Box.Xmin, p.Box.Ymin, p.Box.Xmax - p.Box.Xmin, p.Box.Ymax - p.Box.Ymin, rectPaint);
                canvas.DrawText($"{p.Label}, {p.Confidence:0.00}", p.Box.Xmin, p.Box.Ymin + textBounds.Height, textPaint);
            }

            canvas.Flush();
            using var snapshot = surface.Snapshot();
            using var imageData = snapshot.Encode(SKEncodedImageFormat.Jpeg, 100);
            byte[] bytes = imageData.ToArray();

            return bytes;
        }

        public Tensor<float> GetTensorForImage(SKBitmap image)
        {
            var bytes = image.GetPixelSpan();
            var height = image.Height;
            var width = image.Width;
            var paddedHeight = (int)(Math.Ceiling(height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(width / 32f) * 32f);
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
            var outputLength = paddedHeight * paddedWidth * 3;
            var input = new float[outputLength];
            var bytesPerPixel = image.BytesPerPixel;
            var channelLength = outputLength / 3; 
            var gOffset = channelLength;
            var rOffset = channelLength * 2;

            for (int y = paddedHeight - height; y < height; y++)
            {
                for (int x = paddedWidth - width; x < width; x++)
                {
                    var pixelStartIndex = ((y * width) + x) * bytesPerPixel;
                    var r = bytes[pixelStartIndex];
                    var g = bytes[pixelStartIndex + 1];
                    var b = bytes[pixelStartIndex + 2];

                    var destinationIndex = (y * paddedWidth) + x;
                    input[destinationIndex] = b - mean[0];
                    input[destinationIndex + gOffset] = g - mean[1];
                    input[destinationIndex + rOffset] = r - mean[2];
                }
            }

            return new DenseTensor<float>(new Memory<float>(input), new[] { 3, paddedHeight, paddedWidth });
        }

        public SKBitmap PreprocessSourceImage(byte[] sourceImage)
        {
            // Read image
            using var image = SKBitmap.Decode(sourceImage);

            // Resize image
            float ratio = 800f / Math.Min(image.Width, image.Height);
            var scaledImage = image.Resize(new SKImageInfo((int)(ratio * image.Width), (int)(ratio * image.Height)), SKFilterQuality.Medium);

            // Handle orientation
            // See: https://github.com/mono/SkiaSharp/issues/1551#issuecomment-756685252
            using var memoryStream = new MemoryStream(sourceImage);
            using var imageData = SKData.Create(memoryStream);
            using var codec = SKCodec.Create(imageData);
            var orientation = codec.EncodedOrigin;

            return HandleOrientation(scaledImage, orientation);
        }

        // Address issue with orientation rotation
        // See: https://stackoverflow.com/questions/44181914/iphone-image-orientation-wrong-when-resizing-with-skiasharp
        SKBitmap HandleOrientation(SKBitmap bitmap, SKEncodedOrigin orientation)
        {
            switch (orientation)
            {
                case SKEncodedOrigin.BottomRight:

                    using (var surface = new SKCanvas(bitmap))
                    {
                        surface.RotateDegrees(180, bitmap.Width / 2, bitmap.Height / 2);
                        surface.DrawBitmap(bitmap.Copy(), 0, 0);
                    }

                    return bitmap;

                case SKEncodedOrigin.RightTop:

                    using (var rotated = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var surface = new SKCanvas(rotated))
                        {
                            surface.Translate(rotated.Width, 0);
                            surface.RotateDegrees(90);
                            surface.DrawBitmap(bitmap, 0, 0);
                        }

                        rotated.CopyTo(bitmap);
                        return bitmap;
                    }

                case SKEncodedOrigin.LeftBottom:

                    using (var rotated = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var surface = new SKCanvas(rotated))
                        {
                            surface.Translate(0, rotated.Height);
                            surface.RotateDegrees(270);
                            surface.DrawBitmap(bitmap, 0, 0);
                        }

                        rotated.CopyTo(bitmap);
                        return bitmap;
                    }

                default:
                    return bitmap;
            }
        }
    }
}