using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FasterRcnnSample
{
    public class ImageSharpImageProcessor : IImageProcessor<Image<Rgb24>>
    {
        public byte[] ApplyPredictionsToImage(IList<Prediction> predictions, Image<Rgb24> image)
        {
            byte[] bytes = null;

            // Put boxes, labels and confidence on image and save for viewing
            var font = ResolveSystemFont(32);

            foreach (var p in predictions)
            {
                image.Mutate(x =>
                {
                    x.DrawLines(Color.Red, 2f, new PointF[] {

                        new PointF(p.Box.Xmin, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymin),

                        new PointF(p.Box.Xmax, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymax),

                        new PointF(p.Box.Xmax, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymax),

                        new PointF(p.Box.Xmin, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymin)
                    });

                    x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, Color.White, new PointF(p.Box.Xmin, p.Box.Ymin));
                });
            }

            using var memoryStream = new MemoryStream();
            image.Save(memoryStream, new JpegEncoder());
            bytes = memoryStream.ToArray();

            return bytes;
        }

        public Tensor<float> GetTensorForImage(Image<Rgb24> image)
        {
            var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);

            Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };

            for (int y = paddedHeight - image.Height; y < image.Height; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);

                for (int x = paddedWidth - image.Width; x < image.Width; x++)
                {
                    input[0, y, x] = pixelSpan[x].B - mean[0];
                    input[1, y, x] = pixelSpan[x].G - mean[1];
                    input[2, y, x] = pixelSpan[x].R - mean[2];
                }
            }

            return input;
        }

        public Image<Rgb24> PreprocessSourceImage(byte[] sourceImage)
        {
            // Read image
            var image = Image.Load<Rgb24>(sourceImage);

            // Resize image
            float ratio = 800f / Math.Min(image.Width, image.Height);
            image.Mutate(x => x.Resize((int)(ratio * image.Width), (int)(ratio * image.Height)));

            return image;
        }

        Font ResolveSystemFont(float size)
        {
            if (SystemFonts.Families.Any())
                return SystemFonts.CreateFont(SystemFonts.Families.FirstOrDefault().Name, size);

            // Android workaround for:
            // https://github.com/SixLabors/Fonts/issues/118
            using var assetStreamReader = new StreamReader("/system/fonts/Roboto-Regular.ttf");
            using var ms = new MemoryStream();
            assetStreamReader.BaseStream.CopyTo(ms);
            ms.Position = 0;
            var fontFamily = new FontCollection().Install(ms);

            return fontFamily.CreateFont(size);
        }
    }
}