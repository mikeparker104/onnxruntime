using Xamarin.UITest;

namespace EndToEndTests.Mobile.Automation
{
    public class AppInitializer
    {
        public static IApp StartApp(Platform platform)
        {
            // For local testing of the Android test app, run a command line Debug build first
            // msbuild <path_to_repo>/onnxruntime/csharp/test/Microsoft.ML.OnnxRuntime.Tests.Droid/Microsoft.ML.OnnxRuntime.Tests.Droid.csproj /p:Configuration=Release /t:PackageForAndroid
            if (platform == Platform.Android)
            {
                return ConfigureApp
                    .Android
#if DEBUG
                    .EnableLocalScreenshots()
                    .ApkFile("../../../../../Microsoft.ML.OnnxRuntime.Tests.Droid/bin/Release/com.xamcat.microsoft_ml_onnxruntime_tests_droid.apk")
#endif
                    .StartApp();
            }

            // For local testing of the iOS test app, install to a physical iPhone device first
            return ConfigureApp
                .iOS
#if DEBUG
                .EnableLocalScreenshots()
                .InstalledApp(bundleId: "com.xamcat.Microsoft-ML-OnnxRuntime-Tests-iOS")
#endif
                .StartApp();
        }
    }
}
