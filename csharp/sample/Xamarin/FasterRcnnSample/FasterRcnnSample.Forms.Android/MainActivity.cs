using Android.App;
using Android.Content.PM;
using Android.Runtime;
using Android.OS;

namespace FasterRcnnSample.Forms.Droid
{
    [Activity(Label = "FasterRcnnSample.Forms", Icon = "@mipmap/icon", Theme = "@style/MainTheme", MainLauncher = true, ScreenOrientation = ScreenOrientation.Portrait | ScreenOrientation.ReversePortrait, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode | ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize)]
    public class MainActivity : global::Xamarin.Forms.Platform.Android.FormsAppCompatActivity
    {
        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            Xamarin.Essentials.Platform.Init(this, savedInstanceState);
            global::Xamarin.Forms.Forms.Init(this, savedInstanceState);

            PlatformSessionOptions.SetPlatformOptionsHandler((options)
                => options.AppendExecutionProvider_Nnapi(Microsoft.ML.OnnxRuntime.NnapiFlags.NNAPI_FLAG_USE_NONE));

            LoadApplication(new App());
        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, [GeneratedEnum] Permission[] grantResults)
        {
            Xamarin.Essentials.Platform.OnRequestPermissionsResult(requestCode, permissions, grantResults);

            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}