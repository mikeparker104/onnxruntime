using System;
using Microsoft.ML.OnnxRuntime;

namespace FasterRcnnSample
{
    public static class PlatformSessionOptions
    {
        static Action<SessionOptions> _platformOptionsHandler;

        static Action<SessionOptions> PlatformOptionsHandler => _platformOptionsHandler ??= new Action<SessionOptions>((options) =>
        {
            options.AppendExecutionProvider_CPU();
        });

        public static void SetPlatformOptionsHandler(Action<SessionOptions> handler)
            => _platformOptionsHandler = handler;

        public static void ClearPlatformOptionsHandler()
            => _platformOptionsHandler = null;

        public static SessionOptions Create(SessionOptionMode mode = SessionOptionMode.Default)
            => new SessionOptions().ApplyOptions(mode);

        public static SessionOptions ApplyOptions(this SessionOptions options, SessionOptionMode mode = SessionOptionMode.Default)
        {
            if (mode == SessionOptionMode.Default)
                options.AppendExecutionProvider_CPU();
            else
                PlatformOptionsHandler(options);

            return options;
        }
    }
}