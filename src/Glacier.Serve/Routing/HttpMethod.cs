namespace Glacier.Serve.Routing;

public enum HttpMethod : byte
{
    Unknown = 0,
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options
}
