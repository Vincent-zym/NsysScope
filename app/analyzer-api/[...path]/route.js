const METHODS_WITHOUT_BODY = new Set(["GET", "HEAD"]);

async function proxy(request, context) {
  const base = process.env.NSYSSCOPE_ANALYZER_INTERNAL_URL;
  const token = process.env.NSYSSCOPE_INTERNAL_TOKEN;
  if (!base || !token) {
    return Response.json({
      detail: "本页面未连接本地 Analyzer。请通过 ./nsysscope start 启动本地工具。",
    }, { status: 503 });
  }

  const params = await context.params;
  const suffix = (params.path || []).map(encodeURIComponent).join("/");
  const target = new URL(`${base.replace(/\/$/, "")}/${suffix}`);
  target.search = new URL(request.url).search;

  const headers = new Headers();
  for (const name of ["accept", "content-type"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("X-NsysScope-Token", token);

  let body;
  if (!METHODS_WITHOUT_BODY.has(request.method)) {
    const bytes = await request.arrayBuffer();
    if (bytes.byteLength) body = bytes;
  }

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
    });
    const responseHeaders = new Headers();
    for (const name of ["content-type", "content-disposition"]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch (cause) {
    return Response.json({
      detail: `无法连接本地 Analyzer：${cause.message}`,
    }, { status: 502 });
  }
}

export const dynamic = "force-dynamic";
export const GET = proxy;
export const POST = proxy;
