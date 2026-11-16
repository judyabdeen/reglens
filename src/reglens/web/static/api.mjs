// Talking to the RegLens server. Every failure becomes an ApiError with a plain-words message the page can show.

export class ApiError extends Error {
  constructor(kind, message, extra = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.field = extra.field ?? null;
    this.fix = extra.fix ?? null;
    this.status = extra.status ?? 0;
  }
}

const UNREACHABLE =
  "RegLens could not be reached. Check that the server window is still open (start it with: python scripts/serve.py), then try again.";

async function request(path, init) {
  let response;
  try {
    response = await fetch(path, { cache: "no-store", credentials: "same-origin", ...init });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new ApiError("unreachable", UNREACHABLE);
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    // not JSON: handled below
  }
  if (!response.ok || data === null) {
    const detail = data?.error;
    if (detail?.message) throw new ApiError(detail.kind ?? "error", detail.message, { field: detail.field, fix: detail.fix, status: response.status });
    throw new ApiError("server", `The server answered with an unexpected response (status ${response.status}). The server window shows the details.`, { status: response.status });
  }
  return data;
}

export const getJson = (path, options = {}) => request(path, { method: "GET", signal: options.signal });

export const postJson = (path, body, options = {}) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: options.signal });
