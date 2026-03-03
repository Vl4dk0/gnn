export class HttpError extends Error {
  public readonly status: number;

  public constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type RequestInitJson = Omit<RequestInit, "body"> & {
  body?: unknown;
};

const parseError = async (response: Response): Promise<string> => {
  try {
    const data = (await response.json()) as { error?: string };
    return data.error ?? `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
};

export const fetchJson = async <T>(url: string, init?: RequestInitJson): Promise<T> => {
  const headers = new Headers(init?.headers ?? {});

  let body = init?.body;
  if (body !== undefined && body !== null && typeof body !== "string") {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(body);
  }

  const response = await fetch(url, {
    ...init,
    headers,
    body: body as BodyInit | null | undefined
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new HttpError(response.status, message);
  }

  return (await response.json()) as T;
};
