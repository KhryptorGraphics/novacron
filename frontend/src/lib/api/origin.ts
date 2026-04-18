const DEFAULT_API_ORIGIN = 'http://localhost:8090';

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

function originFromUrl(value: string): string {
  try {
    const url = new URL(value);
    return `${url.protocol}//${url.host}`;
  } catch {
    return trimTrailingSlash(value);
  }
}

function toWebSocketOrigin(value: string): string {
  return value.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
}

const configuredApiOrigin = process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_ORIGIN;
const configuredWsOrigin = process.env.NEXT_PUBLIC_WS_URL;

export const API_ORIGIN = originFromUrl(configuredApiOrigin);
export const API_V1_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || `${API_ORIGIN}/api/v1`;
export const WS_ORIGIN = configuredWsOrigin
  ? originFromUrl(configuredWsOrigin)
  : toWebSocketOrigin(API_ORIGIN);

function normalizePath(path: string): string {
  if (!path.startsWith('/')) {
    return `/${path}`;
  }

  return path;
}

function replacePrefix(path: string, source: string, target: string): string | null {
  if (!path.startsWith(source)) {
    return null;
  }

  return `${target}${path.slice(source.length)}`;
}

export function buildApiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }

  return new URL(normalizePath(path), `${API_ORIGIN}/`).toString();
}

export function buildWebSocketUrls(path: string): string[] {
  if (/^wss?:\/\//.test(path)) {
    return [path];
  }

  const normalized = normalizePath(path);
  const candidates = new Set<string>([normalized]);

  switch (true) {
    case normalized === '/api/ws/monitoring':
      candidates.add('/api/ws/metrics');
      candidates.add('/ws/metrics');
      break;
    case normalized === '/api/ws/security':
      candidates.add('/api/ws/alerts');
      candidates.add('/ws/alerts');
      break;
    case normalized === '/api/ws/security/events':
      candidates.add('/api/security/events/stream');
      break;
    case normalized === '/api/security/events/stream':
      candidates.add('/api/ws/security/events');
      break;
    case normalized.startsWith('/api/ws/alerts'):
      candidates.add(normalized.replace('/api/ws/alerts', '/ws/alerts'));
      break;
    case normalized.startsWith('/api/ws/metrics'):
      candidates.add(normalized.replace('/api/ws/metrics', '/ws/metrics'));
      break;
    case normalized === '/api/ws/logs':
      candidates.add('/ws/logs');
      break;
  }

  const consoleFallback = replacePrefix(normalized, '/api/ws/console/', '/ws/console/');
  if (consoleFallback) {
    candidates.add(consoleFallback);
  }

  const logsFallback = replacePrefix(normalized, '/api/ws/logs/', '/ws/logs/');
  if (logsFallback) {
    candidates.add(logsFallback);
  }

  return Array.from(candidates).map((candidate) =>
    new URL(candidate, `${WS_ORIGIN}/`).toString(),
  );
}
