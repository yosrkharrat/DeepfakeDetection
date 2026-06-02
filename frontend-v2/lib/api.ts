export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "");

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE_URL) {
    return `${API_BASE_URL}${normalizedPath}`;
  }
  return normalizedPath;
}