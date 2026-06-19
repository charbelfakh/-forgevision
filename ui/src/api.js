/**
 * API client for the ForgeVision backend.
 * Override base URL with VITE_API_URL in ui/.env if needed.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export async function fetchCategories() {
  const res = await fetch(`${API_BASE}/categories`);
  if (!res.ok) throw new Error(`Failed to load categories (${res.status})`);
  return res.json();
}

/**
 * @param {File} file
 * @param {string} category
 * @param {"autoencoder"|"patchcore"} method
 */
export async function predict(file, category, method) {
  const form = new FormData();
  form.append("image", file);
  form.append("category", category);
  form.append("method", method);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : data.detail
          ? JSON.stringify(data.detail)
          : `Predict failed (${res.status})`;
    throw new Error(detail);
  }
  return data;
}

export { API_BASE };
