import { useCallback, useEffect, useState } from "react";
import { fetchCategories, predict } from "./api.js";

function ResultPanel({ title, result, previewUrl }) {
  if (!result) return null;

  const badgeClass = result.is_anomaly ? "badge defect" : "badge pass";
  const badgeLabel = result.is_anomaly ? "DEFECT" : "PASS";

  return (
    <div className="result-panel">
      <div className="panel-header">
        <h3>{title}</h3>
        <span className={badgeClass}>{badgeLabel}</span>
      </div>
      <div className="metrics">
        <div>
          <span className="label">Score</span>
          <span className="value">{result.image_score.toFixed(4)}</span>
        </div>
        <div>
          <span className="label">Threshold</span>
          <span className="value">{result.threshold.toFixed(4)}</span>
        </div>
        <div>
          <span className="label">Inference</span>
          <span className="value">{result.inference_ms} ms</span>
        </div>
      </div>
      {result.threshold_is_default && (
        <p className="hint">Using default threshold (train/good not available).</p>
      )}
      <div className="image-row">
        <figure>
          <figcaption>Input</figcaption>
          <img src={previewUrl} alt="Input preview" />
        </figure>
        <figure>
          <figcaption>Heatmap</figcaption>
          <img
            src={`data:image/png;base64,${result.heatmap_png_base64}`}
            alt="Anomaly heatmap"
          />
        </figure>
        <figure>
          <figcaption>Overlay</figcaption>
          <img
            src={`data:image/png;base64,${result.overlay_png_base64}`}
            alt="Overlay"
          />
        </figure>
      </div>
    </div>
  );
}

export default function App() {
  const [catalog, setCatalog] = useState([]);
  const [category, setCategory] = useState("");
  const [method, setMethod] = useState("patchcore");
  const [compareMode, setCompareMode] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [compareResults, setCompareResults] = useState(null);

  useEffect(() => {
    fetchCategories()
      .then((data) => {
        setCatalog(data.categories || []);
        if (data.categories?.length) {
          setCategory(data.categories[0].category);
          const methods = data.categories[0].methods;
          if (methods.includes("patchcore")) setMethod("patchcore");
          else if (methods.length) setMethod(methods[0]);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  const availableMethods = catalog.find((c) => c.category === category)?.methods || [];

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setCompareResults(null);
    setError("");
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f?.type.startsWith("image/")) handleFile(f);
  };

  const runInference = async () => {
    if (!file || !category) {
      setError("Select a category and upload an image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setCompareResults(null);

    try {
      if (compareMode) {
        const methods = ["autoencoder", "patchcore"].filter((m) =>
          availableMethods.includes(m)
        );
        if (methods.length < 2) {
          throw new Error("Compare mode needs both methods trained for this category.");
        }
        const [ae, pc] = await Promise.all([
          predict(file, category, "autoencoder"),
          predict(file, category, "patchcore"),
        ]);
        setCompareResults({ autoencoder: ae, patchcore: pc });
      } else {
        if (!availableMethods.includes(method)) {
          throw new Error(`${method} is not available for ${category}.`);
        }
        const res = await predict(file, category, method);
        setResult(res);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSample = async (path) => {
    try {
      const res = await fetch(path);
      const blob = await res.blob();
      const f = new File([blob], path.split("/").pop(), { type: blob.type });
      handleFile(f);
    } catch {
      setError(`Could not load sample ${path}`);
    }
  };

  return (
    <div className="app">
      <header>
        <div>
          <h1>ForgeVision</h1>
          <p className="subtitle">
            Industrial visual anomaly detection — pick a category, upload an image, inspect the
            heatmap. Auto-category detection is a future enhancement.
          </p>
        </div>
      </header>

      {error && (
        <div className="toast error" role="alert">
          {error}
          <button type="button" onClick={() => setError("")} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      <section className="controls card">
        <div className="control-row">
          <label>
            Category
            <select
              value={category}
              onChange={(e) => {
                setCategory(e.target.value);
                setResult(null);
                setCompareResults(null);
              }}
            >
              {catalog.length === 0 && <option value="">No models found</option>}
              {catalog.map((c) => (
                <option key={c.category} value={c.category}>
                  {c.category} ({c.methods.join(", ")})
                </option>
              ))}
            </select>
          </label>

          {!compareMode && (
            <fieldset className="method-toggle">
              <legend>Method</legend>
              {["autoencoder", "patchcore"].map((m) => (
                <label key={m} className={!availableMethods.includes(m) ? "disabled" : ""}>
                  <input
                    type="radio"
                    name="method"
                    value={m}
                    checked={method === m}
                    disabled={!availableMethods.includes(m)}
                    onChange={() => setMethod(m)}
                  />
                  {m === "autoencoder" ? "Autoencoder" : "PatchCore"}
                </label>
              ))}
            </fieldset>
          )}

          <label className="compare-check">
            <input
              type="checkbox"
              checked={compareMode}
              onChange={(e) => setCompareMode(e.target.checked)}
            />
            Compare both methods
          </label>
        </div>

        <div
          className={`dropzone ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          {previewUrl ? (
            <img className="thumb" src={previewUrl} alt="Upload preview" />
          ) : (
            <p>Drag & drop an image, or use the file picker</p>
          )}
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        <div className="sample-row">
          <span>Try a synthetic sample:</span>
          <button type="button" onClick={() => loadSample("/samples/normal_pattern.png")}>
            Normal pattern
          </button>
          <button type="button" onClick={() => loadSample("/samples/defect_pattern.png")}>
            Defect pattern
          </button>
          <span className="hint-inline">
            Or drag from <code>data/mvtec_ad/&lt;category&gt;/test/</code> (local only, not in repo).
          </span>
        </div>

        <button
          type="button"
          className="primary"
          onClick={runInference}
          disabled={loading || !file || !category}
        >
          {loading ? "Running…" : compareMode ? "Compare both" : "Run inference"}
        </button>
      </section>

      {compareResults && (
        <section className="compare-grid">
          <ResultPanel
            title="Autoencoder"
            result={compareResults.autoencoder}
            previewUrl={previewUrl}
          />
          <ResultPanel
            title="PatchCore"
            result={compareResults.patchcore}
            previewUrl={previewUrl}
          />
        </section>
      )}

      {result && !compareMode && (
        <section className="card">
          <ResultPanel title={method === "autoencoder" ? "Autoencoder" : "PatchCore"} result={result} previewUrl={previewUrl} />
        </section>
      )}

      <footer>
        <p>
          Backend: <code>127.0.0.1:8000</code> · Per-category models · MVTec AD not redistributed
        </p>
      </footer>
    </div>
  );
}
