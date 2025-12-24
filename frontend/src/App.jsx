import { useMemo, useState } from "react";

const API_URL = "http://127.0.0.1:8000/predict";

function clampPct(x) {
  if (Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

function riskColor(risk) {
  // subtle, professional (no neon)
  if (risk === "High") return "bg-zinc-900";
  if (risk === "Moderate") return "bg-zinc-700";
  if (risk === "Low") return "bg-zinc-500";
  return "bg-zinc-400";
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const canAnalyze = useMemo(() => !!file && !loading, [file, loading]);

  function onPickFile(e) {
    const f = e.target.files?.[0];
    setError("");
    setResult(null);
    setFile(f || null);
    setPreviewUrl(f ? URL.createObjectURL(f) : "");
  }

  async function analyze() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(API_URL, {
        method: "POST",
        body: form,
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data?.error || "Request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const confidencePct = result ? clampPct(Math.round(result.confidence * 100)) : 0;

  return (
    <div className="min-h-screen bg-white">
      {/* Top bar */}
      <header className="sticky top-0 z-10 border-b border-zinc-200/70 bg-white/70 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-zinc-900" />
            <div className="leading-tight">
              <div className="text-[15px] font-semibold tracking-tight">Aeriscan AI</div>
              <div className="text-[12px] text-zinc-500">Chest X-ray screening (educational)</div>
            </div>
          </div>
          <div className="text-[12px] text-zinc-500">
            Local API: <span className="font-medium text-zinc-700">127.0.0.1:8000</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-6 py-10">
        <div className="grid gap-8 md:grid-cols-2">
          {/* Upload card */}
          <section className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-[0_10px_30px_rgba(0,0,0,0.06)]">
            <h1 className="text-2xl font-semibold tracking-tight">
              Analyze a chest X-ray
            </h1>
            <p className="mt-2 text-sm text-zinc-600">
              Upload an image (JPG/PNG). The model returns a probability estimate.
              This is <span className="font-medium">not</span> a medical diagnosis.
            </p>

            <div className="mt-6">
              <label className="block text-xs font-medium text-zinc-700">
                X-ray image
              </label>

              <div className="mt-2 flex items-center gap-3">
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={onPickFile}
                  className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-zinc-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-zinc-800"
                />
              </div>

              {previewUrl && (
                <div className="mt-5 overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50">
                  <img
                    src={previewUrl}
                    alt="X-ray preview"
                    className="h-[320px] w-full object-contain"
                  />
                </div>
              )}

              <button
                onClick={analyze}
                disabled={!canAnalyze}
                className="mt-6 inline-flex w-full items-center justify-center rounded-2xl bg-zinc-900 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-300"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>

              {error && (
                <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              )}
            </div>
          </section>

          {/* Results card */}
          <section className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-[0_10px_30px_rgba(0,0,0,0.06)]">
            <h2 className="text-lg font-semibold tracking-tight">Result</h2>
            <p className="mt-2 text-sm text-zinc-600">
              The UI is optimized to feel minimal and premium — no debug HTML, no emojis.
            </p>

            {!result ? (
              <div className="mt-8 rounded-2xl border border-zinc-200 bg-zinc-50 p-6 text-sm text-zinc-600">
                Upload an image and press <span className="font-medium">Analyze</span>.
              </div>
            ) : (
              <div className="mt-8">
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-xs font-medium text-zinc-500">Diagnosis</div>
                    <div className="mt-1 text-3xl font-semibold tracking-tight">
                      {result.diagnosis}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-medium text-zinc-500">Risk</div>
                    <div className="mt-1 text-lg font-semibold">{result.risk}</div>
                  </div>
                </div>

                {/* Big confidence bar */}
                <div className="mt-7">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-medium text-zinc-500">Confidence</div>
                    <div className="text-sm font-semibold text-zinc-900">{confidencePct}%</div>
                  </div>

                  <div className="mt-3 h-5 w-full rounded-full bg-zinc-100 ring-1 ring-zinc-200 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${riskColor(result.risk)}`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>

                  <div className="mt-3 text-xs text-zinc-500">
                    {result.disclaimer}
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>

        <footer className="mt-10 text-xs text-zinc-500">
          Aeriscan AI is for educational purposes only and must not be used for medical decisions.
        </footer>
      </main>
    </div>
  );
}
