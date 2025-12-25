import { useMemo, useState } from "react"
import ResultCard from "./components/ResultCard"

type ApiResult = {
  diagnosis: string
  confidence: number
  risk: string
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ApiResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const canAnalyze = useMemo(() => !!file && !loading, [file, loading])

  const onPick = (f: File | null) => {
    setFile(f)
    setResult(null)
    setError(null)
    if (!f) {
      setPreview(null)
      return
    }
    setPreview(URL.createObjectURL(f))
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append("file", file)

      // Via Vite proxy:
      // Frontend calls /api/analyze -> proxied to http://127.0.0.1:8000/analyze
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || "Backend error")
      }

      const data = (await res.json()) as ApiResult
      setResult(data)
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">Aeriscan AI</div>
        <div className="tag">Educational demo</div>
      </header>

      <main className="container">
        <section className="hero">
          <h1>Chest X-ray Analyzer</h1>
          <p className="sub">
            Upload an image and get a simple pneumonia vs normal prediction.
            <span className="muted"> Educational use only.</span>
          </p>
        </section>

        <section className="card">
          <div className="uploader">
            <input
              id="file"
              type="file"
              accept="image/*"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />

            <label htmlFor="file" className="btn secondary">
              Choose image
            </label>

            <button className="btn primary" disabled={!canAnalyze} onClick={analyze}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {preview && (
            <div className="previewWrap">
              <img className="preview" src={preview} alt="X-ray preview" />
            </div>
          )}
        </section>

        {error && (
          <div className="errorBox">
            <strong>Something went wrong:</strong> {error}
          </div>
        )}

        {result && (
          <ResultCard
            diagnosis={result.diagnosis}
            confidence={result.confidence}
            risk={result.risk}
          />
        )}

        <footer className="footer">
          <div className="disclaimer">
            Educational use only. Not a medical diagnosis.
          </div>
        </footer>
      </main>
    </div>
  )
}
