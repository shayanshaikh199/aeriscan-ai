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

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Backend error (${res.status})`)
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
      <header className="nav">
        <div className="navInner">
          <div className="navBrand">Aeriscan AI</div>
          <div className="navTag">Educational demo</div>
        </div>
      </header>

      <main className="wrap">
        <section className="hero">
          <h1 className="title">Chest X-ray Analyzer</h1>
          <p className="subtitle">
            Upload an image and get a pneumonia vs normal prediction.
            <span className="muted"> Educational use only.</span>
          </p>
        </section>

        <section className="card">
          <div className="actions">
            <input
              id="file"
              type="file"
              accept="image/*"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />

            <label htmlFor="file" className="pill ghost">
              Choose image
            </label>

            <button className="pill primary" disabled={!canAnalyze} onClick={analyze}>
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
          <div className="error">
            <div className="errorTitle">Something went wrong</div>
            <div className="errorMsg">{error}</div>
          </div>
        )}

        {result && (
          <ResultCard
            diagnosis={result.diagnosis}
            confidence={result.confidence}
            risk={result.risk}
          />
        )}

        <footer className="foot">
          <div className="footText">
            Educational use only. Not a medical diagnosis.
          </div>
        </footer>
      </main>
    </div>
  )
}
