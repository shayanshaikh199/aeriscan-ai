import { useMemo, useState } from "react"
import ResultCard from "./components/ResultCard"
import "./App.css"

type ApiResult = {
  diagnosis: "Pneumonia" | "Normal"
  confidence: number
  risk: string
  heatmap?: string
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

  const overlayClass =
    result?.diagnosis === "Pneumonia" ? "heatmap red" : "heatmap green"

  const tooltipText =
    result?.diagnosis === "Pneumonia"
      ? "Highlighted areas indicate regions that most influenced the model’s pneumonia prediction."
      : "Highlighted areas indicate regions that most influenced the model’s normal prediction."

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
            Upload a chest X-ray image to receive a Pneumonia vs Normal prediction.
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

            <button
              className="btn primary"
              disabled={!canAnalyze}
              onClick={analyze}
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {preview && (
            <div className="overlayWrap">
              <img className="xray" src={preview} alt="X-ray preview" />

              {result?.heatmap && (
                <div className="heatmapWrap">
                  <img
                    className={`heatmap ${
                      result.diagnosis === "Pneumonia" ? "red" : "green"
                    }`}
                    src={`data:image/png;base64,${result.heatmap}`}
                    alt="Grad-CAM heatmap"
                  />


                  <div className="heatmapTooltip">
                    Highlighted areas indicate regions that most influenced
                    the model’s prediction.
                  </div>
                </div>
              )}
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
          Aeriscan AI · Created by Shayan Shaikh
        </footer>
      </main>
    </div>
  )
}
