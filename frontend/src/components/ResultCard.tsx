type Props = {
  diagnosis: string
  confidence: number
  risk: string
}

function clamp(n: number) {
  return Math.max(0, Math.min(100, n))
}

export default function ResultCard({ diagnosis, confidence, risk }: Props) {
  const c = clamp(confidence)
  const barClass =
    diagnosis.toLowerCase() === "pneumonia" ? "barFill pneumonia" : "barFill normal"

  return (
    <section className="resultCard">
      <div className="resultHeader">
        <h2>Result</h2>
        <span className="pill">{diagnosis}</span>
      </div>

      <div className="grid">
        <div className="metric">
          <div className="label">Confidence</div>
          <div className="value">{c.toFixed(2)}%</div>
        </div>

        <div className="metric">
          <div className="label">Risk Level</div>
          <div className={`value risk ${risk.toLowerCase()}`}>{risk}</div>
        </div>
      </div>

      <div className="barBg" aria-label="confidence bar">
        <div className={barClass} style={{ width: `${c}%` }} />
      </div>

      <div className="tiny">
        This tool is for learning and demo purposes only and must not be used for medical decisions.
      </div>
    </section>
  )
}
