type Props = {
  diagnosis: string
  confidence: number
  risk: string
}

export default function ResultCard({ diagnosis, confidence, risk }: Props) {
  const pct = Math.round(confidence * 100)
  const isNormal = diagnosis === "Normal"

  return (
    <section className="card result">
      <div className={`badge ${isNormal ? "good" : "bad"}`}>
        {diagnosis}
      </div>

      <div className="resultGrid">
        <div className="stat">
          <div className="label">Confidence</div>
          <div className="value">{pct}%</div>
        </div>

        <div className="stat">
          <div className="label">Risk Level</div>
          <div className="value">{risk}</div>
        </div>
      </div>

      <div className="barBg">
        <div
          className={`barFill ${isNormal ? "fillGood" : "fillBad"}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="fineprint">
        Educational use only. Not a medical diagnosis.
      </div>
    </section>
  )
}
