import { useEffect, useState } from "react"

type Props = {
  diagnosis: string
  confidence: number
  risk: string
}

export default function ResultCard({ diagnosis, confidence, risk }: Props) {
  const [displayValue, setDisplayValue] = useState(0)

  const percent = Math.round(confidence * 100)
  const isNormal = diagnosis === "Normal"

  useEffect(() => {
    let start = 0
    const duration = 900
    const stepTime = 16
    const steps = duration / stepTime
    const increment = percent / steps

    const timer = setInterval(() => {
      start += increment
      if (start >= percent) {
        start = percent
        clearInterval(timer)
      }
      setDisplayValue(Math.round(start))
    }, stepTime)

    return () => clearInterval(timer)
  }, [percent])

  return (
    <section className="card resultCard">
      <div className={`pill ${isNormal ? "pillNormal" : "pillBad"}`}>
        {diagnosis}
      </div>

      <div className="resultGrid">
        <div>
          <div className="label">Confidence</div>
          <div className="confidenceValue">{displayValue}%</div>
        </div>

        <div>
          <div className="label">Risk Level</div>
          <div className="riskValue">{risk}</div>
        </div>
      </div>

      <div className="progressTrack">
        <div
          className={`progressFill ${isNormal ? "good" : "bad"}`}
          style={{ width: `${displayValue}%` }}
        />
      </div>

      <div className="disclaimerSmall">
        Educational use only. Not a medical diagnosis.
      </div>
    </section>
  )
}