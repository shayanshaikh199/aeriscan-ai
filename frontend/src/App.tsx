import { useState } from "react";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Backend error");

      const data = await res.json();
      setResult(data);
    } catch {
      setError("Failed to analyze image");
    } finally {
      setLoading(false);
    }
  };

  const confidencePercent = result
    ? Math.round(result.confidence * 100)
    : 0;

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Aeriscan AI</h1>
        <p style={styles.subtitle}>
          AI-powered chest X-ray analysis for educational use
        </p>

        {/* Upload Card */}
        <div style={styles.card}>
          <label style={styles.uploadLabel}>
            Upload Chest X-ray
            <input
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setFile(f);
                  setPreview(URL.createObjectURL(f));
                }
              }}
            />
          </label>

          {preview && (
            <img
              src={preview}
              alt="X-ray preview"
              style={styles.image}
            />
          )}

          <button
            style={{
              ...styles.button,
              opacity: loading ? 0.6 : 1,
            }}
            onClick={analyze}
            disabled={loading || !file}
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>

          {error && <p style={styles.error}>{error}</p>}
        </div>

        {/* Results */}
        {result && (
          <div style={styles.card}>
            <h2 style={styles.resultTitle}>
              Diagnosis: {result.diagnosis}
            </h2>

            <p style={styles.metricLabel}>Confidence</p>
            <p style={styles.metricValue}>
              {confidencePercent}%
            </p>

            <div style={styles.progressBg}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${confidencePercent}%`,
                  background:
                    result.diagnosis === "Normal"
                      ? "linear-gradient(90deg, #22c55e, #4ade80)"
                      : "linear-gradient(90deg, #ef4444, #f87171)",
                }}
              />
            </div>

            <p style={styles.metricLabel}>Risk Level</p>
            <p style={styles.metricValue}>{result.risk}</p>

            <p style={styles.disclaimer}>
              Educational use only. Not a medical diagnosis.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================
   APPLE-STYLE DESIGN TOKENS
========================= */

const styles: { [key: string]: React.CSSProperties } = {
  page: {
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif",
    backgroundColor: "#ffffff",
    color: "#111111",
    minHeight: "100vh",
    padding: "60px 20px",
  },
  container: {
    maxWidth: "520px",
    margin: "0 auto",
  },
  title: {
    fontSize: "32px",
    fontWeight: 600,
    letterSpacing: "-0.02em",
  },
  subtitle: {
    color: "#6e6e73",
    fontSize: "16px",
    marginBottom: "40px",
  },
  card: {
    backgroundColor: "#f5f5f7",
    borderRadius: "18px",
    padding: "32px",
    marginBottom: "32px",
    border: "1px solid #e5e5ea",
  },
  uploadLabel: {
    display: "inline-block",
    padding: "14px 18px",
    borderRadius: "12px",
    backgroundColor: "#ffffff",
    border: "1px solid #d1d1d6",
    cursor: "pointer",
    fontWeight: 500,
    marginBottom: "20px",
  },
  image: {
    width: "100%",
    borderRadius: "12px",
    marginBottom: "20px",
  },
  button: {
    width: "100%",
    padding: "14px",
    borderRadius: "12px",
    border: "none",
    backgroundColor: "#111111",
    color: "#ffffff",
    fontSize: "16px",
    fontWeight: 500,
    cursor: "pointer",
  },
  error: {
    color: "#dc2626",
    marginTop: "12px",
  },
  resultTitle: {
    fontSize: "22px",
    fontWeight: 600,
    marginBottom: "20px",
  },
  metricLabel: {
    fontSize: "13px",
    color: "#6e6e73",
    marginTop: "16px",
  },
  metricValue: {
    fontSize: "26px",
    fontWeight: 500,
    marginTop: "6px",
  },
  progressBg: {
    width: "100%",
    height: "16px",
    backgroundColor: "#e5e5ea",
    borderRadius: "999px",
    overflow: "hidden",
    marginTop: "12px",
  },
  progressFill: {
    height: "100%",
    borderRadius: "999px",
  },
  disclaimer: {
    fontSize: "12px",
    color: "#9ca3af",
    marginTop: "24px",
  },
};

export default App;
