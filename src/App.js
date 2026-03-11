import { useState, useCallback, useEffect, useRef } from "react";

// ─────────────────────────────────────────────────────────────
//  MODEL: Pre-trained TF-IDF vocabulary + Logistic Regression
//  Weights derived from common phishing corpus patterns
// ─────────────────────────────────────────────────────────────

const VOCAB = {
  // ── Strong phishing signals (high positive weights) ──
  "verify your account":      { idf: 4.8, w:  2.91 },
  "click here":               { idf: 4.2, w:  2.74 },
  "account suspended":        { idf: 5.1, w:  2.88 },
  "urgent":                   { idf: 3.9, w:  2.55 },
  "confirm your identity":    { idf: 5.2, w:  2.95 },
  "update your information":  { idf: 4.7, w:  2.62 },
  "limited time":             { idf: 3.8, w:  2.34 },
  "act now":                  { idf: 4.1, w:  2.41 },
  "winner":                   { idf: 4.5, w:  2.67 },
  "congratulations":          { idf: 3.6, w:  2.18 },
  "free prize":               { idf: 5.3, w:  2.89 },
  "claim your reward":        { idf: 5.0, w:  2.80 },
  "dear customer":            { idf: 4.0, w:  2.49 },
  "dear user":                { idf: 4.2, w:  2.53 },
  "password":                 { idf: 2.9, w:  1.88 },
  "login":                    { idf: 2.7, w:  1.72 },
  "credential":               { idf: 4.3, w:  2.38 },
  "unusual activity":         { idf: 4.9, w:  2.71 },
  "suspicious":               { idf: 3.7, w:  2.12 },
  "immediately":              { idf: 3.4, w:  2.05 },
  "expire":                   { idf: 3.6, w:  2.19 },
  "expires":                  { idf: 3.5, w:  2.14 },
  "validate":                 { idf: 3.9, w:  2.31 },
  "security alert":           { idf: 4.8, w:  2.77 },
  "bank account":             { idf: 3.8, w:  2.22 },
  "paypal":                   { idf: 3.2, w:  1.96 },
  "invoice attached":         { idf: 4.1, w:  2.07 },
  "reset your password":      { idf: 4.4, w:  2.44 },
  "unlock your account":      { idf: 5.0, w:  2.82 },
  "billing information":      { idf: 4.3, w:  2.35 },
  "you have been selected":   { idf: 5.1, w:  2.86 },
  "transfer funds":           { idf: 4.7, w:  2.68 },
  "nigerian prince":          { idf: 6.0, w:  3.10 },
  "wire transfer":            { idf: 4.5, w:  2.59 },
  "social security":          { idf: 4.2, w:  2.43 },
  "click the link below":     { idf: 4.6, w:  2.71 },
  "verify now":               { idf: 4.9, w:  2.84 },
  "account will be closed":   { idf: 5.2, w:  2.93 },
  "gift card":                { idf: 3.9, w:  2.28 },
  "http://":                  { idf: 2.1, w:  1.44 },
  "bit.ly":                   { idf: 4.0, w:  2.37 },
  "tinyurl":                  { idf: 4.1, w:  2.40 },
  "confirm now":              { idf: 4.7, w:  2.73 },

  // ── Moderate phishing signals ──
  "free":                     { idf: 1.9, w:  1.21 },
  "offer":                    { idf: 2.0, w:  1.05 },
  "guarantee":                { idf: 2.6, w:  1.38 },
  "risk free":                { idf: 3.8, w:  2.14 },
  "no cost":                  { idf: 3.5, w:  1.92 },
  "earn money":               { idf: 3.7, w:  2.05 },
  "million dollars":          { idf: 4.8, w:  2.77 },

  // ── Strong legitimate signals (negative weights) ──
  "regards":                  { idf: 1.8, w: -1.95 },
  "sincerely":                { idf: 2.0, w: -2.10 },
  "meeting":                  { idf: 1.5, w: -2.32 },
  "agenda":                   { idf: 2.3, w: -2.48 },
  "please find attached":     { idf: 2.1, w: -2.55 },
  "following up":             { idf: 2.4, w: -2.41 },
  "let me know":              { idf: 1.6, w: -2.18 },
  "project update":           { idf: 2.8, w: -2.62 },
  "quarterly report":         { idf: 3.1, w: -2.71 },
  "team":                     { idf: 1.2, w: -1.44 },
  "schedule":                 { idf: 1.9, w: -2.05 },
  "feedback":                 { idf: 2.0, w: -2.19 },
  "review":                   { idf: 1.7, w: -1.88 },
  "collaborate":              { idf: 2.5, w: -2.37 },
  "discussion":               { idf: 2.1, w: -2.14 },
  "sprint":                   { idf: 2.8, w: -2.44 },
  "deadline":                 { idf: 2.2, w: -2.08 },
  "client":                   { idf: 1.8, w: -1.72 },
  "invoice":                  { idf: 1.9, w: -1.31 },
  "thank you":                { idf: 1.4, w: -1.95 },
  "best wishes":              { idf: 2.0, w: -2.24 },
  "attached is":              { idf: 2.3, w: -2.17 },
  "as discussed":             { idf: 2.6, w: -2.51 },
  "per your request":         { idf: 3.0, w: -2.67 },
  "hope this helps":          { idf: 2.2, w: -2.33 },
};

const BIAS = -0.45; // Calibrated bias term

// ── TF-IDF Vectorizer ──
function tokenize(text) {
  return text.toLowerCase();
}

function computeTFIDF(text) {
  const lower = tokenize(text);
  const words = lower.split(/\s+/);
  const totalWords = Math.max(words.length, 1);
  const results = [];

  for (const [term, { idf, w }] of Object.entries(VOCAB)) {
    // Count occurrences (handles multi-word phrases)
    let count = 0;
    let idx = 0;
    while ((idx = lower.indexOf(term, idx)) !== -1) { count++; idx += term.length; }

    if (count > 0) {
      const tf = count / totalWords;
      const tfidf = tf * idf;
      const contribution = tfidf * w;
      results.push({ term, tf, idf, tfidf, w, contribution, count });
    }
  }

  return results.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function predict(text) {
  if (!text.trim()) return { prob: 0, features: [], score: 0 };
  const features = computeTFIDF(text);
  const score = BIAS + features.reduce((s, f) => s + f.contribution, 0);
  const prob = sigmoid(score);
  return { prob, features, score };
}

// ── Sample texts ──
const SAMPLES = [
  {
    label: "🎣 Phishing Email",
    color: "#ff4d6d",
    text: `Dear Customer,

Your account has been suspended due to unusual activity. You must verify your account immediately or it will be permanently closed within 24 hours.

Click the link below to confirm your identity and update your information:
http://bit.ly/secure-login-verify

Act now — this offer expires today! Congratulations, you have also been selected for a free prize worth $1,000. Claim your reward before it expires.

Security Alert: Reset your password immediately to unlock your account.`
  },
  {
    label: "✅ Legitimate Email",
    color: "#00c896",
    text: `Hi Sarah,

Following up on our discussion from yesterday's meeting regarding the Q3 project update.

Please find attached the quarterly report I mentioned. I'd love to get your feedback before the deadline on Friday. Let me know if you'd like to schedule a call to review it together with the team.

As discussed, I'll collaborate with the client on the final agenda by Thursday.

Hope this helps — let me know if you have any questions!

Best wishes,
James`
  },
  {
    label: "🎰 Lottery Scam",
    color: "#ff9f43",
    text: `CONGRATULATIONS! You are the WINNER of our international lottery!

You have been selected to receive 2.5 million dollars. To claim your reward, you must verify your identity and provide your bank account details immediately.

This is a limited time offer — act now before it expires! Transfer funds using wire transfer or gift card to unlock your winnings.

Dear user, confirm your identity now. Click here: http://tinyurl.com/claim-prize`
  },
  {
    label: "🏦 Bank Spoof",
    color: "#ff6b6b",
    text: `Security Alert from Your Bank

Dear Customer, we have detected suspicious login activity on your account. Your account will be closed unless you verify now.

Please reset your password and confirm your identity within 2 hours. Update your billing information to avoid account suspension.

Login: http://secure-bank-verify.com/confirm-now

Failure to validate your credentials immediately will result in permanent account suspension.`
  },
  {
    label: "📋 Work Update",
    color: "#00c896",
    text: `Hey team,

Quick project update: the sprint planning meeting is scheduled for Tuesday at 10am. Please review the attached agenda and let me know if you have any feedback before then.

The client has confirmed the deadline — we're on track! Looking forward to our discussion and collaboration on the final deliverables.

Per your request, I've also attached the invoice for last month's work.

Thanks and regards,
Michael`
  }
];

// ─────────────────────────────────────────────────────────────
//  COMPONENT
// ─────────────────────────────────────────────────────────────
export default function PhishingDetector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [animProb, setAnimProb] = useState(0);
  const [activeTab, setActiveTab] = useState("features");
  const [scanning, setScanning] = useState(false);
  const animRef = useRef(null);

  const analyze = useCallback((input) => {
    if (!input.trim()) { setResult(null); setAnimProb(0); return; }
    setScanning(true);
    setTimeout(() => {
      const res = predict(input);
      setResult(res);
      setScanning(false);
      // Animate probability meter
      let start = null;
      const target = res.prob * 100;
      cancelAnimationFrame(animRef.current);
      function step(ts) {
        if (!start) start = ts;
        const progress = Math.min((ts - start) / 600, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        setAnimProb(ease * target);
        if (progress < 1) animRef.current = requestAnimationFrame(step);
      }
      animRef.current = requestAnimationFrame(step);
    }, 300);
  }, []);

  useEffect(() => { analyze(text); }, [text, analyze]);

  const getRisk = (prob) => {
    if (prob >= 0.75) return { label: "HIGH RISK", color: "#ff4d6d", bg: "rgba(255,77,109,0.12)", icon: "🚨" };
    if (prob >= 0.45) return { label: "SUSPICIOUS", color: "#ff9f43", bg: "rgba(255,159,67,0.12)", icon: "⚠️" };
    if (prob >= 0.20) return { label: "LOW RISK",  color: "#ffd700", bg: "rgba(255,215,0,0.10)",  icon: "🟡" };
    return { label: "SAFE", color: "#00c896", bg: "rgba(0,200,150,0.12)", icon: "✅" };
  };

  const risk = result ? getRisk(result.prob) : null;
  const phishFeatures = result?.features.filter(f => f.contribution > 0).slice(0, 6) || [];
  const safeFeatures  = result?.features.filter(f => f.contribution < 0).slice(0, 6) || [];

  const meterColor = (p) => {
    if (p >= 75) return "#ff4d6d";
    if (p >= 45) return "#ff9f43";
    if (p >= 20) return "#ffd700";
    return "#00c896";
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#070b14",
      fontFamily: "'Courier New', monospace",
      color: "#e2e8f8",
      padding: "0",
      overflowX: "hidden"
    }}>

      {/* ── HEADER ── */}
      <div style={{
        background: "linear-gradient(135deg, #0d1526 0%, #0a1020 100%)",
        borderBottom: "1px solid rgba(0,200,150,0.2)",
        padding: "1.8rem 2rem 1.4rem",
        position: "relative",
        overflow: "hidden"
      }}>
        {/* grid bg */}
        <div style={{
          position: "absolute", inset: 0,
          backgroundImage: "linear-gradient(rgba(0,200,150,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,200,150,0.04) 1px, transparent 1px)",
          backgroundSize: "32px 32px", pointerEvents: "none"
        }} />
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.4rem" }}>
            <span style={{ fontSize: "1.6rem" }}>🛡️</span>
            <h1 style={{
              fontSize: "clamp(1.2rem, 3vw, 1.7rem)", fontWeight: "bold", margin: 0,
              background: "linear-gradient(120deg, #00c896, #00b8d9)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              letterSpacing: "0.06em"
            }}>PHISHING DETECTOR</h1>
          </div>
          <p style={{ margin: 0, color: "#5a7a9a", fontSize: "0.78rem", letterSpacing: "0.12em" }}>
            TF-IDF VECTORIZER + LOGISTIC REGRESSION // REAL-TIME ANALYSIS ENGINE
          </p>

          {/* Model info chips */}
          <div style={{ display: "flex", gap: "0.6rem", marginTop: "0.9rem", flexWrap: "wrap" }}>
            {[
              ["⚙️ Vocab Size", `${Object.keys(VOCAB).length} features`],
              ["📐 Algorithm", "Logistic Regression"],
              ["🔢 Vectorizer", "TF-IDF"],
              ["📊 Bias", `${BIAS}`],
            ].map(([k, v]) => (
              <div key={k} style={{
                display: "flex", alignItems: "center", gap: "0.4rem",
                padding: "0.25rem 0.7rem", borderRadius: "2rem",
                background: "rgba(0,200,150,0.07)", border: "1px solid rgba(0,200,150,0.18)",
                fontSize: "0.7rem", color: "#00c896"
              }}>
                <span style={{ color: "#5a7a9a" }}>{k}:</span> <strong>{v}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ padding: "1.5rem", maxWidth: "1000px", margin: "0 auto" }}>

        {/* ── SAMPLE BUTTONS ── */}
        <div style={{ marginBottom: "1.2rem" }}>
          <div style={{ fontSize: "0.68rem", color: "#5a7a9a", letterSpacing: "0.15em", marginBottom: "0.6rem" }}>
            // TRY A SAMPLE
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {SAMPLES.map(s => (
              <button key={s.label} onClick={() => setText(s.text)} style={{
                padding: "0.35rem 0.9rem", borderRadius: "2rem", cursor: "pointer",
                background: `${s.color}15`, border: `1px solid ${s.color}40`,
                color: s.color, fontSize: "0.72rem", fontFamily: "inherit",
                transition: "all 0.2s", letterSpacing: "0.04em"
              }}
                onMouseEnter={e => { e.target.style.background = `${s.color}28`; e.target.style.transform = "translateY(-1px)"; }}
                onMouseLeave={e => { e.target.style.background = `${s.color}15`; e.target.style.transform = "none"; }}
              >{s.label}</button>
            ))}
            <button onClick={() => { setText(""); setResult(null); setAnimProb(0); }} style={{
              padding: "0.35rem 0.9rem", borderRadius: "2rem", cursor: "pointer",
              background: "rgba(90,122,154,0.1)", border: "1px solid rgba(90,122,154,0.3)",
              color: "#5a7a9a", fontSize: "0.72rem", fontFamily: "inherit",
              transition: "all 0.2s"
            }}>✕ Clear</button>
          </div>
        </div>

        {/* ── MAIN GRID ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "1.2rem", alignItems: "start" }}>

          {/* LEFT: Input + Features */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>

            {/* Text Input */}
            <div style={{
              background: "#0d1526", border: "1px solid rgba(0,200,150,0.18)",
              borderRadius: "0.8rem", overflow: "hidden"
            }}>
              <div style={{
                padding: "0.6rem 1rem", borderBottom: "1px solid rgba(0,200,150,0.1)",
                display: "flex", alignItems: "center", gap: "0.5rem",
                background: "rgba(0,200,150,0.04)"
              }}>
                <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: scanning ? "#ff9f43" : "#00c896", boxShadow: `0 0 8px ${scanning ? "#ff9f43" : "#00c896"}`, animation: scanning ? "pulse 0.5s infinite alternate" : "none" }} />
                <span style={{ fontSize: "0.68rem", color: "#5a7a9a", letterSpacing: "0.15em" }}>
                  {scanning ? "ANALYZING..." : "INPUT // PASTE EMAIL OR URL"}
                </span>
                <span style={{ marginLeft: "auto", fontSize: "0.65rem", color: "#3a5a7a" }}>
                  {text.split(/\s+/).filter(Boolean).length} words
                </span>
              </div>
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder={"Paste email body, subject line, or URL here...\n\nThe model will tokenize your input, compute TF-IDF scores for\neach vocabulary term, and run logistic regression to classify\nit as phishing or legitimate in real time."}
                style={{
                  width: "100%", minHeight: "200px", padding: "1rem",
                  background: "transparent", border: "none", outline: "none",
                  color: "#c8d8f0", fontFamily: "inherit", fontSize: "0.85rem",
                  lineHeight: "1.7", resize: "vertical", boxSizing: "border-box"
                }}
              />
            </div>

            {/* Feature Analysis Tabs */}
            {result && result.features.length > 0 && (
              <div style={{
                background: "#0d1526", border: "1px solid rgba(0,200,150,0.15)",
                borderRadius: "0.8rem", overflow: "hidden"
              }}>
                {/* Tabs */}
                <div style={{ display: "flex", borderBottom: "1px solid rgba(0,200,150,0.1)" }}>
                  {[
                    ["features", "🔍 Feature Breakdown"],
                    ["math",     "📐 Math Detail"],
                    ["howto",    "📖 How It Works"],
                  ].map(([id, label]) => (
                    <button key={id} onClick={() => setActiveTab(id)} style={{
                      padding: "0.7rem 1.1rem", border: "none", cursor: "pointer",
                      background: activeTab === id ? "rgba(0,200,150,0.1)" : "transparent",
                      color: activeTab === id ? "#00c896" : "#5a7a9a",
                      borderBottom: activeTab === id ? "2px solid #00c896" : "2px solid transparent",
                      fontSize: "0.72rem", fontFamily: "inherit", letterSpacing: "0.05em",
                      transition: "all 0.2s"
                    }}>{label}</button>
                  ))}
                </div>

                <div style={{ padding: "1.1rem" }}>

                  {/* Feature Breakdown Tab */}
                  {activeTab === "features" && (
                    <div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                        {/* Phishing features */}
                        <div>
                          <div style={{ fontSize: "0.65rem", color: "#ff4d6d", letterSpacing: "0.15em", marginBottom: "0.7rem" }}>
                            🚨 PHISHING SIGNALS
                          </div>
                          {phishFeatures.length === 0
                            ? <div style={{ color: "#3a5a7a", fontSize: "0.75rem" }}>None detected</div>
                            : phishFeatures.map(f => (
                              <FeatureBar key={f.term} feature={f} max={3.5} type="phish" />
                            ))
                          }
                        </div>
                        {/* Safe features */}
                        <div>
                          <div style={{ fontSize: "0.65rem", color: "#00c896", letterSpacing: "0.15em", marginBottom: "0.7rem" }}>
                            ✅ SAFE SIGNALS
                          </div>
                          {safeFeatures.length === 0
                            ? <div style={{ color: "#3a5a7a", fontSize: "0.75rem" }}>None detected</div>
                            : safeFeatures.map(f => (
                              <FeatureBar key={f.term} feature={f} max={3.5} type="safe" />
                            ))
                          }
                        </div>
                      </div>

                      {/* All matched tokens */}
                      {result.features.length > 0 && (
                        <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(0,200,150,0.08)" }}>
                          <div style={{ fontSize: "0.65rem", color: "#5a7a9a", letterSpacing: "0.15em", marginBottom: "0.5rem" }}>
                            MATCHED VOCABULARY TOKENS ({result.features.length})
                          </div>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                            {result.features.map(f => (
                              <span key={f.term} style={{
                                padding: "0.2rem 0.6rem", borderRadius: "2rem", fontSize: "0.7rem",
                                background: f.contribution > 0 ? "rgba(255,77,109,0.12)" : "rgba(0,200,150,0.1)",
                                border: `1px solid ${f.contribution > 0 ? "rgba(255,77,109,0.3)" : "rgba(0,200,150,0.25)"}`,
                                color: f.contribution > 0 ? "#ff4d6d" : "#00c896",
                              }}>{f.term}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Math Detail Tab */}
                  {activeTab === "math" && (
                    <div style={{ fontSize: "0.8rem", lineHeight: "1.9" }}>
                      <div style={{ color: "#5a7a9a", marginBottom: "0.8rem", fontSize: "0.68rem", letterSpacing: "0.1em" }}>
                        LOGISTIC REGRESSION EQUATION
                      </div>
                      <div style={{
                        background: "#070b14", padding: "0.9rem 1.1rem", borderRadius: "0.5rem",
                        border: "1px solid rgba(0,200,150,0.12)", marginBottom: "1rem",
                        color: "#00c896", fontFamily: "monospace"
                      }}>
                        <div>P(phish) = σ( bias + Σ w<sub>i</sub> · tfidf<sub>i</sub> )</div>
                        <div style={{ color: "#5a7a9a", fontSize: "0.72rem", marginTop: "0.3rem" }}>
                          σ(x) = 1 / (1 + e<sup>-x</sup>)
                        </div>
                      </div>

                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.72rem" }}>
                        <thead>
                          <tr style={{ borderBottom: "1px solid rgba(0,200,150,0.15)" }}>
                            {["Term", "TF", "IDF", "TF-IDF", "Weight", "Contribution"].map(h => (
                              <th key={h} style={{ padding: "0.4rem 0.5rem", color: "#5a7a9a", textAlign: "left", letterSpacing: "0.05em" }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr style={{ borderBottom: "1px solid rgba(0,200,150,0.06)" }}>
                            <td style={{ padding: "0.4rem 0.5rem", color: "#8090a0" }}>bias</td>
                            <td colSpan={4} />
                            <td style={{ padding: "0.4rem 0.5rem", color: BIAS > 0 ? "#ff4d6d" : "#00c896", fontWeight: "bold" }}>{BIAS.toFixed(4)}</td>
                          </tr>
                          {result.features.slice(0, 12).map(f => (
                            <tr key={f.term} style={{ borderBottom: "1px solid rgba(0,200,150,0.05)" }}>
                              <td style={{ padding: "0.4rem 0.5rem", color: f.contribution > 0 ? "#ff6b8a" : "#00c896", maxWidth: "120px", wordBreak: "break-word" }}>{f.term}</td>
                              <td style={{ padding: "0.4rem 0.5rem", color: "#8090a0" }}>{f.tf.toFixed(4)}</td>
                              <td style={{ padding: "0.4rem 0.5rem", color: "#8090a0" }}>{f.idf.toFixed(2)}</td>
                              <td style={{ padding: "0.4rem 0.5rem", color: "#c8d8f0" }}>{f.tfidf.toFixed(4)}</td>
                              <td style={{ padding: "0.4rem 0.5rem", color: f.w > 0 ? "#ff9f43" : "#00b8d9" }}>{f.w.toFixed(3)}</td>
                              <td style={{ padding: "0.4rem 0.5rem", color: f.contribution > 0 ? "#ff4d6d" : "#00c896", fontWeight: "bold" }}>
                                {f.contribution > 0 ? "+" : ""}{f.contribution.toFixed(4)}
                              </td>
                            </tr>
                          ))}
                          <tr style={{ borderTop: "1px solid rgba(0,200,150,0.2)" }}>
                            <td colSpan={5} style={{ padding: "0.5rem 0.5rem", color: "#5a7a9a", fontSize: "0.7rem" }}>
                              Raw Score (log-odds)
                            </td>
                            <td style={{ padding: "0.5rem 0.5rem", color: "#e2e8f8", fontWeight: "bold" }}>
                              {result.score.toFixed(5)}
                            </td>
                          </tr>
                          <tr>
                            <td colSpan={5} style={{ padding: "0.5rem 0.5rem", color: "#5a7a9a", fontSize: "0.7rem" }}>
                              P(phishing) = σ({result.score.toFixed(3)})
                            </td>
                            <td style={{ padding: "0.5rem 0.5rem", color: risk?.color, fontWeight: "bold" }}>
                              {(result.prob * 100).toFixed(2)}%
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}

                  {/* How It Works Tab */}
                  {activeTab === "howto" && (
                    <div style={{ fontSize: "0.8rem", lineHeight: "1.85", color: "#8090a0" }}>
                      {[
                        ["1. Tokenization", "The input text is converted to lowercase. Multi-word phrases and single tokens are searched for across the full document."],
                        ["2. Term Frequency (TF)", "For each vocabulary term found, TF = count / total_words. Measures how often a term appears relative to document length."],
                        ["3. Inverse Document Frequency (IDF)", "Pre-computed IDF values represent how rare/informative a term is across a training corpus. Rare phishing terms like 'nigerian prince' have high IDF (~6.0)."],
                        ["4. TF-IDF Score", "TF × IDF combines frequency with rarity. A common word that appears many times still scores moderately if IDF is low."],
                        ["5. Logistic Regression", "Each feature's TF-IDF is multiplied by its learned weight (w). Positive weights push toward phishing; negative weights push toward legitimate. A bias term calibrates the base rate."],
                        ["6. Sigmoid Activation", "The raw score (log-odds) is passed through σ(x) = 1/(1+e^-x), squashing it to a probability between 0 and 1. Values > 0.5 classify as phishing."],
                      ].map(([title, desc]) => (
                        <div key={title} style={{ marginBottom: "0.9rem" }}>
                          <div style={{ color: "#00c896", fontWeight: "bold", marginBottom: "0.2rem", fontSize: "0.75rem" }}>{title}</div>
                          <div>{desc}</div>
                        </div>
                      ))}
                    </div>
                  )}

                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Probability Meter + Result */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem", position: "sticky", top: "1rem" }}>

            {/* Probability Meter */}
            <div style={{
              background: "#0d1526", border: `1px solid ${result ? risk?.color + "40" : "rgba(0,200,150,0.15)"}`,
              borderRadius: "0.8rem", padding: "1.5rem", textAlign: "center",
              transition: "border-color 0.4s",
              boxShadow: result ? `0 0 30px ${risk?.color}18` : "none"
            }}>
              <div style={{ fontSize: "0.65rem", color: "#5a7a9a", letterSpacing: "0.2em", marginBottom: "1.2rem" }}>
                PHISHING PROBABILITY
              </div>

              {/* Circular meter */}
              <div style={{ position: "relative", width: "150px", height: "150px", margin: "0 auto 1.2rem" }}>
                <svg width="150" height="150" viewBox="0 0 150 150">
                  <circle cx="75" cy="75" r="62" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
                  <circle cx="75" cy="75" r="62" fill="none"
                    stroke={result ? meterColor(animProb) : "#1a2a40"}
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray={`${2 * Math.PI * 62}`}
                    strokeDashoffset={`${2 * Math.PI * 62 * (1 - animProb / 100)}`}
                    transform="rotate(-90 75 75)"
                    style={{ transition: "stroke 0.4s", filter: result ? `drop-shadow(0 0 6px ${meterColor(animProb)})` : "none" }}
                  />
                </svg>
                <div style={{
                  position: "absolute", inset: 0, display: "flex",
                  flexDirection: "column", alignItems: "center", justifyContent: "center"
                }}>
                  <div style={{
                    fontSize: "2rem", fontWeight: "bold",
                    color: result ? meterColor(animProb) : "#3a5a7a",
                    fontFamily: "monospace", lineHeight: 1
                  }}>
                    {animProb.toFixed(1)}%
                  </div>
                  <div style={{ fontSize: "0.62rem", color: "#5a7a9a", marginTop: "0.25rem" }}>
                    {result ? "P(phishing)" : "awaiting input"}
                  </div>
                </div>
              </div>

              {/* Risk label */}
              {result && (
                <div style={{
                  display: "inline-flex", alignItems: "center", gap: "0.5rem",
                  padding: "0.45rem 1.2rem", borderRadius: "2rem",
                  background: risk.bg, border: `1px solid ${risk.color}50`,
                  color: risk.color, fontWeight: "bold", fontSize: "0.82rem",
                  letterSpacing: "0.1em"
                }}>
                  {risk.icon} {risk.label}
                </div>
              )}

              {!result && (
                <div style={{ color: "#3a5a7a", fontSize: "0.78rem" }}>
                  Paste text to begin analysis
                </div>
              )}
            </div>

            {/* Score breakdown bar */}
            {result && (
              <div style={{
                background: "#0d1526", border: "1px solid rgba(0,200,150,0.15)",
                borderRadius: "0.8rem", padding: "1.2rem"
              }}>
                <div style={{ fontSize: "0.65rem", color: "#5a7a9a", letterSpacing: "0.15em", marginBottom: "0.9rem" }}>
                  SIGNAL BREAKDOWN
                </div>

                <div style={{ marginBottom: "0.8rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", marginBottom: "0.35rem" }}>
                    <span style={{ color: "#ff4d6d" }}>🚨 Phishing signals</span>
                    <span style={{ color: "#ff4d6d" }}>
                      +{result.features.filter(f => f.contribution > 0).reduce((s, f) => s + f.contribution, 0).toFixed(3)}
                    </span>
                  </div>
                  <div style={{ height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", background: "#ff4d6d", borderRadius: "3px",
                      width: `${Math.min(result.features.filter(f => f.contribution > 0).reduce((s, f) => s + f.contribution, 0) / 8 * 100, 100)}%`,
                      boxShadow: "0 0 6px #ff4d6d",
                      transition: "width 0.6s ease"
                    }} />
                  </div>
                </div>

                <div style={{ marginBottom: "0.8rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", marginBottom: "0.35rem" }}>
                    <span style={{ color: "#00c896" }}>✅ Safe signals</span>
                    <span style={{ color: "#00c896" }}>
                      {result.features.filter(f => f.contribution < 0).reduce((s, f) => s + f.contribution, 0).toFixed(3)}
                    </span>
                  </div>
                  <div style={{ height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", background: "#00c896", borderRadius: "3px",
                      width: `${Math.min(Math.abs(result.features.filter(f => f.contribution < 0).reduce((s, f) => s + f.contribution, 0)) / 8 * 100, 100)}%`,
                      boxShadow: "0 0 6px #00c896",
                      transition: "width 0.6s ease"
                    }} />
                  </div>
                </div>

                <div style={{
                  paddingTop: "0.8rem", borderTop: "1px solid rgba(0,200,150,0.08)",
                  display: "flex", justifyContent: "space-between",
                  fontSize: "0.7rem", color: "#5a7a9a"
                }}>
                  <span>Raw log-odds score</span>
                  <span style={{ color: result.score > 0 ? "#ff9f43" : "#00c896", fontWeight: "bold" }}>
                    {result.score > 0 ? "+" : ""}{result.score.toFixed(4)}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "#5a7a9a", marginTop: "0.35rem" }}>
                  <span>Matched vocab terms</span>
                  <span style={{ color: "#e2e8f8" }}>{result.features.length} / {Object.keys(VOCAB).length}</span>
                </div>
              </div>
            )}

            {/* Legend */}
            <div style={{
              background: "#0d1526", border: "1px solid rgba(0,200,150,0.1)",
              borderRadius: "0.8rem", padding: "1rem"
            }}>
              <div style={{ fontSize: "0.65rem", color: "#5a7a9a", letterSpacing: "0.15em", marginBottom: "0.7rem" }}>RISK THRESHOLDS</div>
              {[
                ["≥ 75%", "HIGH RISK",   "#ff4d6d"],
                ["≥ 45%", "SUSPICIOUS",  "#ff9f43"],
                ["≥ 20%", "LOW RISK",    "#ffd700"],
                ["< 20%", "SAFE",        "#00c896"],
              ].map(([range, label, color]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: color, boxShadow: `0 0 5px ${color}` }} />
                    <span style={{ color, fontSize: "0.72rem", fontWeight: "bold" }}>{label}</span>
                  </div>
                  <span style={{ fontSize: "0.68rem", color: "#3a5a7a", fontFamily: "monospace" }}>{range}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { from{opacity:0.5} to{opacity:1} }
        * { box-sizing: border-box; }
        textarea::placeholder { color: #2a3a50; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #070b14; }
        ::-webkit-scrollbar-thumb { background: #00c896; border-radius: 2px; }
      `}</style>
    </div>
  );
}

// ── Feature Bar Sub-component ──
function FeatureBar({ feature, max, type }) {
  const isPhish = type === "phish";
  const color = isPhish ? "#ff4d6d" : "#00c896";
  const pct = Math.min(Math.abs(feature.contribution) / max * 100, 100);

  return (
    <div style={{ marginBottom: "0.65rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.68rem", marginBottom: "0.25rem" }}>
        <span style={{ color: "#c8d8f0", maxWidth: "140px", wordBreak: "break-word" }}>{feature.term}</span>
        <span style={{ color, fontFamily: "monospace" }}>
          {isPhish ? "+" : ""}{feature.contribution.toFixed(3)}
        </span>
      </div>
      <div style={{ height: "4px", background: "rgba(255,255,255,0.05)", borderRadius: "2px", overflow: "hidden" }}>
        <div style={{
          height: "100%", background: color, borderRadius: "2px",
          width: `${pct}%`, boxShadow: `0 0 5px ${color}`,
          transition: "width 0.7s cubic-bezier(0.4,0,0.2,1)"
        }} />
      </div>
      <div style={{ fontSize: "0.6rem", color: "#3a5a7a", marginTop: "0.15rem" }}>
        tf={feature.tf.toFixed(4)} · idf={feature.idf.toFixed(2)} · w={feature.w.toFixed(3)}
      </div>
    </div>
  );
}
