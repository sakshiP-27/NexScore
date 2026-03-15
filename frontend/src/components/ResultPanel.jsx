import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import { useEffect, useRef } from 'react'
import '../styles/Result.css'

const FINANCE_IMG = 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1400&q=80&auto=format&fit=crop'

const RISK = {
  'Low Risk':    { color: '#4ade80', glow: 'rgba(74,222,128,0.2)',  border: 'rgba(74,222,128,0.3)'  },
  'Medium Risk': { color: '#fbbf24', glow: 'rgba(251,191,36,0.2)',  border: 'rgba(251,191,36,0.3)'  },
  'High Risk':   { color: '#f87171', glow: 'rgba(248,113,113,0.2)', border: 'rgba(248,113,113,0.3)' },
}

const CIRCUMFERENCE = 2 * Math.PI * 54

function AnimatedScore({ score }) {
  const count = useMotionValue(0)
  const rounded = useTransform(count, v => Math.round(v))
  const ref = useRef(null)

  useEffect(() => {
    const controls = animate(count, score, { duration: 1.6, ease: 'easeOut' })
    return controls.stop
  }, [score])

  return <motion.span ref={ref}>{rounded}</motion.span>
}

export default function ResultPanel({ result, onReset }) {
  const { credit_score, default_probability, explanations, loan_status, risk_band } = result
  const risk     = RISK[risk_band] || RISK['Medium Risk']
  const accepted = loan_status === 'Accepted'
  const pct      = Math.min((credit_score / 850) * 100, 100)
  const offset   = CIRCUMFERENCE * (1 - pct / 100)

  // Borderline = accepted but not Low Risk, or rejected but not High Risk
  const isBorderline = (accepted && risk_band !== 'Low Risk') || (!accepted && risk_band !== 'High Risk')

  // strip ✓/✗ prefix the backend may add → we render our own indicators
  const clean = s => s.replace(/^[✓✗]\s*/, '').trim()

  const headline = accepted
    ? isBorderline
      ? <>Your application was <em>conditionally accepted</em></>
      : <>Your application looks <em>strong</em></>
    : isBorderline
      ? <>Your application <em>needs improvement</em></>
      : <>Your application needs <em>attention</em></>

  const subline = accepted
    ? isBorderline
      ? 'Your profile meets the minimum threshold, but there are areas that could strengthen your position. Review the factors below.'
      : 'Based on your financial profile, our model predicts a high likelihood of loan acceptance.'
    : isBorderline
      ? 'Your profile is close to the acceptance threshold. Addressing the factors below could improve your chances significantly.'
      : 'Our model identified several risk factors in your profile. See the breakdown below.'

  return (
    <div className={`result-page ${accepted ? 'mood-accepted' : 'mood-rejected'}`}>

      {/* ── BACKGROUND ── */}
      <div className="res-bg">
        <img src={FINANCE_IMG} alt="" className="res-bg-img" />
        <div className="res-bg-overlay" />
        <div className="res-bg-grid" />
      </div>

      {/* ── FLOATING RINGS ── */}
      <div className="res-rings" aria-hidden>
        {[...Array(5)].map((_, i) => (
          <motion.div key={i} className="res-ring" style={{ '--ri': i }}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1.2, delay: i * 0.15, ease: 'easeOut' }}
          />
        ))}
      </div>

      <div className="res-wrap">

        {/* ── NAV BAR ── */}
        <motion.nav className="res-nav"
          initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="res-logo">
            <LogoMark />
            <span>NexScore</span>
          </div>
          <button className="res-new-btn" onClick={onReset}>
            New Assessment
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </motion.nav>

        {/* ── HERO VERDICT ── */}
        <motion.section className="res-hero"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.1 }}
        >
          <motion.div className={`res-verdict-card ${accepted ? 'vc-accepted' : 'vc-rejected'}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          >
            <span className="vc-label">Your Loan is</span>
            <div className="vc-status-row">
              <span className="vc-icon">{accepted ? '✓' : '✕'}</span>
              <span className="vc-status">{loan_status}</span>
            </div>
          </motion.div>

          <motion.h1 className="res-headline"
            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
            {headline}
          </motion.h1>

          <motion.p className="res-subline"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            {subline}
          </motion.p>

          {/* ── KEY METRICS ROW ── */}
          <motion.div className="res-metrics"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            {/* Score arc */}
            <div className="metric-arc">
              <svg viewBox="0 0 120 120" className="arc-svg">
                <circle cx="60" cy="60" r="54" fill="none"
                  stroke="rgba(255,255,255,0.06)" strokeWidth="7"/>
                <motion.circle cx="60" cy="60" r="54" fill="none"
                  stroke="url(#arcGrad)" strokeWidth="7"
                  strokeLinecap="round"
                  strokeDasharray={CIRCUMFERENCE}
                  strokeDashoffset={CIRCUMFERENCE}
                  transform="rotate(-90 60 60)"
                  animate={{ strokeDashoffset: offset }}
                  transition={{ duration: 1.6, delay: 0.6, ease: 'easeOut' }}
                />
                <defs>
                  <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor={accepted ? '#5ba3d0' : '#f87171'}/>
                    <stop offset="100%" stopColor={accepted ? '#b8973a' : '#fbbf24'}/>
                  </linearGradient>
                </defs>
              </svg>
              <div className="arc-center">
                <div className="arc-score">
                  <AnimatedScore score={credit_score} />
                </div>
                <div className="arc-label">Credit Score</div>
              </div>
            </div>

            <div className="metric-divider" />

            <div className="metric-stat">
              <div className="ms-value">{(default_probability * 100).toFixed(1)}%</div>
              <div className="ms-label">Default Probability</div>
            </div>

            <div className="metric-divider" />

            <div className="metric-stat">
              <div className="ms-value" style={{ color: risk.color }}>{risk_band}</div>
              <div className="ms-label">Risk Classification</div>
            </div>
          </motion.div>
        </motion.section>

        {/* ── FACTORS ── */}
        <motion.section className="res-factors"
          initial={{ opacity: 0, y: 32 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.55 }}
        >
          {explanations.positive.length > 0 && (
            <div className="factor-col">
              <div className="factor-col-header positive-header">
                <div className="fch-icon">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8l5-5 5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <span>Strengths</span>
                <span className="fch-count">{explanations.positive.length}</span>
              </div>
              <ul className="factor-list">
                {explanations.positive.map((item, i) => (
                  <motion.li key={i} className="factor-item positive-item"
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.65 + i * 0.05 }}
                  >
                    <span className="fi-dot pos-dot" />
                    <span>{clean(item)}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          )}

          {explanations.negative.length > 0 && (
            <div className="factor-col">
              <div className="factor-col-header negative-header">
                <div className="fch-icon neg-fch-icon">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8l5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <span>Areas of Concern</span>
                <span className="fch-count neg-count">{explanations.negative.length}</span>
              </div>
              <ul className="factor-list">
                {explanations.negative.map((item, i) => (
                  <motion.li key={i} className="factor-item negative-item"
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.65 + i * 0.05 }}
                  >
                    <span className="fi-dot neg-dot" />
                    <span>{clean(item)}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          )}
        </motion.section>

        {/* ── FOOTER ── */}
        <motion.footer className="res-footer"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          <p className="res-disclaimer">
            This assessment is for informational purposes only and does not constitute financial advice.
          </p>
          <button className="res-reset-btn" onClick={onReset}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M2 8a6 6 0 1 1 1.5 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M2 12V8h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Start a new assessment
          </button>
        </motion.footer>

      </div>
    </div>
  )
}

function LogoMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="#b8973a" strokeWidth="1.5"/>
      <path d="M8 14 L14 8 L20 14 L14 20 Z" fill="none" stroke="#b8973a" strokeWidth="1.5"/>
      <circle cx="14" cy="14" r="3" fill="#b8973a"/>
    </svg>
  )
}
