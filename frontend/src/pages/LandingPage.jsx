import { useNavigate } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import { useRef } from 'react'
import '../styles/Landing.css'

const HERO_IMG   = 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1600&q=80&auto=format&fit=crop'
const CITY_IMG   = 'https://images.unsplash.com/photo-1444653614773-995cb1ef9efa?w=1400&q=75&auto=format&fit=crop'
const DESK_IMG   = 'https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=900&q=80&auto=format&fit=crop'
const CHART_IMG  = 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=900&q=80&auto=format&fit=crop'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 32 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }
})

const fadeIn = (delay = 0) => ({
  initial: { opacity: 0 },
  whileInView: { opacity: 1 },
  viewport: { once: true },
  transition: { duration: 0.8, delay }
})

export default function LandingPage() {
  const navigate = useNavigate()
  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const heroImgY = useTransform(scrollYProgress, [0, 1], ['0%', '20%'])
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0])

  return (
    <div className="landing">

      {/* ── NAV ── */}
      <nav className="landing-nav">
        <div className="nav-logo">
          <LogoMark />
          <span>NexScore</span>
        </div>
        <div className="nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="#features">Features</a>
          <a href="#trust">Why us</a>
        </div>
        <button className="nav-cta" onClick={() => navigate('/apply')}>Check Eligibility</button>
      </nav>

      {/* ── HERO ── */}
      <section className="hero-section" ref={heroRef}>
        <motion.div className="hero-bg-img" style={{ y: heroImgY }}>
          <img src={HERO_IMG} alt="" />
          <div className="hero-img-overlay" />
        </motion.div>

        <div className="hero-grid-lines">
          {[...Array(8)].map((_, i) => (
            <motion.div key={i} className="grid-line"
              style={{ '--i': i }}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              transition={{ duration: 1.5, delay: 0.3 + i * 0.1 }}
            />
          ))}
        </div>

        <motion.div className="hero-content" style={{ opacity: heroOpacity }}>
          <motion.div className="hero-badge"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="badge-dot" />
            AI-Powered Credit Intelligence
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 36 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.12, ease: [0.22, 1, 0.36, 1] }}
          >
            Know Your Loan<br />
            <em className="gold-italic">Eligibility</em> Before<br />
            You Apply
          </motion.h1>

          <motion.p className="hero-sub"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.24 }}
          >
            Our intelligent credit model analyzes your full financial profile and
            delivers a transparent, data-driven assessment — in seconds.
          </motion.p>

          <motion.div className="hero-actions"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.36 }}
          >
            <button className="btn-primary" onClick={() => navigate('/apply')}>
              Start Your Assessment
              <ArrowRight />
            </button>
            <a href="#how-it-works" className="btn-ghost">See how it works</a>
          </motion.div>

          <motion.div className="hero-stats"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.55 }}
          >
            {[['< 3s', 'Result time'], ['10', 'Model features'], ['98.67%', 'Model accuracy']].map(([val, lbl]) => (
              <div key={lbl} className="hero-stat">
                <span className="stat-val">{val}</span>
                <span className="stat-lbl">{lbl}</span>
              </div>
            ))}
          </motion.div>
        </motion.div>

        <motion.div className="hero-card-wrap"
          initial={{ opacity: 0, x: 48, y: 16 }} animate={{ opacity: 1, x: 0, y: 0 }}
          transition={{ duration: 0.9, delay: 0.45, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="floating-score-card">
            <div className="fsc-header">
              <span className="fsc-label">Credit Assessment</span>
              <span className="fsc-live"><span className="live-dot" />Live</span>
            </div>
            <div className="fsc-score-row">
              <div className="fsc-score">849</div>
              <div className="fsc-badge accepted">Accepted</div>
            </div>
            <div className="fsc-risk">Low Risk Profile</div>
            <div className="fsc-bar-wrap">
              <div className="fsc-bar-labels"><span>Poor</span><span>Excellent</span></div>
              <div className="fsc-bar">
                <motion.div className="fsc-fill"
                  initial={{ width: 0 }} animate={{ width: '92%' }}
                  transition={{ duration: 1.4, delay: 1, ease: 'easeOut' }}
                />
                <motion.div className="fsc-thumb"
                  initial={{ left: 0 }} animate={{ left: '92%' }}
                  transition={{ duration: 1.4, delay: 1, ease: 'easeOut' }}
                />
              </div>
            </div>
            <div className="fsc-factors">
              {['Strong credit history', 'Low utilization', 'Stable income'].map(f => (
                <div key={f} className="fsc-factor"><span className="fsc-check">✓</span>{f}</div>
              ))}
            </div>
          </div>

          <motion.div className="floating-chip chip-1"
            animate={{ y: [0, -8, 0] }} transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <span className="chip-icon">📈</span> Score improved
          </motion.div>
          <motion.div className="floating-chip chip-2"
            animate={{ y: [0, 8, 0] }} transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut', delay: 0.8 }}
          >
            <span className="chip-icon">🔒</span> Secure analysis
          </motion.div>
        </motion.div>

        <div className="hero-scroll-hint">
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.8, repeat: Infinity }}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M5 8l5 5 5-5" stroke="rgba(245,240,232,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </motion.div>
        </div>
      </section>

      {/* ── TRUST STRIP ── */}
      <section className="trust-strip">
        <p className="trust-label">Trusted financial intelligence</p>
        <div className="trust-items">
          {['256-bit Encryption', 'No Data Stored', 'Instant Results', 'Transparent Scoring', 'ML-Powered'].map(t => (
            <div key={t} className="trust-item">
              <span className="trust-dot" />{t}
            </div>
          ))}
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="how-section" id="how-it-works">
        <div className="how-inner">
          <motion.div className="section-label" {...fadeUp(0)}>Process</motion.div>
          <motion.h2 {...fadeUp(0.05)}>Three steps to clarity</motion.h2>
          <motion.p className="section-sub" {...fadeUp(0.1)}>
            No paperwork. No waiting. Just your data and an honest answer.
          </motion.p>

          <div className="steps-grid">
            {[
              { num: '01', title: 'Enter your details', desc: 'Fill in your loan requirements and financial profile through our guided, step-by-step form.', color: 'var(--blue-pale)' },
              { num: '02', title: 'AI analyzes your profile', desc: 'Our XGBoost model evaluates 10 financial indicators — FICO score, DTI ratio, employment stability, and more.', color: 'var(--gold-pale)' },
              { num: '03', title: 'Receive your result', desc: 'Get a clear acceptance decision, credit score, risk band, and a full breakdown of influencing factors.', color: 'var(--cream-mid)' },
            ].map((s, i) => (
              <motion.div key={s.num} className="step-card" style={{ '--step-bg': s.color }} {...fadeUp(i * 0.12)}>
                <div className="step-num">{s.num}</div>
                <h3>{s.title}</h3>
                <p>{s.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── SPLIT SECTION: IMAGE + TEXT ── */}
      <section className="split-section dark-split">
        <motion.div className="split-img-col" {...fadeIn(0)}>
          <img src={DESK_IMG} alt="Financial analysis" />
          <div className="split-img-overlay" />
          <div className="split-img-card">
            <div className="sic-label">Default Probability</div>
            <div className="sic-value">0.0%</div>
            <div className="sic-sub">Exceptional profile</div>
          </div>
        </motion.div>
        <motion.div className="split-text-col" {...fadeUp(0.15)}>
          <div className="section-label light">Intelligence</div>
          <h2>More than a<br /><em>credit check</em></h2>
          <p>
            Traditional lenders look at a handful of metrics. NexScore evaluates
            your complete financial picture — employment history, revolving credit,
            public records, and more — to give you a result you can actually trust.
          </p>
          <ul className="split-list">
            {['10 financial features analyzed', 'Explainable AI — see every factor', 'Risk band classification', 'Instant, no-impact assessment'].map(item => (
              <li key={item}><span className="list-check">✦</span>{item}</li>
            ))}
          </ul>
          <button className="btn-primary-dark" onClick={() => navigate('/apply')}>
            Try it now <ArrowRight />
          </button>
        </motion.div>
      </section>

      {/* ── FEATURES ── */}
      <section className="features-section" id="features">
        <div className="features-inner">
          <motion.div className="section-label" {...fadeUp(0)}>Features</motion.div>
          <motion.h2 {...fadeUp(0.05)}>Built for financial clarity</motion.h2>

          <div className="features-grid">
            {[
              { icon: '⚡', title: 'Instant Analysis', desc: 'Results delivered in under 3 seconds. No waiting, no callbacks.', bg: 'var(--sky)' },
              { icon: '🔍', title: 'Transparent Scoring', desc: 'Every factor that influenced your result is clearly explained.', bg: 'var(--gold-pale)' },
              { icon: '🔒', title: 'Secure & Private', desc: 'Your data is never stored or shared. Analysis happens in real time.', bg: 'var(--cream-mid)' },
              { icon: '📊', title: 'Risk Classification', desc: 'Low, Medium, or High risk — with a full breakdown of what it means.', bg: 'var(--blue-pale)' },
              { icon: '🧠', title: 'ML-Powered Model', desc: 'XGBoost classifier trained on 160,000 real loan records with 98.67% accuracy.', bg: 'var(--cream-mid)' },
              { icon: '📋', title: 'Detailed Report', desc: 'Positive and negative factors listed so you know exactly where you stand.', bg: 'var(--gold-pale)' },
            ].map((f, i) => (
              <motion.div key={f.title} className="feat-card" style={{ '--feat-bg': f.bg }} {...fadeUp(i * 0.07)}>
                <div className="feat-icon">{f.icon}</div>
                <h4>{f.title}</h4>
                <p>{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CITY BANNER ── */}
      <section className="city-banner" id="trust">
        <div className="city-bg">
          <img src={CITY_IMG} alt="" />
          <div className="city-overlay" />
        </div>
        <div className="city-content">
          <motion.div className="section-label light" {...fadeUp(0)}>Why NexScore</motion.div>
          <motion.h2 {...fadeUp(0.08)}>
            Finance decisions deserve<br /><em>better data</em>
          </motion.h2>
          <motion.p className="city-sub" {...fadeUp(0.14)}>
            We built NexScore because loan applicants deserve to understand their
            standing before they walk into a bank. Knowledge is leverage.
          </motion.p>
          <motion.button className="btn-primary" onClick={() => navigate('/apply')} {...fadeUp(0.2)}>
            Check My Eligibility <ArrowRight />
          </motion.button>
        </div>
      </section>

      {/* ── CHART SECTION ── */}
      <section className="chart-section">
        <motion.div className="chart-text" {...fadeUp(0)}>
          <div className="section-label">Accuracy</div>
          <h2>Data you can<br />act on</h2>
          <p>
            Our model surfaces the exact factors driving your credit assessment —
            so you can improve your profile, not just wonder about the outcome.
          </p>
          <div className="accuracy-stats">
            {[['850', 'Max credit score'], ['98.67%', 'Model accuracy'], ['10', 'Features analyzed']].map(([v, l]) => (
              <div key={l} className="acc-stat">
                <div className="acc-val">{v}</div>
                <div className="acc-lbl">{l}</div>
              </div>
            ))}
          </div>
        </motion.div>
        <motion.div className="chart-img-col" {...fadeIn(0.15)}>
          <img src={CHART_IMG} alt="Financial charts" />
          <div className="chart-img-overlay" />
        </motion.div>
      </section>

      {/* ── CTA BANNER ── */}
      <section className="cta-section">
        <motion.div className="cta-inner" {...fadeUp(0)}>
          <div className="cta-deco-line" />
          <h2>Ready to know where<br />you stand?</h2>
          <p>It takes less than 5 minutes. No credit impact. No sign-up required.</p>
          <button className="btn-primary-lg" onClick={() => navigate('/apply')}>
            Start Free Assessment <ArrowRight />
          </button>
          <div className="cta-deco-line" />
        </motion.div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="landing-footer">
        <div className="footer-logo">
          <LogoMark />
          <span>NexScore</span>
        </div>
        <p className="footer-copy">
          © 2026 NexScore™ — For informational purposes only. Not financial advice.
        </p>
        <p className="footer-credit">
          Crafted with ❤️ by{' '}
          <a href="https://portfolio-website-umber-pi-71.vercel.app/" target="_blank" rel="noopener noreferrer">
            Sakshi Paygude
          </a>
          ™
        </p>
      </footer>

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

function ArrowRight() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
      <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}
