import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import ResultPanel from '../components/ResultPanel'
import MonthYearPicker from '../components/MonthYearPicker'
import '../styles/Apply.css'

const STEPS = [
  {
    id: 'loan',
    title: 'Loan Details',
    subtitle: "Tell us about the loan you're seeking.",
    icon: '◈',
    fields: [
      { key: 'loan_amnt',   label: 'Loan Amount',         type: 'number', prefix: '$', placeholder: '15000',  tooltip: 'The total amount of money you want to borrow.' },
      { key: 'term',        label: 'Loan Term',           type: 'select', options: ['36 months','60 months'],  tooltip: 'Duration over which you will repay the loan.' },
      { key: 'purpose',     label: 'Loan Purpose',        type: 'select', options: ['debt_consolidation','credit_card','home_improvement','major_purchase','medical','car','small_business','vacation','moving','house','renewable_energy','educational','other'], tooltip: 'The primary reason you are taking this loan.' },
      { key: 'int_rate',    label: 'Interest Rate',       type: 'number', suffix: '%', placeholder: '12.5',   tooltip: 'Annual interest rate charged on the loan, expressed as a percentage.' },
      { key: 'installment', label: 'Monthly Installment', type: 'number', prefix: '$', placeholder: '502.30', tooltip: 'Fixed monthly payment amount including principal and interest.', optional: true, hint: 'Leave blank if unknown' },
    ]
  },
  {
    id: 'personal',
    title: 'Personal & Employment',
    subtitle: 'A bit about your financial background.',
    icon: '◇',
    fields: [
      { key: 'annual_inc',          label: 'Annual Income',        type: 'number', prefix: '$', placeholder: '75000', tooltip: 'Your total yearly income before taxes.' },
      { key: 'emp_length',          label: 'Employment Length',    type: 'select', options: ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years'], tooltip: 'How long you have been employed at your current or most recent job.' },
      { key: 'home_ownership',      label: 'Home Ownership',       type: 'select', options: ['RENT','OWN','MORTGAGE','OTHER'], tooltip: 'Your current housing situation → whether you rent, own outright, or have a mortgage.' },
      { key: 'verification_status', label: 'Income Verification',  type: 'select', options: ['Verified','Source Verified','Not Verified'], tooltip: 'Whether your income has been independently verified by the lender.' },
      { key: 'dti',                 label: 'Debt-to-Income Ratio', type: 'number', suffix: '%', placeholder: '18.5', tooltip: 'Your total monthly debt payments divided by gross monthly income, as a percentage. Lower is better.', hint: 'Monthly debt ÷ monthly income × 100' },
    ]
  },
  {
    id: 'credit',
    title: 'Credit History',
    subtitle: 'Your credit profile helps us assess risk accurately.',
    icon: '◉',
    fields: [
      { key: 'fico_range_low',   label: 'FICO Score (Low)',              type: 'number', placeholder: '700',      tooltip: 'The lower bound of your FICO credit score range. Scores range from 300 to 850.' },
      { key: 'fico_range_high',  label: 'FICO Score (High)',             type: 'number', placeholder: '704',      tooltip: 'The upper bound of your FICO credit score range.' },
      { key: 'earliest_cr_line', label: 'Earliest Credit Line',          type: 'monthyear', placeholder: 'Jan-2010', tooltip: 'The month and year your oldest credit account was opened. Longer history is generally better.' },
      { key: 'inq_last_6mths',   label: 'Credit Inquiries (Last 6 mo.)', type: 'number', placeholder: '1',        tooltip: 'Number of hard credit inquiries in the past 6 months. More inquiries can lower your score.' },
      { key: 'delinq_2yrs',      label: 'Delinquencies (Last 2 yrs)',    type: 'number', placeholder: '0',        tooltip: 'Number of times you were 30+ days past due on a payment in the last 2 years.', optional: true },
    ]
  },
  {
    id: 'accounts',
    title: 'Account Summary',
    subtitle: 'Details about your existing credit accounts.',
    icon: '⬡',
    fields: [
      { key: 'open_acc',   label: 'Open Credit Accounts',  type: 'number', placeholder: '8',    tooltip: 'Total number of currently open credit lines (cards, loans, etc.).' },
      { key: 'total_acc',  label: 'Total Credit Accounts', type: 'number', placeholder: '20',   tooltip: 'Total number of credit accounts ever opened, including closed ones.' },
      { key: 'revol_bal',  label: 'Revolving Balance',     type: 'number', prefix: '$', placeholder: '5000', tooltip: 'Total outstanding balance across all revolving credit accounts (e.g. credit cards).' },
      { key: 'revol_util', label: 'Revolving Utilization', type: 'number', suffix: '%', placeholder: '35.0', tooltip: 'Percentage of your revolving credit limit currently in use. Below 30% is ideal.' },
      { key: 'pub_rec',    label: 'Public Records',        type: 'number', placeholder: '0',    tooltip: 'Number of derogatory public records such as bankruptcies or tax liens.', optional: true },
    ]
  },
  {
    id: 'advanced',
    title: 'Final Details',
    subtitle: 'Almost there → just a few more data points.',
    icon: '✦',
    fields: [
      { key: 'pub_rec_bankruptcies', label: 'Public Record Bankruptcies',         type: 'number', placeholder: '0', tooltip: 'Number of public record bankruptcies on file.', optional: true },
      { key: 'delinq_amnt',          label: 'Delinquent Amount',                  type: 'number', prefix: '$', placeholder: '0', tooltip: 'Total dollar amount currently past due across all accounts.', optional: true },
      { key: 'num_tl_30dpd',         label: 'Accounts 30 Days Past Due',          type: 'number', placeholder: '0', tooltip: 'Number of accounts currently 30 days past due.', optional: true },
      { key: 'num_tl_90g_dpd_24m',   label: 'Accounts 90+ Days Past Due (24mo)', type: 'number', placeholder: '0', tooltip: 'Number of accounts that were 90+ days past due in the last 24 months.', optional: true },
      { key: 'issue_d',              label: 'Issue Date',                         type: 'monthyear', placeholder: 'Mar-2026', tooltip: 'The month and year the loan is expected to be issued.' },
    ]
  }
]

const DEFAULT_VALUES = {
  loan_amnt: '', term: '36 months', int_rate: '', installment: '',
  purpose: 'debt_consolidation', annual_inc: '', dti: '',
  home_ownership: 'RENT', emp_length: '5 years',
  verification_status: 'Verified', inq_last_6mths: '', delinq_2yrs: '',
  open_acc: '', total_acc: '', revol_bal: '', revol_util: '',
  pub_rec: '', pub_rec_bankruptcies: '', earliest_cr_line: '',
  fico_range_low: '', fico_range_high: '', issue_d: 'Mar-2026',
  delinq_amnt: '', num_tl_30dpd: '', num_tl_90g_dpd_24m: ''
}

const slideVariants = {
  enter: (dir) => ({ x: dir > 0 ? 56 : -56, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit:  (dir) => ({ x: dir > 0 ? -56 : 56, opacity: 0 }),
}

function validate(fields, values) {
  const errors = {}
  fields.forEach(f => {
    if (!f.optional && f.type !== 'select') {
      const v = values[f.key]
      if (v === '' || v === null || v === undefined) {
        errors[f.key] = 'This field is required'
      }
    }
  })
  return errors
}

export default function ApplyPage() {
  const navigate = useNavigate()
  const [step, setStep]       = useState(0)
  const [dir, setDir]         = useState(1)
  const [values, setValues]   = useState(DEFAULT_VALUES)
  const [touched, setTouched] = useState({})
  const [errors, setErrors]   = useState({})
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState(null)

  const current  = STEPS[step]
  const progress = (step / STEPS.length) * 100

  const set = (key, val) => {
    setValues(v => ({ ...v, [key]: val }))
    if (errors[key]) setErrors(e => { const n = { ...e }; delete n[key]; return n })
  }

  const touch = (key) => setTouched(t => ({ ...t, [key]: true }))

  const next = () => {
    const errs = validate(current.fields, values)
    if (Object.keys(errs).length > 0) {
      setErrors(errs)
      const t = {}
      current.fields.forEach(f => { if (!f.optional) t[f.key] = true })
      setTouched(prev => ({ ...prev, ...t }))
      return
    }
    setErrors({})
    setDir(1)
    setStep(s => Math.min(s + 1, STEPS.length - 1))
  }

  const prev = () => {
    setErrors({})
    setDir(-1)
    setStep(s => Math.max(s - 1, 0))
  }

  const submit = async () => {
    const errs = validate(current.fields, values)
    if (Object.keys(errs).length > 0) {
      setErrors(errs)
      const t = {}
      current.fields.forEach(f => { if (!f.optional) t[f.key] = true })
      setTouched(prev => ({ ...prev, ...t }))
      return
    }
    setLoading(true)
    setApiError(null)
    const payload = { ...values }
    const numFields = ['loan_amnt','int_rate','installment','annual_inc','dti','inq_last_6mths','delinq_2yrs','open_acc','total_acc','revol_bal','revol_util','pub_rec','pub_rec_bankruptcies','fico_range_low','fico_range_high','delinq_amnt','num_tl_30dpd','num_tl_90g_dpd_24m']
    numFields.forEach(k => { if (payload[k] !== '') payload[k] = Number(payload[k]) })
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.error || `Server error ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setApiError(e.message || 'Unable to reach the server. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  if (result) return (
    <ResultPanel
      result={result}
      onReset={() => { setResult(null); setStep(0); setValues(DEFAULT_VALUES); setTouched({}); setErrors({}) }}
    />
  )

  return (
    <div className="apply-page">

      {/* ── SIDEBAR ── */}
      <aside className="apply-sidebar">
        <div className="sidebar-logo" onClick={() => navigate('/')}>
          <LogoMark />
          <span>NexScore</span>
        </div>

        <div className="sidebar-intro">
          <p>Complete each section to receive your personalised credit assessment.</p>
        </div>

        <div className="sidebar-steps">
          {STEPS.map((s, i) => (
            <div key={s.id} className={`sidebar-step ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}>
              <div className="step-dot">
                {i < step ? <CheckIcon /> : <span>{i + 1}</span>}
              </div>
              <div className="step-info">
                <div className="step-name">{s.title}</div>
                {i === step && <div className="step-sub">{s.subtitle}</div>}
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="progress-label">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="progress-track">
            <motion.div
              className="progress-fill"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
          <p className="sidebar-secure">
            <LockIcon /> Your data is never stored
          </p>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="apply-main">

        {/* top bar */}
        <div className="apply-topbar">
          <button className="topbar-back" onClick={() => navigate('/')}>
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
              <path d="M10 4L6 8l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Back to home
          </button>
          <div className="topbar-steps">
            {STEPS.map((_, i) => (
              <div key={i} className={`topbar-pip ${i < step ? 'done' : ''} ${i === step ? 'active' : ''}`} />
            ))}
          </div>
          <div className="topbar-counter">Step {step + 1} / {STEPS.length}</div>
        </div>

        {/* form area */}
        <div className="form-area">
          <AnimatePresence mode="wait" custom={dir}>
            <motion.div
              key={step}
              custom={dir}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.38, ease: [0.22, 1, 0.36, 1] }}
              className="form-step"
            >
              <div className="step-header">
                <div className="step-icon-badge">{current.icon}</div>
                <div>
                  <h2>{current.title}</h2>
                  <p>{current.subtitle}</p>
                </div>
              </div>

              <div className="fields-grid">
                {current.fields.map(field => {
                  const hasError = touched[field.key] && errors[field.key]
                  return (
                    <div key={field.key} className={`field-group ${hasError ? 'has-error' : ''}`}>
                      <label htmlFor={field.key}>
                        {field.label}
                        {field.optional
                          ? <span className="optional-tag">optional</span>
                          : <span className="required-star">*</span>
                        }
                        {field.tooltip && (
                          <span className="field-tooltip-wrap">
                            <span className="field-info-icon">i</span>
                            <span className="field-tooltip-box">{field.tooltip}</span>
                          </span>
                        )}
                      </label>

                      {field.type === 'select' ? (
                        <select
                          id={field.key}
                          value={values[field.key]}
                          onChange={e => set(field.key, e.target.value)}
                          onBlur={() => touch(field.key)}
                        >
                          {field.options.map(o => (
                            <option key={o} value={o}>{o.replace(/_/g, ' ')}</option>
                          ))}
                        </select>
                      ) : field.type === 'monthyear' ? (
                        <MonthYearPicker
                          id={field.key}
                          value={values[field.key]}
                          onChange={val => { set(field.key, val); touch(field.key) }}
                          placeholder={field.placeholder}
                          maxDate={field.key === 'earliest_cr_line' ? 'Mar-2026' : undefined}
                        />
                      ) : (
                        <div className="input-wrap">
                          {field.prefix && <span className="input-affix input-prefix">{field.prefix}</span>}
                          <input
                            id={field.key}
                            type={field.type}
                            placeholder={field.placeholder}
                            value={values[field.key]}
                            onChange={e => set(field.key, e.target.value)}
                            onBlur={() => touch(field.key)}
                            className={`${field.prefix ? 'has-prefix' : ''} ${field.suffix ? 'has-suffix' : ''}`}
                          />
                          {field.suffix && <span className="input-affix input-suffix">{field.suffix}</span>}
                        </div>
                      )}

                      {hasError
                        ? <div className="field-error">{errors[field.key]}</div>
                        : field.hint && <div className="field-hint">{field.hint}</div>
                      }
                    </div>
                  )
                })}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {apiError && (
          <motion.div className="api-error" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
            <span>⚠</span> {apiError}
          </motion.div>
        )}

        {/* nav */}
        <div className="form-nav">
          <button className="btn-back" onClick={prev} disabled={step === 0}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
              <path d="M10 4L6 8l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Back
          </button>

          <div className="nav-right">
            {Object.keys(errors).length > 0 && (
              <motion.span className="nav-error-hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                Please fill required fields
              </motion.span>
            )}

            {step < STEPS.length - 1 ? (
              <button className="btn-next" onClick={next}>
                Continue
                <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                  <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            ) : (
              <button className="btn-submit" onClick={submit} disabled={loading}>
                {loading ? <span className="spinner" /> : (
                  <>
                    Analyze My Application
                    <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
                      <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

function LogoMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="#b8973a" strokeWidth="1.5"/>
      <path d="M8 14 L14 8 L20 14 L14 20 Z" fill="none" stroke="#b8973a" strokeWidth="1.5"/>
      <circle cx="14" cy="14" r="3" fill="#b8973a"/>
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 10 10" fill="none">
      <path d="M2 5l2.5 2.5L8 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
}

function LockIcon() {
  return (
    <svg width="11" height="11" viewBox="0 0 16 16" fill="none" style={{ display: 'inline', marginRight: 5 }}>
      <rect x="3" y="7" width="10" height="8" rx="2" stroke="currentColor" strokeWidth="1.4"/>
      <path d="M5 7V5a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>
  )
}
