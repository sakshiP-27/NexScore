import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './MonthYearPicker.css'

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

// Parse "Jan-2010" → { month: 0, year: 2010 } or null
function parse(val) {
  if (!val) return null
  const [m, y] = val.split('-')
  const mi = MONTHS.indexOf(m)
  const yi = parseInt(y, 10)
  if (mi === -1 || isNaN(yi)) return null
  return { month: mi, year: yi }
}

// Format { month, year } → "Jan-2010"
function fmt({ month, year }) {
  return `${MONTHS[month]}-${year}`
}

export default function MonthYearPicker({ id, value, onChange, placeholder, maxDate }) {
  const parsed   = parse(value)
  const now      = new Date()
  const maxYear  = maxDate ? parse(maxDate)?.year  ?? now.getFullYear() : now.getFullYear()
  const maxMonth = maxDate ? parse(maxDate)?.month ?? now.getMonth()    : now.getMonth()

  const [open, setOpen]       = useState(false)
  const [viewYear, setViewYear] = useState(parsed?.year ?? now.getFullYear())
  const ref = useRef(null)

  // close on outside click
  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // sync viewYear when value changes externally
  useEffect(() => {
    if (parsed) setViewYear(parsed.year)
  }, [value])

  const isDisabled = (mi) => {
    if (viewYear > maxYear) return true
    if (viewYear === maxYear && mi > maxMonth) return true
    return false
  }

  const select = (mi) => {
    if (isDisabled(mi)) return
    onChange(fmt({ month: mi, year: viewYear }))
    setOpen(false)
  }

  const prevYear = () => setViewYear(y => y - 1)
  const nextYear = () => { if (viewYear < maxYear) setViewYear(y => y + 1) }

  return (
    <div className="myp-wrap" ref={ref}>
      <button
        type="button"
        id={id}
        className={`myp-trigger ${value ? 'has-value' : ''}`}
        onClick={() => setOpen(o => !o)}
        aria-haspopup="true"
        aria-expanded={open}
      >
        <svg className="myp-cal-icon" width="15" height="15" viewBox="0 0 16 16" fill="none">
          <rect x="1" y="3" width="14" height="12" rx="2" stroke="currentColor" strokeWidth="1.3"/>
          <path d="M1 7h14" stroke="currentColor" strokeWidth="1.3"/>
          <path d="M5 1v3M11 1v3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
        </svg>
        <span>{value || placeholder || 'Select month & year'}</span>
        <svg className={`myp-chevron ${open ? 'open' : ''}`} width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            className="myp-dropdown"
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
          >
            {/* Year nav */}
            <div className="myp-year-row">
              <button type="button" className="myp-year-btn" onClick={prevYear}>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M8 2L4 6l4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              <span className="myp-year-label">{viewYear}</span>
              <button type="button" className="myp-year-btn" onClick={nextYear} disabled={viewYear >= maxYear}>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>

            {/* Month grid */}
            <div className="myp-months">
              {MONTHS.map((m, i) => {
                const disabled = isDisabled(i)
                const selected = parsed?.month === i && parsed?.year === viewYear
                return (
                  <button
                    key={m}
                    type="button"
                    className={`myp-month ${selected ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
                    onClick={() => select(i)}
                    disabled={disabled}
                  >
                    {m}
                  </button>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
