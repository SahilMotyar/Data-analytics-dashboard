import axios from 'axios'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'

/* ═══════════════════════════════════════════════════════════════════════
   Types
   ═══════════════════════════════════════════════════════════════════════ */

type UploadMeta = {
  upload_id: string
  file_name: string
  analysis_status: string
  sheet_names?: string[]
  active_sheet?: string | null
  warning?: string | null
  row_count_estimate?: number | null
  detected_columns?: string[]
}

type DatasetSummary = {
  rows: number
  columns: number
  memory_mb: number
  quality_score: number
  type_breakdown: Record<string, number>
  trends?: {
    top_correlations?: { col_a: string; col_b: string; r: number }[]
    numeric_summaries?: { column: string; mean: number; median: number; std: number; skew: number }[]
    category_dominance?: { column: string; top_value: string; top_pct: number; unique: number }[]
    time_ranges?: { column: string; from: string; to: string; span_days: number }[]
  }
}

type HealthDimension = {
  status: 'good' | 'warning' | 'critical' | 'na'
  dot: string
  label: string
}

type ColumnHealth = {
  missing: HealthDimension
  outliers: HealthDimension
  distribution: HealthDimension
  cardinality: HealthDimension
  overall: HealthDimension
}

type ColumnItem = {
  name: string
  inferred_type: string
  health?: ColumnHealth
  quality_flags: {
    missing_pct: number
    missing_warn: boolean
    constant: boolean
    high_cardinality: boolean
    potential_id: boolean
  }
}

type KeyFindings = {
  whats_in_this_data: string
  top_findings: string[]
  watch_out_for: string[]
  suggested_next_step: string
}

type AISummary = {
  what_does_this_look_like?: string
  anything_unusual?: string
  what_should_i_do?: string
}

type ColumnStats = Record<string, unknown> & {
  column: string
  inferred_type: string
  health?: ColumnHealth
  ai_summary?: AISummary
  chart_histogram_url?: string
  chart_boxplot_url?: string
  chart_bar_url?: string
  chart_line_url?: string
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type PreAnalysis = {
  smart_type_correction?: {
    reclassifications?: { column: string; from: string; to: string; reason: string; message?: string }[]
    excluded_columns?: { column: string; reason: string }[]
    high_missing_flags?: { column: string; missing_pct: number; message: string }[]
    percentage_flags?: { column: string; message: string }[]
  }
  correlation_analysis?: {
    pairs?: { col_a: string; col_b: string; pearson_r: number; spearman_rho: number; strength: string; non_linear_signal?: boolean }[]
    notable_negative?: { col_a: string; col_b: string; pearson_r: number }[]
    redundant_pairs?: { col_a: string; col_b: string; pearson_r: number }[]
  }
  group_difference_analysis?: {
    strongest_by_category?: {
      categorical_column: string
      numeric_column: string
      effect_size: number
      effect_label: string
      group_means: { group: string; mean: number; std?: number | null; count: number }[]
    }[]
  }
  outlier_characterisation?: {
    multi_column_anomalies?: { row_index: number; columns: string[] }[]
  }
  dataset_level_checks?: {
    duplicate_columns?: { col_a: string; col_b: string; message: string }[]
    sample_size_flags?: string[]
    class_imbalance?: { column: string; top_category: string; top_pct: number; message: string }[]
    datetime_checks?: { column: string; expected_frequency: string; missing_period_count: number; target_trend_direction?: string; target_trend_r2?: number | null }[]
  }
  name_quality_flag?: string | null
}

type GridPreview = {
  rows: { row_index: number; values: Record<string, unknown>; row_flags: string[]; cell_flags: Record<string, string[]> }[]
  columns: { name: string; final_type: string; reclassified: boolean; missing_pct: number }[]
}

/* ═══════════════════════════════════════════════════════════════════════
   Constants & helpers
   ═══════════════════════════════════════════════════════════════════════ */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const api = axios.create({ baseURL: API_BASE })

function chartUrl(path?: string) {
  if (!path) return ''
  return path.startsWith('http') ? path : `${API_BASE}${path}`
}

const fade = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
  transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
}

function qualityLabel(score: number) {
  if (score >= 95) return { text: 'Excellent', color: 'bg-emerald-100 text-emerald-800' }
  if (score >= 85) return { text: 'Good', color: 'bg-blue-100 text-blue-800' }
  if (score >= 70) return { text: 'Fair', color: 'bg-amber-100 text-amber-800' }
  return { text: 'Needs Work', color: 'bg-red-100 text-red-800' }
}

function gradeFromScore(score: number) {
  if (score >= 97) return 'A+'
  if (score >= 93) return 'A'
  if (score >= 90) return 'A-'
  if (score >= 87) return 'B+'
  if (score >= 83) return 'B'
  if (score >= 80) return 'B-'
  if (score >= 70) return 'C'
  if (score >= 60) return 'D'
  return 'F'
}

/* ═══════════════════════════════════════════════════════════════════════
   Application
   ═══════════════════════════════════════════════════════════════════════ */

function App() {
  /* ── State ── */
  const [file, setFile] = useState<File | null>(null)
  const [uploadMeta, setUploadMeta] = useState<UploadMeta | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [runningAnalysis, setRunningAnalysis] = useState(false)
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [keyFindings, setKeyFindings] = useState<KeyFindings | null>(null)
  const [columns, setColumns] = useState<ColumnItem[]>([])
  const [selectedColumn, setSelectedColumn] = useState<string>('')
  const [columnStats, setColumnStats] = useState<ColumnStats | null>(null)
  const [sharePath, setSharePath] = useState<string>('')
  const [error, setError] = useState('')
  const [viewMode, setViewMode] = useState<'simple' | 'analyst'>(() => {
    const stored = localStorage.getItem('datalens:view-mode')
    return stored === 'analyst' ? 'analyst' : 'simple'
  })
  const [showRawStats, setShowRawStats] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatOpen, setChatOpen] = useState(false)
  const [chatLoading, setChatLoading] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const [fabOpen, setFabOpen] = useState(false)
  const [analysisMode, setAnalysisMode] = useState<'auto' | 'quick' | 'focused' | 'full'>('auto')
  const [focusColumns, setFocusColumns] = useState<string[]>([])
  const [detectedColumns, setDetectedColumns] = useState<string[]>([])
  const [rowEstimate, setRowEstimate] = useState<number | null>(null)
  const [showAnalysisOptions, setShowAnalysisOptions] = useState(false)
  const [preAnalysis, setPreAnalysis] = useState<PreAnalysis | null>(null)
  const [gridPreview, setGridPreview] = useState<GridPreview | null>(null)
  const [gridSearch, setGridSearch] = useState('')
  const [gridOutliersOnly, setGridOutliersOnly] = useState(false)
  const [gridMissingOnly, setGridMissingOnly] = useState(false)
  const [compareA, setCompareA] = useState('')
  const [compareB, setCompareB] = useState('')
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)

  const uploadId = uploadMeta?.upload_id
  const canAnalyze = !!uploadId && uploadMeta?.analysis_status !== 'running' && uploadMeta?.analysis_status !== 'queued'
  const isLargeFile = (rowEstimate ?? 0) > 20_000

  useEffect(() => { localStorage.setItem('datalens:view-mode', viewMode) }, [viewMode])
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMessages])

  /* ── Upload ── */
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError('')
    const selected = acceptedFiles[0]
    if (!selected) return
    const valid = selected.name.endsWith('.csv') || selected.name.endsWith('.xlsx') || selected.name.endsWith('.xls')
    if (!valid) { setError('Only .csv, .xlsx, .xls are supported'); return }
    if (selected.size > 50 * 1024 * 1024) { setError('Max file size is 50MB'); return }
    setFile(selected)
    setUploadMeta(null); setSummary(null); setKeyFindings(null); setColumns([])
    setSelectedColumn(''); setColumnStats(null); setSharePath(''); setChatMessages([])
    setPreAnalysis(null); setGridPreview(null); setCompareResult(null)
    setCompareA(''); setCompareB('')
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, multiple: false })

  const uploadFile = async () => {
    if (!file) return
    setUploading(true); setError(''); setUploadProgress(0)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await api.post('/api/v1/uploads', formData, {
        onUploadProgress: (evt) => { if (evt.total) setUploadProgress(Math.round((evt.loaded * 100) / evt.total)) },
      })
      setUploadMeta(response.data.metadata)
    } catch { setError('Upload failed') }
    finally { setUploading(false) }
  }

  /* ── Detect columns after upload ── */
  useEffect(() => {
    if (!uploadMeta) return
    if (uploadMeta.row_count_estimate) setRowEstimate(uploadMeta.row_count_estimate)
    if (uploadMeta.detected_columns) setDetectedColumns(uploadMeta.detected_columns)
    // Fetch full meta (populates detected_columns on server)
    const detectCols = async () => {
      try {
        const res = await api.get(`/api/v1/uploads/${uploadMeta.upload_id}`)
        if (res.data.row_count_estimate) setRowEstimate(res.data.row_count_estimate)
        if (res.data.detected_columns) setDetectedColumns(res.data.detected_columns)
      } catch { /* ignore */ }
    }
    void detectCols()
    // Show options automatically for large files
    if ((uploadMeta.row_count_estimate ?? 0) > 20_000) {
      setShowAnalysisOptions(true)
      setAnalysisMode('quick')
    }
  }, [uploadMeta])

  /* ── Analysis ── */
  const fetchColumnStats = useCallback(async (name: string, id?: string) => {
    const targetId = id || uploadId
    if (!targetId) return
    setSelectedColumn(name)
    const response = await api.get(`/api/v1/analysis/${targetId}/columns/${encodeURIComponent(name)}/stats`)
    setColumnStats(response.data)
  }, [uploadId])

  const loadDashboard = useCallback(async (id: string) => {
    const [summaryRes, columnsRes, findingsRes, preRes, gridRes] = await Promise.all([
      api.get(`/api/v1/analysis/${id}/summary`),
      api.get(`/api/v1/analysis/${id}/columns`),
      api.get(`/api/v1/analysis/${id}/key-findings`),
      api.get(`/api/v1/analysis/${id}/pre-analysis`),
      api.get(`/api/v1/analysis/${id}/grid-preview`, { params: { limit: 120 } }),
    ])
    const loadedColumns: ColumnItem[] = columnsRes.data
    setSummary(summaryRes.data)
    setColumns(loadedColumns)
    setKeyFindings(findingsRes.data)
    setPreAnalysis(preRes.data)
    setGridPreview(gridRes.data)
    if (loadedColumns.length > 0) await fetchColumnStats(loadedColumns[0].name, id)
  }, [fetchColumnStats])

  const startAnalysis = async (modeOverride?: 'auto' | 'quick' | 'focused' | 'full') => {
    if (!uploadId) return
    setRunningAnalysis(true); setError('')
    const mode = modeOverride || analysisMode
    try {
      await api.post(`/api/v1/analysis/${uploadId}/start`, {
        active_sheet: uploadMeta?.active_sheet || undefined,
        mode,
        focus_columns: mode === 'focused' ? focusColumns : [],
      })
      setUploadMeta((prev) => (prev ? { ...prev, analysis_status: 'queued' } : prev))
    } catch { setError('Failed to start analysis'); setRunningAnalysis(false) }
  }

  useEffect(() => {
    if (!uploadId) return
    if (!['queued', 'running'].includes(uploadMeta?.analysis_status || '')) return
    const timer = window.setInterval(async () => {
      const statusRes = await api.get(`/api/v1/analysis/${uploadId}/status`)
      const newStatus = statusRes.data.status
      setUploadMeta((prev) => (prev ? { ...prev, analysis_status: newStatus } : prev))
      if (newStatus === 'completed') { window.clearInterval(timer); setRunningAnalysis(false); await loadDashboard(uploadId) }
      if (newStatus === 'failed') { window.clearInterval(timer); setRunningAnalysis(false); setError(statusRes.data.error || 'Analysis failed') }
    }, 2000)
    return () => window.clearInterval(timer)
  }, [uploadId, uploadMeta?.analysis_status, loadDashboard])

  const refreshGridPreview = useCallback(async () => {
    if (!uploadId || !summary) return
    const res = await api.get(`/api/v1/analysis/${uploadId}/grid-preview`, {
      params: {
        limit: 160,
        outliers_only: gridOutliersOnly,
        missing_only: gridMissingOnly,
      },
    })
    setGridPreview(res.data)
  }, [uploadId, summary, gridOutliersOnly, gridMissingOnly])

  useEffect(() => {
    void refreshGridPreview()
  }, [refreshGridPreview])

  const runCompare = async () => {
    if (!uploadId || !compareA || !compareB || compareA === compareB) return
    const res = await api.get(`/api/v1/analysis/${uploadId}/compare`, {
      params: { col_a: compareA, col_b: compareB },
    })
    setCompareResult(res.data)
  }

  /* ── Column type override ── */
  const updateType = async (name: string, newType: string) => {
    if (!uploadId) return
    await api.patch(`/api/v1/analysis/${uploadId}/columns/${encodeURIComponent(name)}/type`, { new_type: newType })
    setColumns((prev) => prev.map((item) => (item.name === name ? { ...item, inferred_type: newType } : item)))
    await fetchColumnStats(name)
  }

  /* ── Exports ── */
  const createPdf = async () => { if (!uploadId) return; const r = await api.post(`/api/v1/analysis/${uploadId}/export/pdf`); window.open(chartUrl(r.data.pdf_url), '_blank') }
  const createShare = async () => { if (!uploadId) return; const r = await api.post(`/api/v1/analysis/${uploadId}/share`); const path = `${API_BASE}${r.data.share_path}`; setSharePath(path); await navigator.clipboard.writeText(path) }
  const downloadCleanedCsv = () => { if (uploadId) window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/cleaned-csv`, '_blank') }
  const downloadExcel = (sample?: number) => { if (uploadId) window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/excel${sample ? `?sample=${sample}` : ''}`, '_blank') }
  const downloadStatsJson = () => { if (uploadId) window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/stats-json`, '_blank') }

  /* ── Chat ── */
  const starterPrompts = [
    'What stands out in this data?',
    'Are there any data quality issues?',
    'Summarize for a non-technical audience',
    'Which columns need attention?',
    'What would you investigate first?',
  ]

  const sendChatMessage = async (message: string) => {
    if (!uploadId || !message.trim() || chatLoading) return
    const userMessage: ChatMessage = { role: 'user' as const, content: message.trim() }
    const baseHistory = [...chatMessages, userMessage].slice(-10)
    setChatMessages(baseHistory); setChatInput(''); setChatLoading(true)
    try {
      const response = await fetch(`${API_BASE}/api/v1/analysis/${uploadId}/chat/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim(), history: baseHistory }),
      })
      if (!response.body) throw new Error('No stream')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantText = ''
      setChatMessages((prev) => [...prev, { role: 'assistant' as const, content: '' }].slice(-10))
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter((l) => l.startsWith('data: '))
        for (const line of lines) {
          const payload = JSON.parse(line.replace('data: ', '')) as { token?: string; done?: boolean }
          if (payload.token) {
            assistantText += payload.token
            setChatMessages((prev) => { const c = [...prev]; c[c.length - 1] = { role: 'assistant' as const, content: assistantText }; return c.slice(-10) })
          }
        }
      }
    } catch { setChatMessages((prev) => [...prev, { role: 'assistant' as const, content: 'I could not answer right now.' }].slice(-10)) }
    finally { setChatLoading(false) }
  }

  const onChatEnter = (e: KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') { e.preventDefault(); void sendChatMessage(chatInput) } }

  /* ── Derived ── */
  const healthStats = useMemo(() => {
    const total = columns.length
    const flagged = columns.filter((c) => ['warning', 'critical'].includes(c.health?.overall?.status || 'good')).length
    return { total, flagged }
  }, [columns])

  const topCategory = useMemo(() => {
    const cat = columns.find((c) => c.inferred_type === 'categorical')
    return cat?.name || null
  }, [columns])

  const typeOptions = ['numeric', 'categorical', 'boolean', 'datetime', 'free_text', 'id']

  /* ═══════════════════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════════════════ */
  return (
    <div className="min-h-screen bg-[#f7f8fa]">

      {/* ── Sticky Header ── */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <h1 className="text-lg font-bold tracking-tight text-gray-900">DataLens</h1>
          <div className="flex items-center gap-3">
            {summary && (
              <button
                onClick={() => setViewMode((p) => (p === 'simple' ? 'analyst' : 'simple'))}
                className="rounded-lg bg-gray-100 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-200"
              >
                {viewMode === 'simple' ? 'Simple' : 'Analyst'}
              </button>
            )}
            {summary && (
              <button
                onClick={() => { setSummary(null); setUploadMeta(null); setFile(null); setColumns([]); setColumnStats(null); setKeyFindings(null); setSelectedColumn(''); setChatMessages([]); setPreAnalysis(null); setGridPreview(null); setCompareResult(null); setCompareA(''); setCompareB('') }}
                className="rounded-lg bg-gray-100 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-200"
              >
                New Upload
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 pb-24 pt-8">
        <AnimatePresence mode="wait">

          {/* ════════════════════════════════════════════════════════════
             UPLOAD VIEW
             ════════════════════════════════════════════════════════════ */}
          {!summary && (
            <motion.div key="upload" {...fade} className="mx-auto max-w-2xl space-y-6 pt-16">
              <div className="text-center">
                <h2 className="text-3xl font-bold tracking-tight text-gray-900">Instant insights from your data</h2>
                <p className="mt-2 text-gray-500">Drop a CSV or Excel file — no setup needed</p>
              </div>

              <div
                {...getRootProps()}
                className={`cursor-pointer rounded-2xl p-14 text-center transition-all duration-200 ${
                  isDragActive
                    ? 'bg-blue-50 ring-2 ring-blue-300'
                    : 'bg-white ring-1 ring-gray-200 hover:ring-gray-300'
                }`}
              >
                <input {...getInputProps()} />
                <div className="text-4xl">📊</div>
                <p className="mt-4 font-medium text-gray-700">{isDragActive ? 'Drop it here!' : 'Drag & drop your file here'}</p>
                <p className="mt-1 text-sm text-gray-400">CSV, XLS, XLSX · up to 50 MB</p>
                {file && <p className="mt-4 text-sm font-semibold text-gray-900">Selected: {file.name}</p>}
              </div>

              <div className="flex items-center gap-3">
                <button onClick={uploadFile} disabled={!file || uploading}
                  className="rounded-xl bg-gray-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-gray-800 disabled:opacity-40">
                  {uploading ? `Uploading ${uploadProgress}%…` : 'Upload'}
                </button>
                {uploadMeta && !showAnalysisOptions && (
                  <button onClick={() => isLargeFile ? setShowAnalysisOptions(true) : void startAnalysis('auto')} disabled={!canAnalyze || runningAnalysis}
                    className="rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-40">
                    {runningAnalysis ? 'Analyzing…' : 'Start Analysis'}
                  </button>
                )}
                {!!uploadMeta?.sheet_names?.length && (
                  <select value={uploadMeta.active_sheet || ''} onChange={(e) => setUploadMeta((p) => (p ? { ...p, active_sheet: e.target.value } : p))}
                    className="rounded-xl bg-white px-3 py-2.5 text-sm ring-1 ring-gray-200">
                    {uploadMeta.sheet_names.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                )}
              </div>

              {/* ── Analysis Options Panel ── */}
              {uploadMeta && showAnalysisOptions && !runningAnalysis && (
                <motion.div {...fade} className="rounded-2xl bg-white p-6 ring-1 ring-gray-200 space-y-5">
                  <div>
                    <h3 className="text-base font-semibold text-gray-900">How would you like to analyze this data?</h3>
                    {rowEstimate && (
                      <p className="mt-1 text-sm text-gray-500">
                        Your file has ~{rowEstimate.toLocaleString()} rows. Choose a mode to balance speed and depth.
                      </p>
                    )}
                  </div>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    {([['quick', '⚡ Quick Summary', 'Overall trends, correlations & stats — no charts. Fastest.'],
                       ['focused', '🎯 Focus Columns', 'Pick specific columns for charts. Good for large files.'],
                       ['full', '🔬 Full Analysis', 'Every column gets charts & AI summaries. Slowest.']
                    ] as const).map(([mode, title, desc]) => (
                      <button key={mode} onClick={() => setAnalysisMode(mode)}
                        className={`rounded-xl p-4 text-left transition ring-1 ${
                          analysisMode === mode
                            ? 'bg-blue-50 ring-blue-300'
                            : 'bg-gray-50 ring-gray-200 hover:ring-gray-300'
                        }`}>
                        <p className="text-sm font-semibold text-gray-900">{title}</p>
                        <p className="mt-1 text-xs text-gray-500">{desc}</p>
                        {mode === 'quick' && isLargeFile && (
                          <span className="mt-2 inline-block rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">Recommended</span>
                        )}
                      </button>
                    ))}
                  </div>

                  {/* Column selection for focused mode */}
                  {analysisMode === 'focused' && detectedColumns.length > 0 && (
                    <div>
                      <p className="mb-2 text-sm font-medium text-gray-700">Select columns for detailed analysis:</p>
                      <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
                        {detectedColumns.map((col) => (
                          <button key={col}
                            onClick={() => setFocusColumns((prev) => prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col])}
                            className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                              focusColumns.includes(col)
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}>
                            {col}
                          </button>
                        ))}
                      </div>
                      {focusColumns.length > 0 && (
                        <p className="mt-2 text-xs text-gray-400">{focusColumns.length} selected — other columns get quick stats only</p>
                      )}
                    </div>
                  )}
                  {analysisMode === 'focused' && detectedColumns.length === 0 && (
                    <p className="text-sm text-gray-500">Column names will be detected during analysis. All columns will get quick stats, and you can drill into any column later.</p>
                  )}

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => void startAnalysis()}
                      disabled={!canAnalyze || (analysisMode === 'focused' && focusColumns.length === 0 && detectedColumns.length > 0)}
                      className="rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-40">
                      Start {analysisMode === 'quick' ? 'Quick' : analysisMode === 'focused' ? 'Focused' : 'Full'} Analysis
                    </button>
                    <button onClick={() => { setShowAnalysisOptions(false); setAnalysisMode('auto') }}
                      className="text-sm text-gray-500 hover:text-gray-700">
                      Cancel
                    </button>
                  </div>
                </motion.div>
              )}

              {uploading && (
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                  <motion.div className="h-full rounded-full bg-blue-600" initial={{ width: 0 }} animate={{ width: `${uploadProgress}%` }} />
                </div>
              )}
              {runningAnalysis && (
                <div className="flex items-center gap-3 rounded-2xl bg-blue-50 p-4 text-sm text-blue-800">
                  <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>
                  Analyzing your data — this may take a moment for large files…
                </div>
              )}
              {error && <p className="text-sm text-red-600">{error}</p>}
            </motion.div>
          )}

          {/* ════════════════════════════════════════════════════════════
             DASHBOARD VIEW — Bento Grid
             ════════════════════════════════════════════════════════════ */}
          {summary && (
            <motion.div key="dashboard" {...fade} className="space-y-6">

              {/* ── Row 1: Insight Banner (Hero) ── */}
              {keyFindings && (
                <section className="rounded-2xl bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 p-8">
                  <p className="text-sm font-semibold uppercase tracking-wider text-indigo-500">Key Insight</p>
                  <h2 className="mt-2 text-2xl font-bold leading-snug text-gray-900 md:text-3xl">
                    {keyFindings.whats_in_this_data}
                  </h2>
                  {keyFindings.suggested_next_step && (
                    <p className="mt-3 text-sm text-gray-500">
                      <span className="font-medium text-gray-700">Next step:</span> {keyFindings.suggested_next_step}
                    </p>
                  )}
                </section>
              )}

              {/* ── Row 2: KPI Cards ── */}
              <section className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                <KpiCard
                  label="Data Health"
                  value={gradeFromScore(summary.quality_score)}
                  sub={
                    <span className={`mt-1 inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold ${qualityLabel(summary.quality_score).color}`}>
                      {qualityLabel(summary.quality_score).text} ({summary.quality_score.toFixed(0)}%)
                    </span>
                  }
                  icon="🛡️"
                />
                <KpiCard
                  label="Dataset Size"
                  value={`${summary.rows.toLocaleString()} rows`}
                  sub={<span className="text-xs text-gray-400">{summary.columns} columns · {summary.memory_mb} MB</span>}
                  icon="📋"
                />
                <KpiCard
                  label="Top Category"
                  value={topCategory || 'N/A'}
                  sub={<span className="text-xs text-gray-400">{Object.entries(summary.type_breakdown).map(([k, v]) => `${v} ${k}`).join(', ')}</span>}
                  icon="📊"
                />
                <KpiCard
                  label="Anomalies"
                  value={healthStats.flagged > 0 ? `${healthStats.flagged} found` : 'None'}
                  sub={
                    healthStats.flagged > 0
                      ? <span className="text-xs text-amber-600">{healthStats.flagged} of {healthStats.total} columns need attention</span>
                      : <span className="text-xs text-emerald-600">✓ All columns look healthy</span>
                  }
                  icon="⚠️"
                />
              </section>

              {/* ── Row 3a: Trends (shown for quick/focused mode) ── */}
              {summary.trends && (
                <section className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {/* Correlations */}
                  {summary.trends.top_correlations && summary.trends.top_correlations.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">🔗 Top Correlations</h3>
                      <div className="space-y-2">
                        {summary.trends.top_correlations.map((c, i) => (
                          <div key={i} className="flex items-center justify-between text-sm">
                            <span className="text-gray-600 truncate">{c.col_a} ↔ {c.col_b}</span>
                            <span className={`font-mono text-xs font-semibold ${
                              Math.abs(c.r) > 0.7 ? 'text-red-600' : Math.abs(c.r) > 0.4 ? 'text-amber-600' : 'text-gray-500'
                            }`}>{c.r > 0 ? '+' : ''}{c.r}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Numeric summaries */}
                  {summary.trends.numeric_summaries && summary.trends.numeric_summaries.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">📈 Numeric Trends</h3>
                      <div className="space-y-2">
                        {summary.trends.numeric_summaries.slice(0, 6).map((n, i) => (
                          <div key={i} className="text-sm">
                            <span className="font-medium text-gray-800">{n.column}</span>
                            <span className="ml-2 text-xs text-gray-400">
                              mean {n.mean.toLocaleString()} · median {n.median.toLocaleString()} · skew {n.skew}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Category dominance */}
                  {summary.trends.category_dominance && summary.trends.category_dominance.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">📊 Category Leaders</h3>
                      <div className="space-y-2">
                        {summary.trends.category_dominance.slice(0, 6).map((d, i) => (
                          <div key={i} className="flex items-center justify-between text-sm">
                            <span className="text-gray-600 truncate">{d.column}</span>
                            <span className="text-xs">
                              <span className="font-medium text-gray-800">{d.top_value}</span>
                              <span className={`ml-1 ${d.top_pct > 60 ? 'text-amber-600' : 'text-gray-400'}`}>{d.top_pct}%</span>
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Time ranges */}
                  {summary.trends.time_ranges && summary.trends.time_ranges.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">📅 Time Ranges</h3>
                      <div className="space-y-2">
                        {summary.trends.time_ranges.map((t, i) => (
                          <div key={i} className="text-sm">
                            <span className="font-medium text-gray-800">{t.column}</span>
                            <p className="text-xs text-gray-400">{t.from} → {t.to} ({t.span_days.toLocaleString()} days)</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </section>
              )}

              {/* ── Row 3b: Key findings details ── */}
              {keyFindings && (keyFindings.top_findings?.length > 0 || keyFindings.watch_out_for?.length > 0) && (
                <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {keyFindings.top_findings?.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">🔍 Top Findings</h3>
                      <ul className="space-y-2">
                        {keyFindings.top_findings.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                            <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-blue-400" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {keyFindings.watch_out_for?.length > 0 && (
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h3 className="mb-3 text-sm font-semibold text-gray-900">⚠️ Watch Out For</h3>
                      <ul className="space-y-2">
                        {keyFindings.watch_out_for.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                            <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-amber-400" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </section>
              )}

              {/* ── Row 3c: Universal statistical findings ── */}
              {preAnalysis && (
                <section className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                    <h3 className="mb-3 text-sm font-semibold text-gray-900">🧠 Reclassifications & Quality Flags</h3>
                    <div className="space-y-2 text-sm text-gray-600">
                      {preAnalysis.name_quality_flag && (
                        <p className="rounded-lg bg-amber-50 px-3 py-2 text-amber-800">{preAnalysis.name_quality_flag}</p>
                      )}
                      {(preAnalysis.smart_type_correction?.reclassifications || []).slice(0, 6).map((r, i) => (
                        <p key={i}>⚡ <span className="font-medium text-gray-800">{r.column}</span>: {r.from} → {r.to} ({r.reason})</p>
                      ))}
                      {(preAnalysis.smart_type_correction?.high_missing_flags || []).slice(0, 4).map((f, i) => (
                        <p key={`m-${i}`} className="text-amber-700">🔴 {f.message}</p>
                      ))}
                      {(preAnalysis.dataset_level_checks?.sample_size_flags || []).map((f, i) => (
                        <p key={`s-${i}`} className="text-amber-700">⚠️ {f}</p>
                      ))}
                      {(preAnalysis.dataset_level_checks?.class_imbalance || []).slice(0, 4).map((item, i) => (
                        <p key={`c-${i}`} className="text-amber-700">📉 {item.message}</p>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                    <h3 className="mb-3 text-sm font-semibold text-gray-900">🔗 Relationships & Group Effects</h3>
                    <div className="space-y-2 text-sm text-gray-600">
                      {(preAnalysis.correlation_analysis?.pairs || []).slice(0, 5).map((pair, i) => (
                        <p key={i}>
                          <span className="font-medium text-gray-800">{pair.col_a}</span> ↔ <span className="font-medium text-gray-800">{pair.col_b}</span>
                          : r={pair.pearson_r}, ρ={pair.spearman_rho}
                          {pair.non_linear_signal ? <span className="ml-1 rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] text-indigo-700">non-linear</span> : null}
                        </p>
                      ))}
                      {(preAnalysis.correlation_analysis?.redundant_pairs || []).slice(0, 3).map((pair, i) => (
                        <p key={`r-${i}`} className="text-red-700">♻️ Redundant: {pair.col_a} and {pair.col_b} (r={pair.pearson_r})</p>
                      ))}
                      {(preAnalysis.group_difference_analysis?.strongest_by_category || []).slice(0, 4).map((g, i) => (
                        <p key={`g-${i}`}>📌 <span className="font-medium text-gray-800">{g.categorical_column}</span> best explains <span className="font-medium text-gray-800">{g.numeric_column}</span> (effect {g.effect_size})</p>
                      ))}
                      {(preAnalysis.outlier_characterisation?.multi_column_anomalies || []).slice(0, 4).map((row, i) => (
                        <p key={`o-${i}`} className="text-amber-700">🟠 Multi-column anomaly row {row.row_index}: {row.columns.join(', ')}</p>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* ── Row 4: Column pills (replaces sidebar) ── */}
              <section className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">Columns</h3>
                  <span className="text-xs text-gray-400">{columns.length} total</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {columns.map((col) => {
                    const isSelected = selectedColumn === col.name
                    const dot = col.health?.overall?.dot || '🟢'
                    const isFlag = ['warning', 'critical'].includes(col.health?.overall?.status || 'good')
                    return (
                      <button
                        key={col.name}
                        onClick={() => fetchColumnStats(col.name)}
                        className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition ${
                          isSelected
                            ? 'bg-gray-900 text-white'
                            : isFlag
                            ? 'bg-amber-50 text-amber-800 hover:bg-amber-100'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        <span className="text-[10px]">{dot}</span>
                        {col.name}
                        <span className={`rounded px-1 text-[10px] ${isSelected ? 'bg-gray-700 text-gray-300' : 'bg-white/60 text-gray-500'}`}>
                          {col.inferred_type}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </section>

              {/* ── Row 5: Deep Dive — 2/3 chart + 1/3 context ── */}
              {columnStats && (
                <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                  {/* Left: Charts (2/3) */}
                  <div className="space-y-4 lg:col-span-2">
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <div className="mb-4 flex items-center justify-between">
                        <div>
                          <h3 className="text-base font-semibold text-gray-900">Deep Dive: {columnStats.column}</h3>
                          <p className="mt-0.5 text-xs text-gray-400">Auto-selected visualization based on column type</p>
                        </div>
                        {viewMode === 'analyst' && (
                          <select
                            value={columnStats.inferred_type}
                            onChange={(e) => updateType(columnStats.column, e.target.value)}
                            className="rounded-lg bg-gray-50 px-2 py-1 text-xs ring-1 ring-gray-200"
                          >
                            {typeOptions.map((t) => <option key={t} value={t}>{t}</option>)}
                          </select>
                        )}
                      </div>

                      {/* Health badges */}
                      <div className="mb-4 flex flex-wrap gap-1.5">
                        {[columnStats.health?.missing, columnStats.health?.outliers, columnStats.health?.distribution, columnStats.health?.cardinality]
                          .filter((h) => !!h && h?.status !== 'na')
                          .map((h, i) => (
                            <span key={i} className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium ${
                              h?.status === 'critical' ? 'bg-red-50 text-red-700'
                              : h?.status === 'warning' ? 'bg-amber-50 text-amber-700'
                              : 'bg-emerald-50 text-emerald-700'
                            }`}>
                              {h?.dot} {h?.label}
                            </span>
                          ))}
                      </div>

                      {/* Charts */}
                      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                        {columnStats.chart_histogram_url && (
                          <img src={chartUrl(columnStats.chart_histogram_url)} alt="Distribution" className="w-full rounded-xl" />
                        )}
                        {columnStats.chart_boxplot_url && (
                          <img src={chartUrl(columnStats.chart_boxplot_url)} alt="Box plot" className="w-full rounded-xl" />
                        )}
                        {columnStats.chart_bar_url && (
                          <img src={chartUrl(columnStats.chart_bar_url)} alt="Categories" className="w-full rounded-xl md:col-span-2" />
                        )}
                        {columnStats.chart_line_url && (
                          <img src={chartUrl(columnStats.chart_line_url)} alt="Time trend" className="w-full rounded-xl md:col-span-2" />
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Right: Context panel (1/3) */}
                  <div className="space-y-4">
                    {/* AI Summary */}
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <h4 className="mb-3 text-sm font-semibold text-gray-900">💡 AI Summary</h4>
                      <div className="space-y-3 text-sm text-gray-600">
                        <div>
                          <p className="font-medium text-gray-800">What does this look like?</p>
                          <p className="mt-0.5">{columnStats.ai_summary?.what_does_this_look_like || 'Summary pending…'}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-800">Anything unusual?</p>
                          <p className="mt-0.5">{columnStats.ai_summary?.anything_unusual || 'No unusual patterns found.'}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-800">What should I do?</p>
                          <p className="mt-0.5">{columnStats.ai_summary?.what_should_i_do || 'No action needed right now.'}</p>
                        </div>
                      </div>
                    </div>

                    {/* Quick stats */}
                    <div className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-semibold text-gray-900">📈 At a Glance</h4>
                        {viewMode === 'analyst' && (
                          <button onClick={() => setShowRawStats((p) => !p)} className="text-[11px] text-blue-600 hover:underline">
                            {showRawStats ? 'Hide raw' : 'Show raw'}
                          </button>
                        )}
                      </div>
                      <div className="mt-3 space-y-2">
                        {columnStats.inferred_type === 'numeric' && (
                          <>
                            <StatRow label="Mean" value={typeof columnStats.mean === 'number' ? columnStats.mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '–'} />
                            <StatRow label="Median" value={typeof columnStats.median === 'number' ? columnStats.median.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '–'} />
                            <StatRow label="Std Dev" value={typeof columnStats.std === 'number' ? columnStats.std.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '–'} />
                            <StatRow label="Min" value={typeof columnStats.min === 'number' ? columnStats.min.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '–'} />
                            <StatRow label="Max" value={typeof columnStats.max === 'number' ? columnStats.max.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '–'} />
                            <StatRow label="Outliers (IQR)" value={String(columnStats.outliers_iqr_count ?? '–')} />
                          </>
                        )}
                        {columnStats.inferred_type === 'categorical' && (
                          <>
                            <StatRow label="Mode" value={String(columnStats.mode ?? '–')} />
                            <StatRow label="Unique" value={String(columnStats.cardinality ?? columnStats.unique_count ?? '–')} />
                          </>
                        )}
                        {columnStats.inferred_type === 'datetime' && (
                          <>
                            <StatRow label="From" value={String(columnStats.min_date ?? '–')} />
                            <StatRow label="To" value={String(columnStats.max_date ?? '–')} />
                            <StatRow label="Gaps" value={String(columnStats.gap_count ?? '–')} />
                          </>
                        )}
                        <StatRow label="Missing" value={`${columnStats.missing_pct ?? 0}%`} />
                      </div>

                      {viewMode === 'analyst' && showRawStats && (
                        <div className="mt-4 max-h-60 overflow-y-auto rounded-xl bg-gray-50 p-3 text-xs">
                          <table className="w-full">
                            <tbody>
                              {Object.entries(columnStats)
                                .filter(([k]) => !['top_10', 'frequency_table', 'observations_per_period', 'health', 'ai_summary', 'insight_summary'].includes(k))
                                .map(([k, v]) => (
                                  <tr key={k} className="border-b border-gray-100 last:border-0">
                                    <td className="py-1.5 pr-3 font-medium text-gray-500">{k}</td>
                                    <td className="py-1.5 text-gray-700">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>
                                  </tr>
                                ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  </div>
                </section>
              )}

              {/* ── Row 6: Data Wrangler Grid + Compare Mode ── */}
              {gridPreview && (
                <section className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
                  <div className="mb-4 flex flex-wrap items-center gap-2">
                    <h3 className="mr-auto text-sm font-semibold text-gray-900">🧩 Data Wrangler</h3>
                    <input
                      value={gridSearch}
                      onChange={(e) => setGridSearch(e.target.value)}
                      placeholder="Search columns"
                      className="rounded-lg bg-gray-50 px-3 py-1.5 text-xs ring-1 ring-gray-200"
                    />
                    <button onClick={() => setGridOutliersOnly((p) => !p)} className={`rounded-lg px-3 py-1.5 text-xs ${gridOutliersOnly ? 'bg-amber-100 text-amber-800' : 'bg-gray-100 text-gray-600'}`}>🟠 Outlier rows only</button>
                    <button onClick={() => setGridMissingOnly((p) => !p)} className={`rounded-lg px-3 py-1.5 text-xs ${gridMissingOnly ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>🔴 Missing rows only</button>
                    <button onClick={() => void refreshGridPreview()} className="rounded-lg bg-gray-100 px-3 py-1.5 text-xs text-gray-600">Refresh</button>
                  </div>

                  <div className="mb-4 flex flex-wrap items-center gap-2">
                    <select value={compareA} onChange={(e) => setCompareA(e.target.value)} className="rounded-lg bg-gray-50 px-2 py-1.5 text-xs ring-1 ring-gray-200">
                      <option value="">Compare column A</option>
                      {columns.map((c) => <option key={`a-${c.name}`} value={c.name}>{c.name}</option>)}
                    </select>
                    <select value={compareB} onChange={(e) => setCompareB(e.target.value)} className="rounded-lg bg-gray-50 px-2 py-1.5 text-xs ring-1 ring-gray-200">
                      <option value="">Compare column B</option>
                      {columns.map((c) => <option key={`b-${c.name}`} value={c.name}>{c.name}</option>)}
                    </select>
                    <button onClick={() => void runCompare()} className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white">Compare two columns</button>
                    <button onClick={downloadCleanedCsv} className="rounded-lg bg-gray-100 px-3 py-1.5 text-xs text-gray-600">Download filtered CSV</button>
                  </div>

                  {compareResult && (
                    <div className="mb-4 rounded-xl bg-gray-50 p-3 text-xs text-gray-700 ring-1 ring-gray-100">
                      <p className="font-semibold text-gray-900">Compare Mode</p>
                      <p className="mt-1">{String(compareResult.interpretation || compareResult.message || 'Comparison ready')}</p>
                      {'pearson_r' in compareResult && (
                        <p className="mt-1">r={String(compareResult.pearson_r)}{typeof compareResult.spearman_rho !== 'undefined' ? `, ρ=${String(compareResult.spearman_rho)}` : ''}</p>
                      )}
                      {'effect_size' in compareResult && <p className="mt-1">effect size={String(compareResult.effect_size)}</p>}
                    </div>
                  )}

                  <div className="max-h-80 overflow-auto rounded-xl ring-1 ring-gray-100">
                    <table className="min-w-full text-xs">
                      <thead className="sticky top-0 bg-gray-50">
                        <tr>
                          <th className="px-2 py-2 text-left text-gray-500">#</th>
                          {gridPreview.columns
                            .filter((col) => !gridSearch || col.name.toLowerCase().includes(gridSearch.toLowerCase()))
                            .map((col) => (
                              <th key={col.name} className={`px-2 py-2 text-left font-medium ${col.reclassified ? 'bg-purple-50 text-purple-700' : 'text-gray-600'}`}>
                                <div className="flex items-center gap-1">
                                  <span>{col.name}</span>
                                  <span className="rounded bg-gray-100 px-1 text-[10px] text-gray-500">{col.final_type}</span>
                                  {col.reclassified && <span className="rounded bg-purple-100 px-1 text-[10px] text-purple-700">⚡</span>}
                                </div>
                                <div className="mt-1 h-1.5 w-20 overflow-hidden rounded-full bg-gray-200">
                                  <div className="h-full bg-red-400" style={{ width: `${Math.min(100, Number(col.missing_pct || 0))}%` }} />
                                </div>
                              </th>
                            ))}
                        </tr>
                      </thead>
                      <tbody>
                        {gridPreview.rows.map((row) => (
                          <tr key={row.row_index} className={row.row_flags.includes('multi_column_anomaly') ? 'bg-[repeating-linear-gradient(45deg,#fff,#fff_8px,#f8fafc_8px,#f8fafc_16px)]' : ''}>
                            <td className="border-t border-gray-100 px-2 py-1.5 text-gray-400">{row.row_index}</td>
                            {gridPreview.columns
                              .filter((col) => !gridSearch || col.name.toLowerCase().includes(gridSearch.toLowerCase()))
                              .map((col) => {
                                const flags = row.cell_flags[col.name] || []
                                const value = row.values[col.name]
                                return (
                                  <td
                                    key={`${row.row_index}-${col.name}`}
                                    className={`border-t border-gray-100 px-2 py-1.5 ${
                                      flags.includes('missing') ? 'bg-red-50 text-red-700' : flags.includes('outlier') ? 'bg-amber-50 text-amber-800' : flags.includes('reclassified') ? 'bg-purple-50' : ''
                                    }`}
                                  >
                                    {flags.includes('missing') ? '—' : String(value ?? '')}
                                  </td>
                                )
                              })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* ════════════════════════════════════════════════════════════
         Floating Action Bar (bottom-right)
         ════════════════════════════════════════════════════════════ */}
      {summary && (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
          {/* Export / share popover */}
          <AnimatePresence>
            {fabOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="mb-1 flex flex-col gap-1.5 rounded-2xl bg-white p-2 shadow-xl ring-1 ring-gray-200"
              >
                <FabButton label="📄 PDF Report" onClick={createPdf} />
                <FabButton label="📊 Excel (Full)" onClick={() => downloadExcel()} />
                {summary && summary.rows > 10_000 && (
                  <FabButton label="📊 Excel (10K sample)" onClick={() => downloadExcel(10_000)} />
                )}
                <FabButton label="📁 Cleaned CSV" onClick={downloadCleanedCsv} />
                {viewMode === 'analyst' && <FabButton label="{ } Raw JSON" onClick={downloadStatsJson} />}
                <FabButton label="🔗 Copy Share Link" onClick={createShare} />
                {sharePath && <p className="px-3 text-[10px] text-emerald-600">Copied!</p>}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex gap-2">
            <button
              onClick={() => setChatOpen((p) => !p)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-gray-900 text-white shadow-lg transition hover:bg-gray-800"
              title="Ask about your data"
            >
              💬
            </button>
            <button
              onClick={() => setFabOpen((p) => !p)}
              className="flex h-11 w-11 items-center justify-center rounded-full bg-gray-900 text-white shadow-lg transition hover:bg-gray-800"
              title="Export & share"
            >
              ↗
            </button>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════
         Chat Drawer (slides up from bottom-right)
         ════════════════════════════════════════════════════════════ */}
      <AnimatePresence>
        {chatOpen && summary && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            className="fixed bottom-20 right-6 z-50 flex w-96 flex-col rounded-2xl bg-white shadow-2xl ring-1 ring-gray-200"
            style={{ maxHeight: '28rem' }}
          >
            <div className="flex items-center justify-between border-b px-4 py-3">
              <h4 className="text-sm font-semibold text-gray-900">💬 Ask About Your Data</h4>
              <button onClick={() => setChatOpen(false)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>

            {/* Starter chips */}
            {chatMessages.length === 0 && (
              <div className="flex flex-wrap gap-1.5 border-b px-4 py-3">
                {starterPrompts.map((p) => (
                  <button key={p} onClick={() => void sendChatMessage(p)}
                    className="rounded-full bg-gray-100 px-2.5 py-1 text-[11px] text-gray-600 transition hover:bg-gray-200">{p}</button>
                ))}
              </div>
            )}

            <div className="flex-1 overflow-y-auto px-4 py-3 text-sm" style={{ maxHeight: '16rem' }}>
              {chatMessages.length === 0 && <p className="text-gray-400">Ask a question about this dataset.</p>}
              {chatMessages.map((m, i) => (
                <div key={i} className={`mb-2 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
                  <span className={`inline-block max-w-[85%] rounded-2xl px-3 py-2 ${
                    m.role === 'user' ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-800'
                  }`}>{m.content}</span>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div className="flex gap-2 border-t px-3 py-2.5">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={onChatEnter}
                placeholder="Type your question…"
                className="flex-1 rounded-xl bg-gray-50 px-3 py-2 text-sm outline-none ring-1 ring-gray-200 focus:ring-blue-300"
              />
              <button
                onClick={() => void sendChatMessage(chatInput)}
                disabled={!chatInput.trim() || chatLoading}
                className="rounded-xl bg-gray-900 px-3 py-2 text-sm text-white disabled:opacity-40"
              >
                {chatLoading ? '…' : '→'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer */}
      <footer className="py-6 text-center text-xs text-gray-400">
        DataLens — Instant Automated Insights
      </footer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════
   Sub-components
   ═══════════════════════════════════════════════════════════════════════ */

function KpiCard({ label, value, sub, icon }: { label: string; value: string; sub: React.ReactNode; icon: string }) {
  return (
    <motion.div {...fade} className="rounded-2xl bg-white p-6 ring-1 ring-gray-100">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-gray-400">{label}</p>
          <p className="mt-1 text-xl font-bold text-gray-900">{value}</p>
          <div className="mt-1">{sub}</div>
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </motion.div>
  )
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-gray-50 py-1.5 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-xs font-medium text-gray-900">{value}</span>
    </div>
  )
}

function FabButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full rounded-xl px-4 py-2 text-left text-xs font-medium text-gray-700 transition hover:bg-gray-50"
    >
      {label}
    </button>
  )
}

export default App
