import axios from 'axios'
import { useCallback, useEffect, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { useDropzone } from 'react-dropzone'

type UploadMeta = {
  upload_id: string
  file_name: string
  analysis_status: string
  sheet_names?: string[]
  active_sheet?: string | null
  warning?: string | null
}

type DatasetSummary = {
  rows: number
  columns: number
  memory_mb: number
  quality_score: number
  type_breakdown: Record<string, number>
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

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
})

function chartUrl(path?: string) {
  if (!path) return ''
  return path.startsWith('http') ? path : `${API_BASE}${path}`
}

function badgeClass(status: string) {
  if (status === 'critical') return 'bg-red-50 text-red-700 border-red-200'
  if (status === 'warning') return 'bg-amber-50 text-amber-700 border-amber-200'
  return 'bg-emerald-50 text-emerald-700 border-emerald-200'
}

function App() {
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
  const [attentionOnly, setAttentionOnly] = useState(false)
  const [viewMode, setViewMode] = useState<'simple' | 'analyst'>(() => {
    const stored = localStorage.getItem('datalens:view-mode')
    return stored === 'analyst' ? 'analyst' : 'simple'
  })
  const [showRawStats, setShowRawStats] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatOpen, setChatOpen] = useState(viewMode === 'simple')
  const [chatLoading, setChatLoading] = useState(false)

  const uploadId = uploadMeta?.upload_id
  const canAnalyze = !!uploadId && uploadMeta?.analysis_status !== 'running' && uploadMeta?.analysis_status !== 'queued'

  useEffect(() => {
    localStorage.setItem('datalens:view-mode', viewMode)
    setChatOpen(viewMode === 'simple')
  }, [viewMode])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError('')
    const selected = acceptedFiles[0]
    if (!selected) return
    const valid = selected.name.endsWith('.csv') || selected.name.endsWith('.xlsx') || selected.name.endsWith('.xls')
    if (!valid) {
      setError('Only .csv, .xlsx, .xls are supported')
      return
    }
    if (selected.size > 50 * 1024 * 1024) {
      setError('Max file size is 50MB')
      return
    }
    setFile(selected)
    setUploadMeta(null)
    setSummary(null)
    setKeyFindings(null)
    setColumns([])
    setSelectedColumn('')
    setColumnStats(null)
    setSharePath('')
    setChatMessages([])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
  })

  const uploadFile = async () => {
    if (!file) return
    setUploading(true)
    setError('')
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await api.post('/api/v1/uploads', formData, {
        onUploadProgress: (evt) => {
          if (!evt.total) return
          setUploadProgress(Math.round((evt.loaded * 100) / evt.total))
        },
      })
      setUploadMeta(response.data.metadata)
    } catch {
      setError('Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const fetchColumnStats = useCallback(async (name: string, id?: string) => {
    const targetId = id || uploadId
    if (!targetId) return
    setSelectedColumn(name)
    const response = await api.get(`/api/v1/analysis/${targetId}/columns/${encodeURIComponent(name)}/stats`)
    setColumnStats(response.data)
  }, [uploadId])

  const loadDashboard = useCallback(async (id: string) => {
    const [summaryRes, columnsRes, findingsRes] = await Promise.all([
      api.get(`/api/v1/analysis/${id}/summary`),
      api.get(`/api/v1/analysis/${id}/columns`),
      api.get(`/api/v1/analysis/${id}/key-findings`),
    ])

    const loadedColumns: ColumnItem[] = columnsRes.data
    setSummary(summaryRes.data)
    setColumns(loadedColumns)
    setKeyFindings(findingsRes.data)

    if (loadedColumns.length > 0) {
      await fetchColumnStats(loadedColumns[0].name, id)
    }
  }, [fetchColumnStats])

  const startAnalysis = async () => {
    if (!uploadId) return
    setRunningAnalysis(true)
    setError('')
    try {
      await api.post(`/api/v1/analysis/${uploadId}/start`, {
        active_sheet: uploadMeta?.active_sheet || undefined,
      })
      setUploadMeta((prev) => (prev ? { ...prev, analysis_status: 'queued' } : prev))
    } catch {
      setError('Failed to start analysis')
      setRunningAnalysis(false)
    }
  }

  useEffect(() => {
    if (!uploadId) return
    if (!['queued', 'running'].includes(uploadMeta?.analysis_status || '')) return

    const timer = window.setInterval(async () => {
      const statusRes = await api.get(`/api/v1/analysis/${uploadId}/status`)
      const newStatus = statusRes.data.status
      setUploadMeta((prev) => (prev ? { ...prev, analysis_status: newStatus } : prev))

      if (newStatus === 'completed') {
        window.clearInterval(timer)
        setRunningAnalysis(false)
        await loadDashboard(uploadId)
      }
      if (newStatus === 'failed') {
        window.clearInterval(timer)
        setRunningAnalysis(false)
        setError(statusRes.data.error || 'Analysis failed')
      }
    }, 2000)

    return () => window.clearInterval(timer)
  }, [uploadId, uploadMeta?.analysis_status, loadDashboard])

  const updateType = async (name: string, newType: string) => {
    if (!uploadId) return
    await api.patch(`/api/v1/analysis/${uploadId}/columns/${encodeURIComponent(name)}/type`, {
      new_type: newType,
    })

    setColumns((prev) => prev.map((item) => (item.name === name ? { ...item, inferred_type: newType } : item)))
    await fetchColumnStats(name)
  }

  const createPdf = async () => {
    if (!uploadId) return
    const response = await api.post(`/api/v1/analysis/${uploadId}/export/pdf`)
    window.open(chartUrl(response.data.pdf_url), '_blank')
  }

  const createShare = async () => {
    if (!uploadId) return
    const response = await api.post(`/api/v1/analysis/${uploadId}/share`)
    const path = `${API_BASE}${response.data.share_path}`
    setSharePath(path)
    await navigator.clipboard.writeText(path)
  }

  const downloadCleanedCsv = () => {
    if (!uploadId) return
    window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/cleaned-csv`, '_blank')
  }

  const downloadExcel = () => {
    if (!uploadId) return
    window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/excel`, '_blank')
  }

  const downloadStatsJson = () => {
    if (!uploadId) return
    window.open(`${API_BASE}/api/v1/analysis/${uploadId}/export/stats-json`, '_blank')
  }

  const starterPrompts = [
    'What stands out in this data?',
    'Are there any data quality issues?',
    'Summarize this for a non-technical audience',
    'Which columns need attention?',
    'What would you investigate first?',
  ]

  const sendChatMessage = async (message: string) => {
    if (!uploadId || !message.trim() || chatLoading) return
    const userMessage: ChatMessage = { role: 'user', content: message.trim() }
    const baseHistory = [...chatMessages, userMessage].slice(-10)
    setChatMessages(baseHistory)
    setChatInput('')
    setChatLoading(true)

    try {
      const response = await fetch(`${API_BASE}/api/v1/analysis/${uploadId}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim(), history: baseHistory }),
      })

      if (!response.body) {
        throw new Error('No stream available')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantText = ''
      setChatMessages((prev) => [...prev, { role: 'assistant' as const, content: '' }].slice(-10))

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter((line) => line.startsWith('data: '))
        for (const line of lines) {
          const payload = JSON.parse(line.replace('data: ', '')) as { token?: string; done?: boolean }
          if (payload.token) {
            assistantText += payload.token
            setChatMessages((prev) => {
              const copy = [...prev]
              copy[copy.length - 1] = { role: 'assistant', content: assistantText }
              return copy.slice(-10)
            })
          }
        }
      }
    } catch {
      setChatMessages((prev) => [...prev, { role: 'assistant' as const, content: 'I could not answer right now. Please try again.' }].slice(-10))
    } finally {
      setChatLoading(false)
    }
  }

  const onChatEnter = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      void sendChatMessage(chatInput)
    }
  }

  const typeOptions = ['numeric', 'categorical', 'boolean', 'datetime', 'free_text', 'id']

  const healthStats = useMemo(() => {
    const total = columns.length
    const flagged = columns.filter((column) => ['warning', 'critical'].includes(column.health?.overall?.status || 'good')).length
    return { total, flagged }
  }, [columns])

  const visibleColumns = useMemo(() => {
    if (!attentionOnly) return columns
    return columns.filter((column) => ['warning', 'critical'].includes(column.health?.overall?.status || 'good'))
  }, [columns, attentionOnly])

  const hideRaw = viewMode === 'simple'

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">DataLens</h1>
          <button
            className="rounded border bg-white px-4 py-2 text-sm"
            onClick={() => setViewMode((prev) => (prev === 'simple' ? 'analyst' : 'simple'))}
          >
            {viewMode === 'simple' ? '👤 Simple View' : '🔬 Analyst View'}
          </button>
        </div>

        {!summary && (
          <section className="rounded-xl border bg-white p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Upload Page</h2>
            <div
              {...getRootProps()}
              className={`cursor-pointer rounded-lg border-2 border-dashed p-10 text-center ${
                isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300'
              }`}
            >
              <input {...getInputProps()} />
              <p className="text-sm text-slate-600">Drag & drop CSV/XLS/XLSX (max 50MB), or click to select</p>
              {file && <p className="mt-3 text-sm font-medium">Selected: {file.name}</p>}
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <button
                onClick={uploadFile}
                disabled={!file || uploading}
                className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
              >
                {uploading ? 'Uploading...' : 'Upload'}
              </button>

              <button
                onClick={startAnalysis}
                disabled={!canAnalyze || runningAnalysis}
                className="rounded bg-emerald-600 px-4 py-2 text-white disabled:opacity-50"
              >
                {runningAnalysis ? 'Running Analysis...' : 'Start Analysis'}
              </button>

              {!!uploadMeta?.sheet_names?.length && (
                <select
                  value={uploadMeta.active_sheet || ''}
                  onChange={(e) => setUploadMeta((prev) => (prev ? { ...prev, active_sheet: e.target.value } : prev))}
                  className="rounded border px-3 py-2"
                >
                  {uploadMeta.sheet_names.map((sheet) => (
                    <option key={sheet} value={sheet}>
                      {sheet}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {uploading && (
              <div className="mt-4 h-2 w-full rounded bg-slate-200">
                <div className="h-2 rounded bg-blue-600" style={{ width: `${uploadProgress}%` }} />
              </div>
            )}

            {uploadMeta && (
              <div className="mt-4 rounded bg-slate-100 p-3 text-sm">
                <p>Upload ID: {uploadMeta.upload_id}</p>
                <p>Status: {uploadMeta.analysis_status}</p>
                {uploadMeta.warning && <p className="text-amber-700">{uploadMeta.warning}</p>}
              </div>
            )}

            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          </section>
        )}

        {summary && (
          <section className="grid grid-cols-1 gap-4">
            {keyFindings && (
              <div className="rounded-xl border bg-white p-5 shadow-sm">
                <h2 className="mb-4 text-lg font-semibold">📋 Key Findings</h2>
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div>
                    <p className="mb-1 text-sm font-semibold">What's in this data?</p>
                    <p className="text-sm text-slate-700">{keyFindings.whats_in_this_data}</p>
                  </div>
                  <div>
                    <p className="mb-1 text-sm font-semibold">➡️ Suggested Next Step</p>
                    <p className="text-sm text-slate-700">{keyFindings.suggested_next_step}</p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div>
                    <p className="mb-1 text-sm font-semibold">🔍 Top Findings</p>
                    <ul className="list-disc pl-5 text-sm text-slate-700">
                      {keyFindings.top_findings?.map((item, idx) => (
                        <li key={`${item}-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  {!!keyFindings.watch_out_for?.length && (
                    <div>
                      <p className="mb-1 text-sm font-semibold">⚠️ Watch Out For</p>
                      <ul className="list-disc pl-5 text-sm text-slate-700">
                        {keyFindings.watch_out_for.map((item, idx) => (
                          <li key={`${item}-${idx}`}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <h2 className="mb-2 text-lg font-semibold">Dataset Summary Card</h2>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                <div><span className="text-xs text-slate-500">Rows</span><p className="font-semibold">{summary.rows}</p></div>
                <div><span className="text-xs text-slate-500">Columns</span><p className="font-semibold">{summary.columns}</p></div>
                <div><span className="text-xs text-slate-500">Memory</span><p className="font-semibold">{summary.memory_mb} MB</p></div>
                <div><span className="text-xs text-slate-500">Quality Score</span><p className="font-semibold">{summary.quality_score}</p></div>
                <div><span className="text-xs text-slate-500">Types</span><p className="font-semibold">{Object.entries(summary.type_breakdown).map(([k, v]) => `${k}:${v}`).join(' | ')}</p></div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
              <aside className="rounded-xl border bg-white p-3 shadow-sm">
                <h3 className="mb-2 font-semibold">Columns</h3>
                <button
                  onClick={() => setAttentionOnly((prev) => !prev)}
                  className="mb-3 w-full rounded border bg-slate-50 px-3 py-2 text-left text-sm"
                >
                  {healthStats.flagged} of {healthStats.total} columns need attention
                </button>
                {attentionOnly && <p className="mb-2 text-xs text-amber-700">Showing only 🟡/🔴 columns</p>}

                <div className="space-y-2">
                  {visibleColumns.map((column) => (
                    <button
                      key={column.name}
                      onClick={() => fetchColumnStats(column.name)}
                      className={`w-full rounded border px-3 py-2 text-left ${selectedColumn === column.name ? 'border-blue-500 bg-blue-50' : 'border-slate-200'}`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate text-sm font-medium">
                          {(column.health?.overall?.dot || '🟢')} {column.name}
                        </span>
                        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs">{column.inferred_type}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </aside>

              <main className="rounded-xl border bg-white p-4 shadow-sm">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                  <h3 className="text-lg font-semibold">Column Detail Panel</h3>
                  {columnStats && (
                    <select
                      value={columnStats.inferred_type}
                      onChange={(e) => updateType(columnStats.column, e.target.value)}
                      className="rounded border px-3 py-2 text-sm"
                    >
                      {typeOptions.map((typeOption) => (
                        <option key={typeOption} value={typeOption}>{typeOption}</option>
                      ))}
                    </select>
                  )}
                </div>

                {columnStats && (
                  <>
                    <div className="mb-4 flex flex-wrap gap-2">
                      {[
                        columnStats.health?.missing,
                        columnStats.health?.outliers,
                        columnStats.health?.distribution,
                        columnStats.health?.cardinality,
                      ]
                        .filter((item) => !!item && item?.status !== 'na')
                        .map((item, index) => (
                          <span key={`${item?.label}-${index}`} className={`rounded border px-2 py-1 text-xs ${badgeClass(item?.status || 'good')}`}>
                            {item?.dot} {item?.label}
                          </span>
                        ))}
                    </div>

                    {!hideRaw && (
                      <div className="mb-3">
                        <button onClick={() => setShowRawStats((prev) => !prev)} className="rounded border px-3 py-1 text-sm">
                          {showRawStats ? 'Hide raw stats' : 'Show raw stats'}
                        </button>
                      </div>
                    )}

                    {!hideRaw && showRawStats && (
                      <div className="mb-4 overflow-x-auto rounded border">
                        <table className="min-w-full text-sm">
                          <tbody>
                            {Object.entries(columnStats)
                              .filter(([key]) => !['top_10', 'frequency_table', 'observations_per_period', 'health', 'ai_summary'].includes(key))
                              .map(([key, value]) => (
                                <tr key={key} className="border-b">
                                  <td className="bg-slate-50 px-3 py-2 font-medium">{key}</td>
                                  <td className="px-3 py-2">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                      {columnStats.chart_histogram_url && <img src={chartUrl(columnStats.chart_histogram_url)} alt="Histogram" className="w-full rounded border bg-white" />}
                      {columnStats.chart_boxplot_url && <img src={chartUrl(columnStats.chart_boxplot_url)} alt="Boxplot" className="w-full rounded border bg-white" />}
                      {columnStats.chart_bar_url && <img src={chartUrl(columnStats.chart_bar_url)} alt="Bar chart" className="w-full rounded border bg-white lg:col-span-2" />}
                      {columnStats.chart_line_url && <img src={chartUrl(columnStats.chart_line_url)} alt="Time trend" className="w-full rounded border bg-white lg:col-span-2" />}
                    </div>

                    <div className="mt-4 rounded bg-blue-50 p-3 text-sm text-slate-800 space-y-3">
                      <div>
                        <p className="font-medium">What does this look like?</p>
                        <p>{columnStats.ai_summary?.what_does_this_look_like || 'Summary pending'}</p>
                      </div>
                      <div>
                        <p className="font-medium">Anything unusual?</p>
                        <p>{columnStats.ai_summary?.anything_unusual || 'No unusual pattern highlighted yet.'}</p>
                      </div>
                      <div>
                        <p className="font-medium">What should I do?</p>
                        <p>{columnStats.ai_summary?.what_should_i_do || 'No action recommendation available yet.'}</p>
                      </div>
                    </div>
                  </>
                )}
              </main>
            </div>

            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <h3 className="mb-3 font-semibold">Export Controls</h3>
              <div className="flex flex-wrap gap-3">
                <button onClick={createPdf} className="rounded bg-slate-900 px-4 py-2 text-white">Download PDF Report</button>
                <button onClick={downloadExcel} className="rounded bg-slate-700 px-4 py-2 text-white">Download Excel</button>
                <button onClick={downloadCleanedCsv} className="rounded bg-slate-700 px-4 py-2 text-white">Download Cleaned CSV</button>
                {viewMode === 'analyst' && (
                  <button onClick={downloadStatsJson} className="rounded bg-slate-700 px-4 py-2 text-white">Download Raw JSON Stats</button>
                )}
                <button onClick={createShare} className="rounded bg-indigo-600 px-4 py-2 text-white">Copy Share Link</button>
              </div>
              {sharePath && <p className="mt-3 text-sm text-emerald-700">Copied: {sharePath}</p>}
            </div>

            <div className="rounded-xl border bg-white p-4 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="font-semibold">💬 Ask About Your Data</h3>
                <button className="rounded border px-3 py-1 text-sm" onClick={() => setChatOpen((prev) => !prev)}>
                  {chatOpen ? 'Collapse' : 'Open'}
                </button>
              </div>

              {chatOpen && (
                <>
                  <div className="mb-3 flex flex-wrap gap-2">
                    {starterPrompts.map((prompt) => (
                      <button
                        key={prompt}
                        onClick={() => void sendChatMessage(prompt)}
                        className="rounded-full border bg-slate-50 px-3 py-1 text-xs"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>

                  <div className="mb-3 h-56 overflow-y-auto rounded border bg-slate-50 p-3 text-sm">
                    {chatMessages.length === 0 && <p className="text-slate-500">Ask a question about this dataset.</p>}
                    {chatMessages.map((message, index) => (
                      <div key={`${message.role}-${index}`} className={`mb-2 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                        <span className={`inline-block rounded px-3 py-2 ${message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border'}`}>
                          {message.content}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <input
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      onKeyDown={onChatEnter}
                      placeholder="Type your question..."
                      className="flex-1 rounded border px-3 py-2"
                    />
                    <button
                      onClick={() => void sendChatMessage(chatInput)}
                      disabled={!chatInput.trim() || chatLoading}
                      className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
                    >
                      Send
                    </button>
                  </div>
                </>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

export default App
