import { useCallback, useEffect, useMemo, useState } from 'react'
import { DEMO_REPOSITORY, GITHUB_OPERATIONS, operationRequestUrl } from './githubRequests'
import { eventTime, formatTime, latestEvent as selectLatestEvent } from './activityTime'

const requiredRoutes = [
  '/github/activities',
  '/github/repositories',
  '/github/branches',
  '/github/commits',
  '/github/pull-requests',
  '/github/issues',
]

function Icon({ name }) {
  const common = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }

  if (name === 'repositories') return <svg viewBox="0 0 20 20" aria-hidden="true" {...common}><rect x="3" y="3" width="14" height="5" rx="1" /><rect x="3" y="12" width="14" height="5" rx="1" /><path d="M6 5.5h.01M6 14.5h.01" /></svg>
  if (name === 'branches') return <svg viewBox="0 0 20 20" aria-hidden="true" {...common}><circle cx="6" cy="4" r="2" /><circle cx="6" cy="16" r="2" /><circle cx="14" cy="7" r="2" /><path d="M6 6v8M8 14c4 0 6-2 6-5" /></svg>
  if (name === 'commits') return <svg viewBox="0 0 20 20" aria-hidden="true" {...common}><path d="M2 10h5m6 0h5" /><circle cx="10" cy="10" r="3" /></svg>
  if (name === 'pull-requests') return <svg viewBox="0 0 20 20" aria-hidden="true" {...common}><circle cx="5" cy="4" r="2" /><circle cx="5" cy="16" r="2" /><circle cx="15" cy="16" r="2" /><path d="M5 6v8m2-10h4a4 4 0 0 1 4 4v2m-3-2 3 3 3-3" /></svg>
  return <svg viewBox="0 0 20 20" aria-hidden="true" {...common}><circle cx="10" cy="10" r="7" /><path d="M8 8a2.1 2.1 0 1 1 3.6 1.5c-1 .9-1.6 1.2-1.6 2.5M10 14.5h.01" /></svg>
}

function actorName(actor) {
  if (!actor) return 'Unknown actor'
  return actor.login || actor.name || actor.username || 'Unknown actor'
}

function itemHeading(item) {
  return item.full_name || item.name || item.title || item.commit?.message?.split('\n')[0] || item.message?.split('\n')[0] || item.sha || item.node_id || 'Details unavailable'
}

function statusClass(status) {
  if (status === 'processed') return 'status-chip status-processed'
  if (status === 'failed') return 'status-chip status-failed'
  return 'status-chip'
}

function activityIconName(eventType = '') {
  const normalized = String(eventType || '').toLowerCase()
  if (normalized.includes('pull_request')) return 'pull-requests'
  if (normalized.includes('issue')) return 'issues'
  if (normalized.includes('push') || normalized.includes('commit')) return 'commits'
  if (normalized.includes('branch') || normalized.includes('create') || normalized.includes('delete')) return 'branches'
  return 'repositories'
}

export default function App() {
  const repository = `${DEMO_REPOSITORY.owner}/${DEMO_REPOSITORY.repo}`
  const [backendStatus, setBackendStatus] = useState({ state: 'checking', text: 'Checking backend' })
  const [selectedOperation, setSelectedOperation] = useState(null)
  const [results, setResults] = useState([])
  const [resultError, setResultError] = useState('')
  const [loadingResults, setLoadingResults] = useState(false)
  const [events, setEvents] = useState([])
  const [eventError, setEventError] = useState('')
  const [loadingEvents, setLoadingEvents] = useState(false)

  const latestEvent = useMemo(() => {
    return selectLatestEvent(events)
  }, [events])

  const requestJson = useCallback(async (path) => {
    const response = await fetch(path)
    const body = await response.json().catch(() => ({}))
    if (!response.ok) {
      const reason = typeof body.detail === 'string' ? body.detail : `Request failed (${response.status})`
      throw new Error(`${response.status}: ${reason}`)
    }
    return body
  }, [])

  const checkBackend = useCallback(async () => {
    setBackendStatus({ state: 'checking', text: 'Checking backend' })
    try {
      const response = await fetch('/openapi.json')
      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`)
      const specification = await response.json()
      const paths = specification.paths || {}
      const missing = requiredRoutes.filter((route) => !paths[route]?.get)
      if (missing.length) {
        setBackendStatus({ state: 'incomplete', text: `Missing ${missing.length} API route${missing.length === 1 ? '' : 's'}` })
        return
      }
      setBackendStatus({ state: 'online', text: 'Backend connected' })
    } catch (error) {
      setBackendStatus({ state: 'offline', text: 'Backend unavailable' })
    }
  }, [])

  const refreshActivity = useCallback(async () => {
    setLoadingEvents(true)
    setEventError('')
    setEvents([])
    try {
      const query = new URLSearchParams({ limit: '50' })
      query.set('repository', repository)
      const data = await requestJson(`/github/activities?${query}`)
      setEvents(Array.isArray(data) ? data : [])
    } catch (error) {
      setEventError(error.message || 'Activity history could not be loaded.')
    } finally {
      setLoadingEvents(false)
    }
  }, [repository, requestJson])

  const showGitHubData = useCallback(async (operation) => {
    setSelectedOperation(operation)
    setLoadingResults(true)
    setResultError('')
    setResults([])
    try {
      const data = await requestJson(operationRequestUrl(operation.id, DEMO_REPOSITORY))
      setResults(Array.isArray(data) ? data : [data])
    } catch (error) {
      setResultError(error.message || 'GitHub request failed.')
    } finally {
      setLoadingResults(false)
    }
  }, [requestJson])

  useEffect(() => {
    checkBackend()
    refreshActivity()
  }, [])

  return (
    <main className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">GH</span>
          <span className="brand-name">GitHub Agent Demo</span>
        </div>
        <div className="header-status">
          <span className={`connection-dot connection-${backendStatus.state}`} />
          <span className="connection-label">{backendStatus.text}</span>
          <span className="read-only">READ ONLY</span>
        </div>
      </header>

      <div className="dashboard">
        <section className="panel activity-panel">
          <div className="panel-heading">
            <div>
              <h1>Monitor Activity</h1>
            </div>
            <button className="refresh-button" type="button" onClick={() => refreshActivity()} disabled={loadingEvents}>
              <span aria-hidden="true">↻</span>{loadingEvents ? 'Refreshing' : 'Refresh'}
            </button>
          </div>

          <div className="summary-row" aria-live="polite">
            <div className="summary-item">
              <span className="summary-label">TOTAL EVENTS</span>
              <strong>{loadingEvents ? '—' : eventError ? '—' : events.length}</strong>
            </div>
            <div className="summary-item summary-repository">
              <span className="summary-label">REPOSITORY</span>
              <strong>{repository}</strong>
            </div>
            <div className="summary-item summary-latest">
              <span className="summary-label">LATEST EVENT</span>
              <strong>{latestEvent ? eventTime(latestEvent) : 'Time unavailable'}</strong>
            </div>
          </div>

          {loadingEvents && <p className="state-message">Loading persisted activity…</p>}
          {eventError && <p className="state-message error" role="alert">{eventError}</p>}
          {!loadingEvents && !eventError && events.length === 0 && <p className="state-message">No persisted events for this repository.</p>}
          {!loadingEvents && !eventError && events.length > 0 && (
            <div className="table-wrap">
              <table>
                <colgroup>
                  <col className="column-event" />
                  <col className="column-repository" />
                  <col className="column-actor" />
                  <col className="column-occurred" />
                  <col className="column-status" />
                </colgroup>
                <thead>
                  <tr>
                    <th>EVENT TYPE</th>
                    <th>REPOSITORY</th>
                    <th>ACTOR</th>
                    <th>EVENT TIME</th>
                    <th>STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event.activity_id}>
                      <td data-label="Event Type">
                        <span className="event-name">
                          <span className={`event-icon event-icon-${activityIconName(event.event_type)}`}><Icon name={activityIconName(event.event_type)} /></span>
                          <span>{event.event_type || 'Unknown'}{event.action ? ` \u00b7 ${event.action}` : ''}</span>
                        </span>
                      </td>
                      <td className="repository-cell" data-label="Repository">{event.repository?.full_name || 'Unknown repository'}</td>
                      <td className="actor-cell" data-label="Actor">{actorName(event.actor)}</td>
                      <td className="event-time" data-label="Event Time">{formatTime(event.occurred_at || event.processed_at) || 'Time unavailable'}</td>
                      <td data-label="Status"><span className={statusClass(event.status)}>{event.status || 'Unknown'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel explorer-panel">
          <div className="panel-heading explorer-heading">
            <div>
              <h2>GitHub Explorer</h2>
              <p>Read-only repository data</p>
            </div>
          </div>

          <div className="operation-list">
            {GITHUB_OPERATIONS.map((operation) => (
              <button
                className={`operation-button operation-${operation.id}`}
                key={operation.id}
                type="button"
                aria-pressed={selectedOperation?.id === operation.id}
                disabled={loadingResults}
                onClick={() => showGitHubData(operation)}
              >
                <span className="operation-icon"><Icon name={operation.id} /></span>
                <span>{operation.label}</span>
                <span className="operation-arrow" aria-hidden="true">›</span>
              </button>
            ))}
          </div>

          <p className="explorer-target">Repository data: <strong>{DEMO_REPOSITORY.owner}/{DEMO_REPOSITORY.repo}</strong></p>
          <div className="results" aria-live="polite">
            <div className="results-heading">
              <h3>{selectedOperation?.label || 'Results'}</h3>
              {selectedOperation && <span>LIVE GITHUB DATA</span>}
            </div>
            {loadingResults && <p className="state-message">Loading live GitHub data…</p>}
            {resultError && <p className="state-message error" role="alert">{resultError}</p>}
            {!loadingResults && !resultError && results.length === 0 && <p className="state-message">Select an operation to load results.</p>}
            {!loadingResults && !resultError && results.length > 0 && (
              <ul className="result-list">
                {results.map((item, index) => (
                  <li key={item.id || item.sha || item.number || item.name || `${selectedOperation?.id}-${index}`}>
                    <strong>{itemHeading(item)}</strong>
                    {item.html_url && <a href={item.html_url} target="_blank" rel="noreferrer">Open on GitHub</a>}
                    {selectedOperation?.id !== 'issues' && item.state && <span className="result-state">{item.state}</span>}
                    {item.protected !== undefined && <span className="result-state">{item.protected ? 'Protected' : 'Unprotected'}</span>}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}
