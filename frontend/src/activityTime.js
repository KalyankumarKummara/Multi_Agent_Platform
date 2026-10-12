export function formatTime(value) {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export function eventTime(event) {
  if (event.occurred_at) return `Occurred: ${formatTime(event.occurred_at)}`
  if (event.processed_at) return `Processed: ${formatTime(event.processed_at)}`
  return 'Time unavailable'
}

export function bestTimestamp(event) {
  const value = event.occurred_at || event.processed_at
  const timestamp = value ? new Date(value).getTime() : Number.NaN
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp
}

export function latestEvent(events) {
  return events.reduce((latest, event) => (
    !latest || bestTimestamp(event) > bestTimestamp(latest) ? event : latest
  ), null)
}
