import assert from 'node:assert/strict'
import test from 'node:test'

import { eventTime, formatTime, latestEvent } from '../src/activityTime.js'

test('timestamp display distinguishes occurrence, processing, and unavailable times', () => {
  assert.match(eventTime({ occurred_at: '2026-01-01T00:00:00Z', processed_at: '2026-01-02T00:00:00Z' }), /^Occurred:/)
  assert.match(eventTime({ occurred_at: null, processed_at: '2026-01-02T00:00:00Z' }), /^Processed:/)
  assert.equal(eventTime({ occurred_at: null, processed_at: null }), 'Time unavailable')
})

test('row timestamp selection formats occurred_at first and falls back to processed_at', () => {
  const occurred = '2026-01-01T00:00:00Z'
  const processed = '2026-01-02T00:00:00Z'
  assert.equal(formatTime(occurred), new Date(occurred).toLocaleString())
  assert.equal(formatTime(processed), new Date(processed).toLocaleString())
  assert.equal(formatTime(null), null)
})

test('latest event uses occurrence or fallback processing time without sorting the source array', () => {
  const events = [
    { activity_id: 'processed-latest', occurred_at: null, processed_at: '2026-01-03T00:00:00Z' },
    { activity_id: 'occurred-latest', occurred_at: '2026-01-04T00:00:00Z', processed_at: '2026-01-05T00:00:00Z' },
    { activity_id: 'old', occurred_at: '2026-01-01T00:00:00Z', processed_at: '2026-01-06T00:00:00Z' },
  ]
  const originalOrder = events.map((event) => event.activity_id)

  assert.equal(latestEvent(events).activity_id, 'occurred-latest')
  assert.deepEqual(events.map((event) => event.activity_id), originalOrder)
})
