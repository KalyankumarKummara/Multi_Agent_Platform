import assert from 'node:assert/strict'
import test from 'node:test'

import { DEMO_REPOSITORY, operationRequestUrl } from '../src/githubRequests.js'

const demoOwner = 'KalyankumarKummara'
const demoRepo = 'github-agent-test'

test('demo explorer target is the configured monitored repository', () => {
  assert.deepEqual(DEMO_REPOSITORY, { owner: demoOwner, repo: demoRepo })
})

for (const operationId of ['branches', 'commits', 'pull-requests', 'issues']) {
  test(`${operationId} always requests the demo repository, even if the monitored input changes`, () => {
    const url = new URL(operationRequestUrl(operationId, { owner: 'someone-else', repo: 'other-repo' }), 'http://localhost')

    assert.equal(url.searchParams.get('owner'), demoOwner)
    assert.equal(url.searchParams.get('repo'), demoRepo)
    assert.equal(url.searchParams.get('per_page'), '20')
    assert.equal(url.searchParams.get('page'), '1')
  })
}

test('repositories listing continues to use the monitored account owner', () => {
  const url = new URL(operationRequestUrl('repositories', { owner: demoOwner, repo: demoRepo }), 'http://localhost')

  assert.equal(url.pathname, '/github/repositories')
  assert.equal(url.searchParams.get('username'), demoOwner)
  assert.equal(url.searchParams.has('owner'), false)
  assert.equal(url.searchParams.has('repo'), false)
})
