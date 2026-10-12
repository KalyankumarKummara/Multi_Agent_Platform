export const DEMO_REPOSITORY = Object.freeze({
  owner: 'KalyankumarKummara',
  repo: 'github-agent-test',
})

export const GITHUB_OPERATIONS = [
  { id: 'repositories', label: 'Repositories', route: '/github/repositories' },
  { id: 'branches', label: 'Branches', route: '/github/branches' },
  { id: 'commits', label: 'Commits', route: '/github/commits' },
  { id: 'pull-requests', label: 'Pull Requests', route: '/github/pull-requests' },
  { id: 'issues', label: 'Issues', route: '/github/issues' },
]

export function operationRequestUrl(operationId, monitoredRepository) {
  const operation = GITHUB_OPERATIONS.find(({ id }) => id === operationId)
  if (!operation) throw new Error(`Unknown GitHub operation: ${operationId}`)

  const query = new URLSearchParams({ per_page: '20', page: '1' })
  if (operationId === 'repositories') {
    if (!monitoredRepository?.owner) throw new Error('A repository owner is required to list repositories.')
    query.set('username', monitoredRepository.owner)
  } else {
    query.set('owner', DEMO_REPOSITORY.owner)
    query.set('repo', DEMO_REPOSITORY.repo)
  }

  return `${operation.route}?${query}`
}
