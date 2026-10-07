# Multi-Agent Platform — Project Status

Last Updated: 2026-10-07

## 1. Project Overview

We are building a real-world AI Multi-Agent SaaS Platform for company operations.

The platform will contain multiple specialized AI agents, with a Project Manager Agent serving as the primary human-facing agent.

### Main goals

- Microsoft Teams employee communication
- Employee questions and requests
- GitHub / Azure DevOps workflows
- Email management
- Company knowledge search
- Notifications and daily reports
- Approval workflows
- Human escalation for important decisions

---

# 2. Technology Stack

## Backend

- Python
- FastAPI

## Frontend

- React

## LLM

- Groq or another approved LLM provider

## Database

- PostgreSQL or MSSQL

## Agent Framework

- Python tool calling
- LangGraph if required

## Integrations

- Microsoft Teams
- GitHub / Azure DevOps
- Email

---

# 3. High-Level Architecture

The system has two major execution planes.

## Human Interaction Plane

Real PM
    ↓
PM Agent
    ↓
Agent Coordinator
    ↓
Specialist Agent
    ↓
Result
    ↓
PM Agent
    ↓
Real PM

## Autonomous / Background Plane

External System
    ↓
Webhook / Change Notification / Polling
    ↓
Event Gateway
    ↓
Queue
    ↓
Specialist Agent
    ↓
Action / Activity
    ↓
Activity Store

Background monitoring must not depend on the PM Agent being online.

---

# 4. Agent Architecture

## Project Manager Agent

The PM Agent is the primary human-facing agent.

Responsibilities:

- Receive requests from the real PM
- Answer project-level questions
- Use the Agent Coordinator when specialist capabilities are required
- Provide project summaries
- Eventually support voice interaction
- Present important approvals and escalations to the real PM

Voice is an interaction channel, not the core agent architecture.

## Specialist Agents

Specialist agents own specific domains.

Examples:

- GitHub Agent
- Teams Agent
- Outlook Agent

Specialist agents:

- Monitor their external systems
- Process events
- Perform authorized actions
- Maintain relevant activity
- Request other domain capabilities through the Agent Coordinator
- Do not directly communicate peer-to-peer
- Are not direct human-facing chatbots

---

# 5. Agent Coordinator

The Agent Coordinator is infrastructure between agents.

Responsibilities:

- Agent registry
- Capability discovery
- Routing
- Task lifecycle
- Retries
- Timeouts
- Authorization hooks
- Tracing
- Audit support

Specialist agents should not directly call each other.

Example:

Teams Agent
    ↓
Agent Coordinator
    ↓
GitHub Agent
    ↓
Agent Coordinator
    ↓
Teams Agent

---

# 6. Agent Registry

The Agent Registry maintains information about available agents.

Expected information:

- Agent ID
- Agent name
- Capabilities
- Tools
- Permissions
- Endpoint
- Health/status
- Version
- Metadata

The registry answers:

> Which agent can perform this capability?

The Coordinator determines:

> How should the task be routed and managed?

---

# 7. Tool Gateway

Each specialist agent will use a Tool Gateway between the agent and the external platform.

Principle:

> An agent never directly accesses an external platform. It accesses approved capabilities through a controlled Tool Gateway.

Tool Gateway responsibilities:

- Input validation
- Authorization
- Credential handling
- Data minimization
- Sensitive-data filtering/redaction
- Rate-limit handling
- External API error handling
- Audit logging
- Policy enforcement hooks

Generic arbitrary API tools must not be exposed.

Example:

PM Agent
    ↓
Agent Coordinator
    ↓
GitHub Agent
    ↓
GitHub Tool Gateway
    ↓
GitHub API

---

# 8. Human Approval

Sensitive or high-impact actions require human approval.

General flow:

Request
    ↓
Intent / Context
    ↓
Proposed Action
    ↓
Policy
    ├── Safe + Authorized → Execute
    └── Sensitive → Human Approval
                           ↓
                    Approved / Rejected
                           ↓
                       Execute

The LLM proposes an action.

The policy system determines whether approval is required.

The executor performs the action.

---

# 9. Common Agent Foundation

Current status:

The Common Agent Foundation is now functionally complete as the current development foundation.

The foundation has been implemented and integration-tested.

Production hardening remains for future phases.

## Components

| Component | Status |
|---|---|
| Agent Identity | Complete |
| Agent Lifecycle | Complete |
| Agent State | Complete |
| Agent Capabilities | Complete |
| Tool Definition | Complete |
| Tool Registry | Complete |
| Tool Executor | Complete foundation |
| Tool Execution Context | Complete |
| Agent Context | Complete |
| Memory | Foundation complete |
| Policy Engine | Foundation complete |
| Human Approval | Complete foundation |
| Security / Authorization | Complete foundation |
| Structured Errors | Complete |
| Centralized Error Handling | Complete |
| Input Validation | Complete |
| Audit | Complete foundation + integration tested |
| Tracing | Complete foundation + integration tested |
| Configuration | Foundation complete |
| BaseAgent | Complete foundation |
| Foundation Tests | Complete — 14/14 passing |

## Production Hardening

The current foundation is a development foundation, not yet a production-hardened distributed platform.

Future production work includes:

- Persistent audit storage
- Distributed tracing and observability
- Production-grade policy enforcement
- Production authentication and authorization
- Persistent memory
- Advanced retry and recovery
- Security testing
- Load and performance testing

# 10. Common Foundation Files

Current shared structure:

shared/
├── __init__.py
└── agent_core/
    ├── __init__.py
    ├── identity.py
    ├── state.py
    ├── lifecycle.py
    ├── capabilities.py
    ├── context.py
    ├── memory.py
    ├── policy.py
    ├── approval.py
    ├── audit.py
    ├── tracing.py
    ├── errors.py
    ├── error_handler.py
    ├── config.py
    ├── security.py
    ├── validation.py
    ├── base_agent.py
    └── tools/
        ├── __init__.py
        ├── tool.py
        ├── registry.py
        ├── execution.py
        └── executor.py
---

# 11. Verified Security Workflow

The following workflow has been implemented and tested:

Tool Request
    ↓
Input Validation
    ↓
Authorization
    ↓
Policy
    ↓
Approval Required
    ↓
Pending Approval
    ↓
Human Approval
    ↓
Approved
    ↓
Authorization AGAIN
    ↓
Policy AGAIN
    ↓
Tool Execution
    ↓
Audit

## Verified behavior

- Authorized tool execution works.
- Unauthorized execution is blocked.
- Invalid tool input is rejected.
- Missing required fields are rejected.
- Unexpected fields are rejected.
- Invalid field types are rejected.
- Approval-required actions do not execute immediately.
- Approval requests receive unique IDs.
- Multiple approval requests receive different IDs.
- Pending approval can become approved.
- Approved actions execute through the approved execution path.
- Authorization is checked again after approval.
- Policy is checked again after approval.
- Unauthorized approved actions are blocked.
- Policy-denied actions are blocked.
- Structured `AgentError` handling works.
- Tool execution failures are handled.
- Audit events are recorded.
- Trace IDs are associated with audit events.
- High-risk merge operation is protected by approval.

---


# 12. GitHub Agent

## Current status

Milestones 1 through 6 are implemented on the GitHub Agent development branch.

The agent foundation, read-only GitHub tools, signed webhook ingestion, FastAPI
webhook boundary, and activity history are in place. Activity history can use
the in-memory development store or the MSSQL adapter described below.

### M6 Event Processing & Significance

- `GitHubEventProcessor` deterministically classifies normalized events and
  assigns low, medium, or high significance using explicit event/action rules.
- Processing is informational and read-only. No LLM, GitHub write action, or
  automatic follow-up is performed; `requires_action` remains false.
- `GitHubAgent` returns the processing summary with the saved activity ID.
- These rules are initial V1 behavior, not production monitoring or a complete
  interpretation of every GitHub event action.

The GitHub Agent will be the first specialist agent.

### V1 goals

- Connect a test GitHub account/repository
- Receive GitHub events
- Verify incoming webhooks
- Normalize events
- Process events
- Record meaningful activities
- Handle duplicate events
- Handle failures and retries
- Maintain activity history
- Demonstrate automatic activity detection

### M5.1 MSSQL activity persistence

- Dedicated database: `MultiAgentPlatform` (SQL Server 2025 Developer Edition).
- `ManagerAI` is a separate database and is not used by the GitHub activity store.
- `MSSQLActivityStore` implements the existing `ActivityStore` abstraction with
  SQLAlchemy 2.x and pyodbc; `InMemoryActivityStore` remains the default for tests.
- The `github_activities` table stores normalized activity fields and serialized
  minimized dictionaries, never the raw webhook body. The delivery/event ID is
  unique, with indexes for repository, event type, occurrence time, and status.
- Local Windows development uses Windows Authentication, ODBC Driver 18, and
  `TrustServerCertificate=yes`. Set
  `GITHUB_ACTIVITY_DB_TRUST_SERVER_CERTIFICATE=no` when certificate validation
  is available; do not use the development trust setting as a production default.
- Configuration variables are documented in `.env.example`. A full SQLAlchemy
  URL may be supplied through `GITHUB_ACTIVITY_DATABASE_URL`; keep it in a
  secure environment setting, not a committed file.
- The schema initializer creates only the `github_activities` table if it is
  missing. It never drops tables or creates databases.
- The local integration test is opt-in:
  `RUN_MSSQL_INTEGRATION_TESTS=1 python -m tests.test_mssql_activity_store_integration`
- SQLAlchemy 2.x and pyodbc are available in the local virtual environment.
  This repository currently has no dependency manifest; declare these packages
  in one when a project-wide install workflow is established.

Current persistence remains synchronous and uses in-memory webhook idempotency.
Cross-process delivery coordination, database migrations, and deployment secrets
management remain future work.

### Initial event categories

- Repository events
- Push / commit activity
- Branch activity
- Pull requests
- Pull request reviews
- Pull request comments
- Issues
- Issue comments

---

# 13. GitHub Agent Architecture

Test GitHub Account
    ↓
GitHub Webhook
    ↓
Event Gateway
    ↓
Verify / Normalize
    ↓
Queue
    ↓
GitHub Agent
    ├── Read event
    ├── Classify
    ├── Determine significance
    ├── Apply policy
    ├── Perform permitted action
    └── Record activity
    ↓
Activity Store

---

# 14. GitHub Authentication

## Development

Use a dedicated test account/repository where possible.

Use minimal permissions.

Never hardcode credentials.

## Production

Preferred approach:

- GitHub App
- Least-privilege permissions
- Installation access tokens
- Secure secret storage

---

# 15. GitHub Webhook Security

GitHub webhook processing must include:

- `X-Hub-Signature-256` validation
- HMAC-SHA256 verification
- Constant-time signature comparison
- Delivery ID tracking
- Duplicate detection
- Reject invalid requests before processing
- Fast webhook response
- Queue-based processing

---

# 16. Repository / Branch Strategy

Repository:

Multi_Agent_Platform

Desired structure:

Multi_Agent_Platform/
├── README.md
├── PROJECT_STATUS.md
├── shared/
│   └── agent_core/
├── GitHub_agent/
├── teams_agent/
├── outlook_agent/
└── ...

Common foundation is maintained as shared code.

General workflow:

common-agent-foundation
        ↓
      PR
        ↓
      main
      /   \
     /     \
github   teams / outlook
