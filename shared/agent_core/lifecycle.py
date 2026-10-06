from .state import AgentStatus


class AgentLifecycle:

    def __init__(self):
        self.status = AgentStatus.CREATED

    def initialize(self):
        self._require_status(AgentStatus.CREATED)

        self.status = AgentStatus.INITIALIZING

    def mark_ready(self):
        self._require_status(AgentStatus.INITIALIZING)

        self.status = AgentStatus.READY

    def start(self):
        self._require_status(
            AgentStatus.READY,
            AgentStatus.WAITING,
        )

        self.status = AgentStatus.RUNNING

    def wait(self):
        self._require_status(AgentStatus.RUNNING)

        self.status = AgentStatus.WAITING

    def error(self):
        self._require_status(
            AgentStatus.INITIALIZING,
            AgentStatus.READY,
            AgentStatus.RUNNING,
            AgentStatus.WAITING,
            AgentStatus.RECOVERING,
        )

        self.status = AgentStatus.ERROR

    def recover(self):
        self._require_status(AgentStatus.ERROR)

        self.status = AgentStatus.RECOVERING

    def stop(self):
        self._require_status(
            AgentStatus.READY,
            AgentStatus.RUNNING,
            AgentStatus.WAITING,
            AgentStatus.ERROR,
            AgentStatus.RECOVERING,
        )

        self.status = AgentStatus.STOPPING

    def stopped(self):
        self._require_status(AgentStatus.STOPPING)

        self.status = AgentStatus.STOPPED

    def _require_status(self, *allowed_statuses: AgentStatus):
        if self.status not in allowed_statuses:

            allowed = ", ".join(
                status.value
                for status in allowed_statuses
            )

            raise RuntimeError(
                f"Invalid lifecycle transition from "
                f"{self.status.value}. "
                f"Allowed states: {allowed}"
            )