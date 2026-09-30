from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    principal_id: str
    principal_type: str
    permissions: frozenset[str]


class AuthorizationService:

    def is_allowed(
        self,
        principal: Principal,
        required_permission: str
    ) -> bool:

        return required_permission in principal.permissions