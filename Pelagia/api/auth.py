from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from .routes._common import get_context, get_repository


READ_ROLES = {"viewer", "editor", "manager", "admin"}
WRITE_ROLES = {"editor", "manager", "admin"}
MANAGE_ROLES = {"manager", "admin"}


@dataclass(frozen=True, slots=True)
class AuthContext:
    user_id: str
    username: str
    project_id: str | None
    project_key: str | None
    role: str | None
    is_admin: bool = False
    session_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "project_id": self.project_id,
            "project_key": self.project_key,
            "role": self.role,
            "is_admin": self.is_admin,
            "session_id": self.session_id,
        }


def bearer_token(request: Request) -> str | None:
    value = request.headers.get("authorization") or request.headers.get("Authorization")
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _auth_enabled(request: Request) -> bool:
    config = getattr(request.app.state, "config", None)
    auth_config = getattr(config, "auth", None)
    return bool(getattr(auth_config, "enabled", True))


def _dev_auth_context(request: Request) -> AuthContext:
    context = get_context(request)
    auth_config = getattr(context.config, "auth", None)
    project_key = getattr(auth_config, "dev_project_key", None)
    if not project_key:
        raise HTTPException(
            status_code=503,
            detail="Auth-disabled mode requires auth.dev_project_key to name an existing project.",
        )
    project = get_repository(request).get_project_by_key(project_key)
    if project is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dev auth project {project_key!r} is not available. Create it or enable auth.",
        )
    return AuthContext(
        user_id="dev",
        username="dev",
        project_id=str(project["id"]),
        project_key=str(project["project_key"]),
        role="admin",
        is_admin=True,
        session_id=None,
    )


def get_optional_auth_context(request: Request) -> AuthContext | None:
    cached = getattr(request.state, "auth_context", None)
    if cached is not None:
        return cached
    token = bearer_token(request)
    if not token:
        if not _auth_enabled(request):
            auth = _dev_auth_context(request)
            request.state.auth_context = auth
            return auth
        request.state.auth_context = None
        return None
    session = get_repository(request).get_session(token)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired Pelagia session token.")
    project_id = None if session.get("project_id") is None else str(session["project_id"])
    role = (
        None
        if project_id is None
        else session.get("project_role") or ("admin" if session.get("is_admin") else "viewer")
    )
    auth = AuthContext(
        user_id=str(session["user_id"]),
        username=str(session["username"]),
        project_id=project_id,
        project_key=None if session.get("project_key") is None else str(session["project_key"]),
        role=None if role is None else str(role),
        is_admin=bool(session.get("is_admin")),
        session_id=str(session["id"]),
    )
    request.state.auth_context = auth
    return auth


def require_auth(request: Request) -> AuthContext:
    auth = get_optional_auth_context(request)
    if auth is None:
        raise HTTPException(status_code=401, detail="Pelagia session token is required.")
    return auth


def require_project_read(request: Request) -> AuthContext:
    auth = require_auth(request)
    if auth.project_id is None:
        raise HTTPException(status_code=403, detail="Select or create a project before accessing project resources.")
    if auth.is_admin or auth.role in READ_ROLES:
        return auth
    raise HTTPException(status_code=403, detail="Project read permission is required.")


def require_project_write(request: Request) -> AuthContext:
    auth = require_auth(request)
    if auth.project_id is None:
        raise HTTPException(status_code=403, detail="Select or create a project before accessing project resources.")
    if auth.is_admin or auth.role in WRITE_ROLES:
        return auth
    raise HTTPException(status_code=403, detail="Project write permission is required.")


def require_project_manager(request: Request) -> AuthContext:
    auth = require_auth(request)
    if auth.project_id is None:
        raise HTTPException(status_code=403, detail="Select or create a project before accessing project resources.")
    if auth.is_admin or auth.role in MANAGE_ROLES:
        return auth
    raise HTTPException(status_code=403, detail="Project manager permission is required.")


def require_admin(request: Request) -> AuthContext:
    auth = require_auth(request)
    if auth.is_admin:
        return auth
    raise HTTPException(status_code=403, detail="Admin permission is required.")


def scoped_project_id(request: Request) -> str:
    auth = require_project_read(request)
    assert auth.project_id is not None
    return auth.project_id
