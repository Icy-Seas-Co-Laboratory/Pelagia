from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import bearer_token, require_auth
    from ._common import as_response, get_repository

    class LoginRequest(BaseModel):
        username: str
        password: str
        project_id: str | None = None
        project_key: str | None = None
        ttl_seconds: int | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class SwitchProjectRequest(BaseModel):
        project_id: str | None = None
        project_key: str | None = None
        ttl_seconds: int | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    router = APIRouter(prefix="/auth", tags=["auth"])
    projects_router = APIRouter(prefix="/projects", tags=["projects"])
    routers = [projects_router]

    def _resolve_login_project(repository, user: dict[str, Any], body: LoginRequest | SwitchProjectRequest) -> dict[str, Any]:
        if body.project_id:
            project = repository.get_project(body.project_id)
            if project is None:
                raise HTTPException(status_code=404, detail=f"Project {body.project_id!r} was not found.")
            return project
        if body.project_key:
            project = repository.get_project_by_key(body.project_key)
            if project is None:
                raise HTTPException(status_code=404, detail=f"Project {body.project_key!r} was not found.")
            return project
        projects = repository.list_user_projects(str(user["id"]))
        if projects:
            return projects[0]
        if user.get("is_admin"):
            project = repository.get_project_by_key("default")
            if project is not None:
                return project
        raise HTTPException(status_code=403, detail="User does not belong to any active project.")

    def _session_response(repository, token: str) -> dict[str, Any]:
        session = repository.get_session(token, touch=False)
        if session is None:
            raise HTTPException(status_code=500, detail="Created session could not be resolved.")
        return {
            "token": token,
            "session": session,
            "user": {
                "id": session["user_id"],
                "username": session["username"],
                "display_name": session.get("display_name"),
                "is_admin": session.get("is_admin"),
            },
            "project": {
                "id": session["project_id"],
                "project_key": session["project_key"],
                "project_name": session["project_name"],
                "role": session.get("project_role"),
            },
        }

    def _default_ttl_seconds(request: Request) -> int:
        auth_config = getattr(getattr(request.app.state, "config", None), "auth", None)
        return max(1, int(getattr(auth_config, "session_ttl_seconds", 7 * 24 * 60 * 60)))

    def _session_ttl_seconds(request: Request, requested_ttl_seconds: int | None) -> int:
        default_ttl = _default_ttl_seconds(request)
        if requested_ttl_seconds is None:
            return default_ttl
        return max(1, min(int(requested_ttl_seconds), default_ttl))

    @router.post("/login")
    def login(request: Request, body: LoginRequest) -> dict:
        repository = get_repository(request)
        user = repository.verify_user_password(body.username, body.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        project = _resolve_login_project(repository, user, body)
        try:
            result = repository.create_session(
                str(user["id"]),
                str(project["id"]),
                ttl_seconds=_session_ttl_seconds(request, body.ttl_seconds),
                user_agent=request.headers.get("user-agent"),
                remote_addr=None if request.client is None else request.client.host,
                metadata=body.metadata,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response(_session_response(repository, result["token"]))

    @router.get("/me")
    def me(request: Request) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        user = repository.get_user(auth.user_id)
        project = repository.get_project(auth.project_id)
        return as_response(
            {
                "auth": auth.as_dict(),
                "user": user,
                "project": project,
                "projects": repository.list_user_projects(auth.user_id),
            }
        )

    @router.post("/logout")
    def logout(request: Request) -> dict:
        token = bearer_token(request)
        if not token:
            raise HTTPException(status_code=401, detail="Pelagia session token is required.")
        revoked = get_repository(request).revoke_session(token)
        return {"revoked": revoked is not None}

    @router.post("/switch-project")
    def switch_project(request: Request, body: SwitchProjectRequest) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        user = repository.get_user(auth.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Session user was not found.")
        project = _resolve_login_project(repository, user, body)
        try:
            result = repository.create_session(
                auth.user_id,
                str(project["id"]),
                ttl_seconds=_session_ttl_seconds(request, body.ttl_seconds),
                user_agent=request.headers.get("user-agent"),
                remote_addr=None if request.client is None else request.client.host,
                metadata=body.metadata,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return as_response(_session_response(repository, result["token"]))

    @projects_router.get("")
    def list_projects(request: Request, include_all_names: bool = False) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        projects = repository.list_projects(active_only=True) if auth.is_admin else repository.list_user_projects(auth.user_id)
        response = {"projects": as_response(projects)}
        if include_all_names:
            all_projects = repository.list_projects(active_only=True)
            response["all_project_names"] = [
                str(project.get("project_name") or project.get("project_key"))
                for project in all_projects
            ]
        return response
else:
    router = None
    projects_router = None
    routers = []
