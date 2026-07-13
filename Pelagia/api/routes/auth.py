from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import MANAGE_ROLES, bearer_token, require_auth
    from ...services.projects import initialize_project_kvstore
    from ...services.project_settings import (
        merge_project_settings,
        invalidate_project_settings,
        resolve_project_settings,
        resolve_project_storage_settings,
        storage_settings_payload,
    )
    from ...storage.postgres import DEFAULT_PROJECT_ID, PROJECT_ROLES
    from ._common import as_response, get_context, get_repository

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

    class CreateProjectRequest(BaseModel):
        project_key: str
        project_name: str | None = None
        description: str | None = None
        kvstore_root_path: str | None = None
        is_active: bool = True
        metadata: dict[str, Any] = Field(default_factory=dict)

    class UpdateProjectRequest(BaseModel):
        project_name: str | None = None
        description: str | None = None
        kvstore_root_path: str | None = None
        is_active: bool | None = None
        metadata: dict[str, Any] | None = None

    class UpdateProjectStorageSettingsRequest(BaseModel):
        frame_encoding: str | None = None
        frame_quality: int | None = None
        roi_encoding: str | None = None

    class CreateUserRequest(BaseModel):
        username: str
        password: str | None = Field(default=None, min_length=1)
        display_name: str | None = None
        is_admin: bool = False
        is_active: bool = True
        project_id: str | None = None
        project_key: str | None = None
        role: str = "viewer"
        metadata: dict[str, Any] = Field(default_factory=dict)

    class ResetPasswordRequest(BaseModel):
        password: str = Field(min_length=1)

    router = APIRouter(prefix="/auth", tags=["auth"])
    projects_router = APIRouter(prefix="/projects", tags=["projects"])
    users_router = APIRouter(prefix="/users", tags=["users"])
    routers = [projects_router, users_router]

    def _resolve_login_project(repository, user: dict[str, Any], body: LoginRequest | SwitchProjectRequest) -> dict[str, Any]:
        if body.project_id:
            if not _is_uuid(body.project_id):
                raise HTTPException(status_code=422, detail="project_id must be a UUID.")
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

    def _is_uuid(value: str | None) -> bool:
        if not value:
            return False
        try:
            UUID(str(value))
        except ValueError:
            return False
        return True

    def _project_by_id_or_key(repository, project_id: str) -> dict[str, Any] | None:
        if _is_uuid(project_id):
            project = repository.get_project(project_id)
            if project is not None:
                return project
        return repository.get_project_by_key(project_id)

    def _user_by_id_or_username(repository, user_id: str) -> dict[str, Any] | None:
        if _is_uuid(user_id):
            user = repository.get_user(user_id)
            if user is not None:
                return user
        return repository.get_user_by_username(user_id)

    def _public_user(user: dict[str, Any] | None) -> dict[str, Any] | None:
        if user is None:
            return None
        hidden = {"password", "password_hash"}
        return {key: value for key, value in user.items() if key not in hidden}

    def _target_membership_project(auth, repository, project_id: str | None, project_key: str | None) -> dict[str, Any]:
        if project_id:
            if not _is_uuid(project_id):
                raise HTTPException(status_code=422, detail="project_id must be a UUID.")
            project = repository.get_project(project_id)
        elif project_key:
            project = repository.get_project_by_key(project_key)
        else:
            project = repository.get_project(auth.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project was not found.")
        if not project.get("is_active", True):
            raise HTTPException(status_code=422, detail="Project is not active.")
        if not auth.is_admin and str(project["id"]) != auth.project_id:
            raise HTTPException(status_code=403, detail="Project manager permission is limited to the active project.")
        return project

    def _require_project_management(auth, repository, project_id: str) -> None:
        if auth.is_admin:
            return
        membership = repository.get_project_membership(auth.user_id, project_id)
        if membership is not None and str(membership.get("role")) in MANAGE_ROLES:
            return
        raise HTTPException(status_code=403, detail="Project manager permission is required.")

    def _require_user_management(auth, repository, user: dict[str, Any]) -> None:
        if auth.is_admin:
            return
        if auth.role not in MANAGE_ROLES:
            raise HTTPException(status_code=403, detail="Project manager permission is required.")
        if user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Only user admins can manage user admin accounts.")
        membership = repository.get_project_membership(str(user["id"]), auth.project_id)
        if membership is None:
            raise HTTPException(status_code=404, detail="User was not found in the active project.")

    def _normalize_requested_role(role: str) -> str:
        normalized = str(role).strip().lower()
        if normalized not in PROJECT_ROLES:
            raise HTTPException(status_code=422, detail=f"Project role must be one of: {', '.join(sorted(PROJECT_ROLES))}.")
        return normalized

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

    @projects_router.post("")
    def create_project(request: Request, body: CreateProjectRequest) -> dict:
        auth = require_auth(request)
        if not auth.is_admin:
            raise HTTPException(status_code=403, detail="User admin permission is required to create projects.")
        repository = get_repository(request)
        if repository.get_project_by_key(body.project_key) is not None:
            raise HTTPException(status_code=409, detail=f"Project {body.project_key!r} already exists.")
        try:
            project = repository.create_project(
                body.project_key,
                project_name=body.project_name,
                description=body.description,
                kvstore_root_path=body.kvstore_root_path,
                is_active=body.is_active,
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        user = repository.get_user(auth.user_id)
        membership = None
        if user is not None:
            membership = repository.add_project_member(auth.user_id, str(project["id"]), role="admin")
        kvstore = initialize_project_kvstore(get_context(request), project)
        return as_response({"project": project, "membership": membership, "kvstore": kvstore})

    @projects_router.patch("/{project_id}")
    def update_project(request: Request, project_id: str, body: UpdateProjectRequest) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        project = _project_by_id_or_key(repository, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        _require_project_management(auth, repository, str(project["id"]))
        updated = repository.update_project(
            str(project["id"]),
            project_name=body.project_name,
            description=body.description,
            kvstore_root_path=body.kvstore_root_path,
            is_active=body.is_active,
            metadata=body.metadata,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        invalidate_project_settings(get_context(request), str(project["id"]))
        return as_response({"project": updated})

    @projects_router.get("/{project_id}/storage-settings")
    def get_project_storage_settings(request: Request, project_id: str) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        project = _project_by_id_or_key(repository, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        _require_project_management(auth, repository, str(project["id"]))
        effective = resolve_project_storage_settings(
            get_context(request),
            str(project["id"]),
        )
        return as_response(
            {
                "project_id": project["id"],
                "configured": (project.get("settings") or {}).get("storage", {}),
                "effective": effective.as_dict(),
            }
        )

    @projects_router.patch("/{project_id}/storage-settings")
    def update_project_storage_settings(
        request: Request,
        project_id: str,
        body: UpdateProjectStorageSettingsRequest,
    ) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        project = _project_by_id_or_key(repository, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        _require_project_management(auth, repository, str(project["id"]))
        try:
            patch = storage_settings_payload(
                frame_encoding=body.frame_encoding,
                frame_quality=body.frame_quality,
                roi_encoding=body.roi_encoding,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not patch:
            raise HTTPException(status_code=422, detail="Provide at least one storage setting.")
        settings = merge_project_settings(project.get("settings"), patch)
        updated = repository.update_project(str(project["id"]), settings=settings)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        context = get_context(request)
        invalidate_project_settings(context, str(project["id"]))
        effective = resolve_project_settings(context, str(project["id"]))
        return as_response(
            {
                "project": updated,
                "configured": settings.get("storage", {}),
                "effective": effective.storage.as_dict(),
            }
        )

    @projects_router.delete("/{project_id}")
    def delete_project(request: Request, project_id: str) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        project = _project_by_id_or_key(repository, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        if str(project["id"]) == DEFAULT_PROJECT_ID or str(project.get("project_key")) == "default":
            raise HTTPException(status_code=422, detail="The default project cannot be deleted.")
        _require_project_management(auth, repository, str(project["id"]))
        deleted = repository.deactivate_project(
            str(project["id"]),
            metadata={"deleted_by_user_id": auth.user_id},
        )
        if deleted is None:
            raise HTTPException(status_code=404, detail=f"Project {project_id!r} was not found.")
        return as_response({"deleted": True, "project": deleted})

    @users_router.get("")
    def list_users(
        request: Request,
        include_all_projects: bool = False,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        auth = require_auth(request)
        if include_all_projects and not auth.is_admin:
            raise HTTPException(status_code=403, detail="User admin permission is required to list all users.")
        repository = get_repository(request)
        users = repository.list_users(
            project_id=None if include_all_projects else auth.project_id,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )
        return as_response({"users": [_public_user(user) for user in users]})

    @users_router.post("")
    def create_user(request: Request, body: CreateUserRequest) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        project = _target_membership_project(auth, repository, body.project_id, body.project_key)
        _require_project_management(auth, repository, str(project["id"]))
        role = _normalize_requested_role(body.role)
        if repository.get_user_by_username(body.username) is not None:
            raise HTTPException(status_code=409, detail=f"User {body.username!r} already exists.")
        if not auth.is_admin:
            if body.is_admin:
                raise HTTPException(status_code=403, detail="Only user admins can create user admin accounts.")
            if role not in {"viewer", "editor"}:
                raise HTTPException(status_code=403, detail="Project managers can create users with viewer or editor roles only.")
        try:
            user = repository.create_user(
                body.username,
                password=body.password,
                display_name=body.display_name,
                is_admin=body.is_admin,
                is_active=body.is_active,
                metadata=body.metadata,
            )
            membership = repository.add_project_member(str(user["id"]), str(project["id"]), role=role)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return as_response({"user": _public_user(user), "membership": membership})

    @users_router.post("/{user_id}/deactivate")
    def deactivate_user(request: Request, user_id: str) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        user = _user_by_id_or_username(repository, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        if str(user["id"]) == auth.user_id:
            raise HTTPException(status_code=422, detail="A session cannot deactivate its own user account.")
        _require_user_management(auth, repository, user)
        deactivated = repository.deactivate_user(
            str(user["id"]),
            metadata={"deactivated_by_user_id": auth.user_id},
        )
        if deactivated is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        return as_response({"deactivated": True, "user": _public_user(deactivated)})

    @users_router.post("/{user_id}/reset-password")
    def reset_user_password(request: Request, user_id: str, body: ResetPasswordRequest) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        user = _user_by_id_or_username(repository, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        _require_user_management(auth, repository, user)
        updated = repository.reset_user_password(
            str(user["id"]),
            body.password,
            metadata={"password_reset_by_user_id": auth.user_id},
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        return as_response({"reset": True, "user": _public_user(updated)})

    @users_router.delete("/{user_id}")
    def delete_user(request: Request, user_id: str) -> dict:
        auth = require_auth(request)
        repository = get_repository(request)
        user = _user_by_id_or_username(repository, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        if str(user["id"]) == auth.user_id:
            raise HTTPException(status_code=422, detail="A session cannot delete its own user account.")
        _require_user_management(auth, repository, user)
        deleted = repository.delete_user(str(user["id"]))
        if deleted is None:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} was not found.")
        return as_response({"deleted": True, "user": _public_user(deleted)})
else:
    router = None
    projects_router = None
    users_router = None
    routers = []
