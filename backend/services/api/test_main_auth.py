import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

try:
    from jose import jwt
except ImportError:
    import jwt


MODULE_PATH = Path(__file__).with_name("main.py")
SPEC = importlib.util.spec_from_file_location("novacron_api_main", MODULE_PATH)
api_main = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(api_main)


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch):
    api_main.app.state.config = {}
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)


def test_load_config_raises_when_config_file_is_missing(monkeypatch):
    monkeypatch.setenv("CONFIG_PATH", "/tmp/novacron-missing-api-config.yaml")

    with pytest.raises(RuntimeError, match="Failed to load configuration"):
        api_main.load_config()


def test_load_config_expands_environment_variables(tmp_path, monkeypatch):
    config_path = tmp_path / "api.yaml"
    config_path.write_text(
        "\n".join(
            [
                "hypervisor:",
                "  url: http://localhost:9000",
                "auth:",
                "  enabled: true",
                "  jwt_secret: ${JWT_SECRET}",
            ]
        )
    )
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("JWT_SECRET", "super-secret")

    config = api_main.load_config()

    assert config["auth"]["jwt_secret"] == "super-secret"
    assert api_main.app.state.config["auth"]["jwt_secret"] == "super-secret"


@pytest.mark.asyncio
async def test_get_current_user_fails_closed_when_auth_is_disabled():
    api_main.app.state.config = {"auth": {"enabled": False}}

    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_current_user(None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Authentication is disabled"


@pytest.mark.asyncio
async def test_get_current_user_requires_credentials():
    api_main.app.state.config = {"auth": {"enabled": True, "jwt_secret": "super-secret"}}

    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_current_user(None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_get_current_user_rejects_invalid_bearer_token():
    api_main.app.state.config = {"auth": {"enabled": True, "jwt_secret": "super-secret"}}
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-jwt")

    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_current_user(credentials)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid authentication credentials"


@pytest.mark.asyncio
async def test_get_current_user_rejects_misconfigured_auth_secret():
    api_main.app.state.config = {"auth": {"enabled": True, "jwt_secret": "${JWT_SECRET}"}}
    token = jwt.encode({"sub": "user-123"}, "super-secret", algorithm="HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_current_user(credentials)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Authentication is misconfigured"


@pytest.mark.asyncio
async def test_get_current_user_validates_token_and_does_not_default_to_admin():
    api_main.app.state.config = {"auth": {"enabled": True, "jwt_secret": "super-secret"}}
    token = jwt.encode({"sub": "user-123"}, "super-secret", algorithm="HS256")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = await api_main.get_current_user(credentials)

    assert user == {"id": "user-123", "role": "user"}


@pytest.mark.asyncio
async def test_get_current_user_preserves_role_claim_from_token():
    api_main.app.state.config = {"auth": {"enabled": True, "jwt_secret": "super-secret"}}
    token = jwt.encode(
        {"sub": "operator-1", "role": "operator"},
        "super-secret",
        algorithm="HS256",
    )
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = await api_main.get_current_user(credentials)

    assert user == {"id": "operator-1", "role": "operator"}
