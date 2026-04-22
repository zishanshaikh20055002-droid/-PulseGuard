from datetime import datetime, timedelta
from typing import Optional
import json
import os

from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
IS_DEV = APP_ENV in {"dev", "development", "local", "test", "testing"}

# In non-dev environments this must be configured. Generate one with:
#   openssl rand -hex 32
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
if not SECRET_KEY:
    if IS_DEV:
        SECRET_KEY = "dev-only-change-this-in-production"
    else:
        raise RuntimeError("JWT_SECRET_KEY must be set outside development")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _load_users_from_env() -> dict[str, dict[str, str]]:
    """
    Load users from AUTH_USERS_JSON, or from role-specific password hashes.

    AUTH_USERS_JSON example:
      {"admin":{"hashed_password":"$2b$...","role":"admin"}}
    """
    users_json = os.getenv("AUTH_USERS_JSON", "").strip()
    if users_json:
        parsed = json.loads(users_json)
        if not isinstance(parsed, dict):
            raise RuntimeError("AUTH_USERS_JSON must be a JSON object")

        users = {}
        for username, record in parsed.items():
            if not isinstance(record, dict):
                continue
            normalized = str(username).strip().lower()
            hashed = str(record.get("hashed_password", "")).strip()
            role = str(record.get("role", "operator")).strip().lower()
            if normalized and hashed and role in {"admin", "operator"}:
                users[normalized] = {
                    "username": normalized,
                    "hashed_password": hashed,
                    "role": role,
                }

        if users:
            return users
        raise RuntimeError("AUTH_USERS_JSON did not contain any usable users")

    users = {}
    for username, role in (("admin", "admin"), ("operator", "operator")):
        hashed = os.getenv(f"{username.upper()}_PASSWORD_HASH", "").strip()
        if hashed:
            users[username] = {
                "username": username,
                "hashed_password": hashed,
                "role": role,
            }

    if users:
        return users

    if not IS_DEV:
        raise RuntimeError(
            "Configure AUTH_USERS_JSON or *_PASSWORD_HASH outside development"
        )

    return {
        "admin": {
            "username": "admin",
            "hashed_password": pwd_context.hash(os.getenv("DEV_ADMIN_PASSWORD", "admin123")),
            "role": "admin",
        },
        "operator": {
            "username": "operator",
            "hashed_password": pwd_context.hash(os.getenv("DEV_OPERATOR_PASSWORD", "operator123")),
            "role": "operator",
        },
    }


USERS_DB = _load_users_from_env()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class User(BaseModel):
    username: str
    role: str


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_user(username: str | None) -> Optional[dict]:
    if not username:
        return None
    return USERS_DB.get(str(username).strip().lower())


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency for REST endpoints. Reads Bearer token from the header."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = get_user(payload.get("sub"))
        if not user:
            raise credentials_exception
        return User(username=user["username"], role=user["role"])
    except JWTError:
        raise credentials_exception


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for admin-only endpoints."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def get_ws_user(websocket: WebSocket) -> Optional[User]:
    """Verify JWT from WebSocket query param: ws://host/ws?token=..."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = get_user(payload.get("sub"))
        if not user:
            await websocket.close(code=1008)
            return None
        return User(username=user["username"], role=user["role"])
    except JWTError:
        await websocket.close(code=1008)
        return None
