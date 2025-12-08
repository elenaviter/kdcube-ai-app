# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# auth/sessions
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Dict, List

from redis import asyncio as aioredis

from kdcube_ai_app.auth.AuthManager import User
from kdcube_ai_app.infra.namespaces import REDIS, ns_key

ATOMIC_SESSION_GET_OR_CREATE = r"""
-- Atomic get-or-create for any session key.
-- Semantics:
--   session key:  <SESSION_PREFIX>:<kind>:<id>
--   index key:    <SESSION_INDEX_PREFIX>:<session_id> -> session_key

-- KEYS[1] = session_key

-- ARGV[1] = new_session_json
-- ARGV[2] = ttl_sec (number)
-- ARGV[3] = index_prefix (string)
-- ARGV[4] = refresh_existing_ttl ("1" or "0")  -- optional

local session_key = KEYS[1]

local new_json = ARGV[1]
local ttl = tonumber(ARGV[2]) or 0
local index_prefix = ARGV[3]
local refresh = ARGV[4] or "0"

local function set_with_ttl(key, val)
    if ttl > 0 then
        redis.call("SETEX", key, ttl, val)
    else
        redis.call("SET", key, val)
    end
end

local function expire_if_needed(key)
    if ttl > 0 then
        redis.call("EXPIRE", key, ttl)
    end
end

local function ensure_index_for(json_val)
    local ok, obj = pcall(cjson.decode, json_val)
    if ok and obj then
        local sid = obj["session_id"] or obj["id"]
        if sid and index_prefix then
            local index_key = index_prefix .. ":" .. sid

            if redis.call("EXISTS", index_key) == 0 then
                set_with_ttl(index_key, session_key)
            elseif refresh == "1" then
                expire_if_needed(index_key)
            end

            return index_key
        end
    end
    return nil
end

-- If session exists: return it (and best-effort repair index)
if redis.call("EXISTS", session_key) == 1 then
    local existing = redis.call("GET", session_key)

    local idx = ensure_index_for(existing)

    if refresh == "1" and ttl > 0 then
        redis.call("EXPIRE", session_key, ttl)
        if idx then redis.call("EXPIRE", idx, ttl) end
    end

    return {0, existing}
end

-- Create new session
set_with_ttl(session_key, new_json)

-- Create corresponding session_id -> session_key index
ensure_index_for(new_json)

return {1, new_json}
"""



@dataclass
class RequestContext:
    """Framework-agnostic request context"""
    client_ip: str
    user_agent: str
    authorization_header: Optional[str] = None
    id_token: Optional[str] = None

    def get_fingerprint(self) -> str:
        """Generate user fingerprint"""
        fingerprint_data = f"{self.client_ip}:{self.user_agent}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

class UserType(Enum):
    ANONYMOUS = "anonymous"
    REGISTERED = "registered"
    PRIVILEGED = "privileged"
    PAYED = "payed"


@dataclass
class UserSession:
    """Simple user session data"""
    session_id: str
    user_type: UserType
    fingerprint: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    created_at: float = 0
    last_seen: float = 0
    email: Optional[str] = None
    request_context: Optional[RequestContext] = None

    def __post_init__(self):
        if isinstance(self.user_type, str):
            # in case someone passed "REGISTERED" or "UserType.REGISTERED"
            token = self.user_type.split(".")[-1]
            try:
                self.user_type = UserType[token]      # by NAME
            except KeyError:
                self.user_type = UserType(token.lower())
        if isinstance(self.request_context, dict):
            try:
                self.request_context = RequestContext(**self.request_context)
            except Exception:
                self.request_context = None

        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []
        if self.created_at == 0:
            self.created_at = time.time()
        self.last_seen = time.time()

    def to_user(self) -> User:
        """Convert session to User object for auth validation"""
        return User(
            username=self.username or self.fingerprint,
            email=self.email,
            name=self.username,
            roles=self.roles,
            permissions=self.permissions
        )

    def serialize_to_dict(self):
        session_dict=asdict(self)
        if isinstance(self.user_type, UserType):
            session_dict["user_type"] = self.user_type.value
        return session_dict

class SessionManager:
    """Simple session management"""

    def __init__(self,
                 redis_url: str,
                 tenant: str,
                 project: str,
                 session_ttl: int = 86400):
        self.redis_url = redis_url
        self.redis = None
        self.tenant = tenant
        self.project = project
        self.SESSION_PREFIX = self.ns(REDIS.SESSION)
        self.SESSION_TTL = session_ttl
        self.SESSION_INDEX_PREFIX = f"{self.SESSION_PREFIX}:index"

    def ns(self, base: str) -> str:
        return ns_key(base, tenant=self.tenant, project=self.project)

    async def init_redis(self):
        if not self.redis:
            self.redis = aioredis.from_url(self.redis_url)

    # async def get_or_create_session(self,
    #                                 context: RequestContext,
    #                                 user_type: UserType,
    #                                 user_data: Optional[Dict] = None) -> UserSession:
    #     """Get existing session or create new one"""
    #     await self.init_redis()
    #
    #     fingerprint = context.get_fingerprint()
    #
    #     if user_type in [UserType.REGISTERED, UserType.PRIVILEGED] and user_data:
    #         session_key = f"{self.SESSION_PREFIX}:registered:{user_data['user_id']}"
    #     else:
    #         session_key = f"{self.SESSION_PREFIX}:anonymous:{fingerprint}"
    #
    #     # Try to get existing session
    #     session_data = await self.redis.get(session_key)
    #     if session_data:
    #         try:
    #             session_dict = json.loads(session_data)
    #             # session_dict["user_type"] = UserType(session_dict["user_type"])
    #
    #             session = UserSession(**session_dict)
    #
    #             # update this from IDP info!
    #             session.last_seen = time.time()
    #             session.user_type = user_type
    #
    #             if user_data:
    #                 session.roles = user_data.get('roles', [])
    #                 session.permissions = user_data.get('permissions', [])
    #             session.request_context = context
    #             await self._save_session(session_key, session)
    #             return session
    #         except Exception as e:
    #             import traceback
    #             print(traceback.format_exc())
    #             raise e
    #
    #     # Create new session
    #     session = UserSession(
    #         session_id=str(uuid.uuid4()),
    #         user_type=user_type,
    #         fingerprint=fingerprint,
    #         user_id=user_data.get("user_id") if user_data else None,
    #         username=user_data.get("username") if user_data else None,
    #         roles=user_data.get("roles", []) if user_data else [],
    #         permissions=user_data.get("permissions", []) if user_data else [],
    #         request_context=context
    #     )
    #
    #     await self._save_session(session_key, session)
    #     return session

    async def get_or_create_session(
            self,
            context: RequestContext,
            user_type: UserType,
            user_data: Optional[Dict] = None
    ) -> UserSession:
        """Get existing session or create new one (atomic for all types)."""
        await self.init_redis()

        fingerprint = context.get_fingerprint()

        # Decide key exactly as before
        if user_type in [UserType.REGISTERED, UserType.PRIVILEGED] and user_data:
            session_key = f"{self.SESSION_PREFIX}:registered:{user_data['user_id']}"
        else:
            session_key = f"{self.SESSION_PREFIX}:anonymous:{fingerprint}"

        # Build candidate (GUID session_id) exactly as your semantics
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_type=user_type,
            fingerprint=fingerprint,
            user_id=user_data.get("user_id") if user_data else None,
            username=user_data.get("username") if user_data else None,
            roles=user_data.get("roles", []) if user_data else [],
            permissions=user_data.get("permissions", []) if user_data else [],
            email=user_data.get("email") if user_data else None,
            request_context=context
        )

        payload = json.dumps(session.serialize_to_dict(), ensure_ascii=False)

        # Atomic get-or-create + index repair/create
        result = await self.redis.eval(
            ATOMIC_SESSION_GET_OR_CREATE,
            1,  # number of KEYS
            session_key,
            payload,
            str(self.SESSION_TTL),
            self.SESSION_INDEX_PREFIX,
            "0",  # no sliding TTL here; Python save below keeps your current refresh semantics
        )

        created_flag = result[0]
        session_json = result[1]

        if isinstance(created_flag, (bytes, bytearray)):
            created_flag = int(created_flag.decode("utf-8"))
        else:
            created_flag = int(created_flag)

        if isinstance(session_json, (bytes, bytearray)):
            session_json = session_json.decode("utf-8")

        stored = UserSession(**json.loads(session_json))

        # Keep session_id intact; update mutable fields like before
        stored.last_seen = time.time()
        stored.user_type = user_type

        if user_data:
            stored.roles = user_data.get("roles", stored.roles or [])
            stored.permissions = user_data.get("permissions", stored.permissions or [])
            stored.user_id = user_data.get("user_id", stored.user_id)
            stored.username = user_data.get("username", stored.username)
            stored.email = user_data.get("email", stored.email)

        stored.request_context = context

        # Preserve your existing "always save" semantics (TTL + index write)
        await self._save_session(session_key, stored, write_index=True)

        return stored


    async def _save_session(self, session_key: str, session: UserSession, *, write_index: bool = True):

        await self.init_redis()
        session_data = json.dumps(session.serialize_to_dict(), ensure_ascii=False)
        # Store session JSON with TTL
        await self.redis.setex(session_key, self.SESSION_TTL, session_data)

        if write_index:
            # Store index: session_id -> session_key
            index_key = f"{self.SESSION_INDEX_PREFIX}:{session.session_id}"
            await self.redis.setex(index_key, self.SESSION_TTL, session_key)

    async def get_session_by_id(self, session_id: str) -> Optional[UserSession]:
        """Retrieve a session by its session_id"""
        await self.init_redis()
        index_key = f"{self.SESSION_INDEX_PREFIX}:{session_id}"
        session_key_raw = await self.redis.get(index_key)
        session_key = self._as_str(session_key_raw)
        if not session_key:
            return None

        session_data_raw = await self.redis.get(session_key)
        session_dict = self._loads_json(session_data_raw)
        if not session_dict:
            return None

        try:
            return UserSession(**session_dict)
        except Exception:
            # If schema changed or stored data is partial/corrupt
            return None

    def _as_str(self, v: Optional[object]) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                # last-resort decode; should be rare
                return v.decode("utf-8", errors="replace")
        return str(v)

    def _loads_json(self, s: Optional[object]) -> Optional[Dict]:
        text = self._as_str(s)
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

