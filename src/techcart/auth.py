"""Microsoft Entra ID sign-in (OpenID Connect authorization-code flow with PKCE, via MSAL) + server-side sessions.

* No passwords are stored here: Entra ID authenticates; we keep a customer row keyed by (tenant id, object id).
* Sessions: random token in an HttpOnly cookie; only its SHA-256 hash is stored in the `sessions` table.
* Admin access: the signed-in user's email must be listed in ADMIN_EMAILS.
* Identity always comes from the session on the server - never from a customer id sent by the browser.

App registration (Azure portal > Microsoft Entra ID > App registrations > New registration):
  * Supported account types: "Accounts in any organizational directory and personal Microsoft accounts"
    for a public shop (then ENTRA_TENANT=common), or "single tenant" for staff-only (ENTRA_TENANT=<tenant id>)
  * Redirect URI (Web): http://localhost:8000/auth/callback
  * Certificates & secrets > New client secret -> ENTRA_CLIENT_SECRET
"""
from __future__ import annotations

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from urllib.parse import quote

from . import services
from .config import settings

COOKIE = "tc_session"
SCOPES = ["User.Read"]          # MSAL adds openid/profile/offline_access itself
_FLOWS: dict[str, dict] = {}    # state -> pending auth-code flow (expires after 10 minutes)


class AuthNotConfigured(RuntimeError):
    pass


def configured() -> bool:
    return bool(settings.entra_client_id and settings.entra_client_secret)


def _authority() -> str:
    return f"https://login.microsoftonline.com/{settings.entra_tenant}"


def _app():
    if not configured():
        raise AuthNotConfigured("Set ENTRA_CLIENT_ID and ENTRA_CLIENT_SECRET (see README: Microsoft Entra ID sign-in)")
    import msal

    return msal.ConfidentialClientApplication(settings.entra_client_id, client_credential=settings.entra_client_secret,
                                              authority=_authority())


# ------------------------------------------------------------------------------------------ login flow
def begin_login(next_path: str = "/") -> str:
    """Returns the Microsoft sign-in URL to redirect the browser to."""
    now = time.time()
    for k in [k for k, f in _FLOWS.items() if f["_ts"] < now - 600]:
        _FLOWS.pop(k, None)
    flow = _app().initiate_auth_code_flow(SCOPES, redirect_uri=settings.entra_redirect_uri, prompt="select_account")
    flow["_ts"], flow["_next"] = now, next_path if next_path.startswith("/") else "/"
    _FLOWS[flow["state"]] = flow
    return flow["auth_uri"]


def complete_login(query: dict) -> tuple[str, str]:
    """Handles the redirect back from Entra ID. Returns (session token, path to go to next)."""
    flow = _FLOWS.pop(query.get("state", ""), None)
    if not flow:
        raise PermissionError("Sign-in expired or was started elsewhere. Please try again.")
    next_path = flow.pop("_next", "/")
    flow.pop("_ts", None)
    result = _app().acquire_token_by_auth_code_flow(flow, query)
    if "error" in result:
        raise PermissionError(result.get("error_description") or result["error"])
    claims = result["id_token_claims"]
    customer_id = upsert_customer(claims)
    return create_session(customer_id), next_path


def upsert_customer(claims: dict) -> str:
    oid, tid = claims["oid"], claims.get("tid", "")
    email = claims.get("email") or claims.get("preferred_username") or ""
    now = datetime.now().isoformat(timespec="seconds")
    existing = services.q("SELECT customer_id FROM customers WHERE entra_oid=? AND tenant_id=?", (oid, tid))
    if existing:
        cid = existing[0]["customer_id"]
        services.execute("UPDATE customers SET name=?, email=?, last_login_at=? WHERE customer_id=?",
                         (claims.get("name", ""), email, now, cid))
        return cid
    cid = "U" + hashlib.sha256(f"{tid}:{oid}".encode()).hexdigest()[:12]
    services.execute("""INSERT INTO customers (customer_id, name, email, created_at, marketing_consent, is_business,
                        entra_oid, tenant_id, last_login_at) VALUES (?,?,?,?,0,0,?,?,?)""",
                     (cid, claims.get("name", ""), email, now, oid, tid, now))
    services.audit("-", cid, "customer_registered", {"via": "entra_id"})
    return cid


def logout_url(post_logout: str) -> str:
    return f"{_authority()}/oauth2/v2.0/logout?post_logout_redirect_uri={quote(post_logout, safe='')}"


# ------------------------------------------------------------------------------------------ sessions
def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(customer_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    services.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                     (_hash(token), customer_id, now.isoformat(timespec="seconds"),
                      (now + timedelta(hours=settings.session_ttl_hours)).isoformat(timespec="seconds")))
    return token


def end_session(token: str | None) -> None:
    if token:
        services.execute("DELETE FROM sessions WHERE token_hash=?", (_hash(token),))


def user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    r = services.q("""SELECT c.customer_id, c.name, c.email, c.city, c.phone, c.marketing_consent, c.created_at
                      FROM sessions s JOIN customers c USING(customer_id)
                      WHERE s.token_hash=? AND s.expires_at > ?""", (_hash(token), datetime.now().isoformat()))
    if not r:
        return None
    u = r[0]
    u["is_admin"] = (u["email"] or "").lower() in settings.admin_emails
    return u


def update_profile(customer_id: str, city: str | None, phone: str | None, marketing_consent: bool) -> None:
    services.execute("UPDATE customers SET city=?, phone=?, marketing_consent=? WHERE customer_id=?",
                     (city, phone, int(marketing_consent), customer_id))
    services.audit("-", customer_id, "profile_updated", {"consent": bool(marketing_consent)})
