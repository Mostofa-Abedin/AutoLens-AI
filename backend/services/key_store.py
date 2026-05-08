import json
import os
from pathlib import Path

from cryptography.fernet import Fernet

_STORE_PATH = Path(__file__).parent.parent / "data" / "key_store.enc"
_ENV_PATH   = Path(__file__).parent.parent / ".env"
_ENV_VAR    = "AUTOLENS_SECRET_KEY"


def _fernet() -> Fernet:
    key = os.environ.get(_ENV_VAR, "").strip()
    if not key:
        key = Fernet.generate_key().decode()
        os.environ[_ENV_VAR] = key
        with open(_ENV_PATH, "a") as f:
            f.write(f"\n{_ENV_VAR}={key}\n")
    return Fernet(key.encode())


def _load() -> dict:
    if not _STORE_PATH.exists():
        return {}
    return json.loads(_STORE_PATH.read_text())


def _save(store: dict) -> None:
    _STORE_PATH.write_text(json.dumps(store))


def save_key(provider: str, api_key: str) -> None:
    store = _load()
    store[provider] = _fernet().encrypt(api_key.encode()).decode()
    _save(store)


def load_key(provider: str) -> str | None:
    store = _load()
    if provider not in store:
        return None
    return _fernet().decrypt(store[provider].encode()).decode()


def delete_key(provider: str) -> None:
    store = _load()
    store.pop(provider, None)
    _save(store)


def list_saved() -> list[str]:
    return list(_load().keys())
