"""PBX-neutral contract for provisioning adapters.

Adapters plan and reconcile PBX-owned objects outside modem workers.  The plan
summary is safe to display; adapter-private artifacts may contain credentials
and must never appear in reprs or logs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ProvisionTarget:
    """PBX target identity and adapter-specific non-secret settings."""
    adapter_id: str
    identity: str
    settings: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ModemSipAccount:
    """Normalized modem account; secrets are excluded from repr/log output."""
    modem_id: str
    username: str
    password: str = field(repr=False)
    registrar: str = ''
    caller_id: str = ''


@dataclass(frozen=True)
class ProvisionRequest:
    """PBX-neutral desired state assembled by El-Modem's provisioning layer."""
    target: ProvisionTarget
    accounts: tuple[ModemSipAccount, ...]


@dataclass(frozen=True)
class ProvisionPlan:
    adapter_id: str
    target_id: str | None
    summary: tuple[str, ...]
    artifacts: tuple[Any, ...] = field(repr=False)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProvisionCheck:
    consistent: bool
    details: tuple[str, ...]
    registrations: tuple[tuple[str, str], ...] = ()
    runtime: tuple[tuple[str, str], ...] = ()


def read_saved_network_profile(config_path: Path, modem_id: str) -> dict[str, str]:
    """Read one modem's protected saved Hayes network profile.

    This persistence boundary is PBX-neutral so each vendor adapter consumes
    the same validated SIP account source without touching modem workers.
    """
    if not isinstance(modem_id,str) or not re.fullmatch(r'modem-[1-9][0-9]*',modem_id):
        raise ValueError('invalid modem ID')
    config_dir=Path(config_path).expanduser().resolve().parent
    instances_dir=config_dir/'instances'
    instance_dir=instances_dir/modem_id
    profile_path=instance_dir/'profiles.json'
    if instances_dir.is_symlink() or instance_dir.is_symlink() or profile_path.is_symlink():
        raise ValueError(f'{modem_id} profile path and its instance directories must not be symlinks')
    if not profile_path.is_file():
        raise ValueError(f'{modem_id} has no saved profile; save its SIP settings first')
    # Check the resolved parent too, so a symlink in any path component cannot
    # redirect credential reads outside this modem's instance directory.
    if profile_path.resolve().parent != instance_dir.resolve():
        raise ValueError(f'{modem_id} profile resolves outside its instance directory')
    try:
        profile_stat=profile_path.stat()
        if os.name!='nt' and profile_stat.st_mode&0o077:
            raise ValueError(f'{modem_id} profile permissions are too open; restrict profiles.json to its owner')
        profile=json.loads(profile_path.read_text(encoding='utf-8'))
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError(f'{modem_id} saved profile cannot be read: {exc}') from exc
    if not isinstance(profile,dict):
        raise ValueError(f'{modem_id} profile must contain an object')
    profiles=profile.get('profiles',{})
    if not isinstance(profiles,dict):
        raise ValueError(f'{modem_id} profiles section is malformed')
    snapshot=profiles.get('0',{})
    if not isinstance(snapshot,dict):
        raise ValueError(f'{modem_id} profile 0 is malformed')
    network=snapshot.get('net',{})
    if not isinstance(network,dict):
        raise ValueError(f'{modem_id} network settings are malformed')
    return {str(key):str(value) for key,value in network.items()}


class ProvisioningAdapter(Protocol):
    """Vendor adapter contract; no modem/DSP objects cross this boundary."""
    adapter_id: str

    def plan(self, request: ProvisionRequest) -> ProvisionPlan: ...

    def apply(self, plan: ProvisionPlan) -> Sequence[Any]: ...

    def check(self, plan: ProvisionPlan, *, live: bool = False) -> ProvisionCheck: ...

    def rollback(self, target: ProvisionTarget) -> Sequence[Any]: ...
