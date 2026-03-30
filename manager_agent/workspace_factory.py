"""Factory for selecting a Workspace backend via the WORKSPACE_BACKEND env var.

Set WORKSPACE_BACKEND=funky_sdk  to use the funky-sdk package.
Defaults to the legacy httpx-based implementation in workspace_utils.
"""

from __future__ import annotations

import os


def get_workspace_class():
    """Return the Workspace class for the configured backend.

    Returns workspace_utils.Workspace (legacy) or _SDKWorkspace (funky-sdk)
    depending on the WORKSPACE_BACKEND environment variable.
    """
    backend = os.environ.get("WORKSPACE_BACKEND", "legacy").lower()
    if backend == "funky_sdk":
        print("Using funky-sdk workspace backend")
        return _SDKWorkspace
    print("Using legacy workspace backend")
    return _legacy_workspace()


def _legacy_workspace():
    from manager_agent.workspace_utils import Workspace
    return Workspace


# ---------------------------------------------------------------------------
# Adapter: wraps funky-sdk's Workspace to match workspace_utils.Workspace API
# ---------------------------------------------------------------------------

class _SDKWorkspace:
    """Adapter that exposes the same interface as workspace_utils.Workspace
    but delegates to funky.Workspace from funky-sdk."""

    def __init__(self, _sdk_ws) -> None:
        self._ws = _sdk_ws
        print(f"[funky-sdk] Workspace ready: claim={_sdk_ws.claim_name!r} "
              f"ns={_sdk_ws.namespace!r} pod={_sdk_ws.pod_name!r}")

    # -- creation ------------------------------------------------------------

    @classmethod
    def create(cls) -> _SDKWorkspace:
        from funky import Workspace as _FunkyWS
        sdk_ws = _FunkyWS.create(timeout=300.0)
        return cls(sdk_ws)

    # -- execution -----------------------------------------------------------

    def exec(self, command: str) -> dict:
        """Execute a command; returns a dict compatible with the legacy API."""
        from funky.errors import APIError
        try:
            result = self._ws.execute(command)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
        except APIError as e:
            return {
                "error": True,
                "status_code": e.status_code,
                "detail": str(e),
            }

    # -- lifecycle -----------------------------------------------------------

    def delete(self) -> dict:
        return self._ws.delete()

    # -- fork (snapshot + restore) ------------------------------------------

    @classmethod
    async def fork(
        cls,
        source: _SDKWorkspace,
        *,
        num_of_workspace: int = 1,
        timeout: float = 60.0,  # kept for API compatibility, not used by SDK
    ) -> list[_SDKWorkspace]:
        from funky import Workspace as _FunkyWS
        print(f"[funky-sdk] Snapshotting workspace {source._ws.claim_name!r}...")
        snapshot_name = source._ws.snapshot()
        print(f"[funky-sdk] Snapshot ready: {snapshot_name!r}, "
              f"restoring {num_of_workspace} workspace(s)...")
        results = []
        for _ in range(num_of_workspace):
            sdk_ws = _FunkyWS.restore(
                source._ws.claim_name,
                source._ws.namespace,
                snapshot_name,
                timeout=300.0,
            )
            results.append(cls(sdk_ws))
        return results

    def __repr__(self) -> str:
        return f"_SDKWorkspace({self._ws.claim_name!r})"
