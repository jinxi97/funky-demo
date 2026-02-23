"""Utility functions for the Funky workspace management API."""

from __future__ import annotations

import time

import httpx

BASE_URL = "http://34.173.139.208"
DEFAULT_TIMEOUT = 30.0


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT)


# ---------- Health / root ----------

def health_check() -> dict:
    """GET /healthz — check API server health."""
    with _client() as c:
        resp = c.get("/healthz")
        resp.raise_for_status()
        return resp.json()


def root() -> dict:
    """GET / — root endpoint."""
    with _client() as c:
        resp = c.get("/")
        resp.raise_for_status()
        return resp.json()


# ---------- Workspaces ----------

def create_workspace() -> str:
    """POST /workspaces — create a new workspace and return its workspace_id."""
    with _client() as c:
        resp = c.post("/workspaces")
        resp.raise_for_status()
        return resp.json()["workspace_id"]


def exec_command(workspace_id: str, command: str) -> dict:
    """POST /workspaces/{workspace_id}/exec?command=... — execute a command."""
    timeout = httpx.Timeout(
        connect=10.0,
        read=300.0,   # allow long tool executions
        write=30.0,
        pool=10.0,
    )
    with _client() as c:
        resp = c.post(
            f"/workspaces/{workspace_id}/exec",
            params={"command": command},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()


def delete_workspace(workspace_id: str) -> dict:
    """DELETE /workspaces/{workspace_id} — delete a workspace."""
    with _client() as c:
        resp = c.delete(f"/workspaces/{workspace_id}")
        resp.raise_for_status()
        return resp.json()


# ---------- Snapshots ----------

def create_snapshot_trigger(workspace_id: str) -> dict:
    """POST /snapshots/triggers — create a snapshot trigger for a workspace."""
    with _client() as c:
        resp = c.post("/snapshots/triggers", json={"workspace_id": workspace_id})
        resp.raise_for_status()
        return resp.json()


def delete_snapshot_trigger(trigger_name: str) -> dict:
    """DELETE /snapshots/triggers/{trigger_name} — delete a snapshot trigger."""
    with _client() as c:
        resp = c.delete(f"/snapshots/triggers/{trigger_name}")
        resp.raise_for_status()
        return resp.json()


def get_snapshot_status(trigger_name: str) -> dict:
    """GET /snapshots/status?trigger_name=... — get snapshot status."""
    with _client() as c:
        resp = c.get("/snapshots/status", params={"trigger_name": trigger_name})
        resp.raise_for_status()
        return resp.json()


def restore_from_snapshot(snapshot_name: str) -> dict:
    """POST /snapshots/restore — restore from a named snapshot."""
    with _client() as c:
        resp = c.post("/snapshots/restore", json={"snapshot_name": snapshot_name})
        resp.raise_for_status()
        return resp.json()


# ---------- Workspace class ----------

class Workspace:
    """High-level wrapper around a Funky workspace.

    Usage::

        ws = Workspace.create()
        result = ws.exec("ls -la")
        ws.delete()
    """

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    @classmethod
    def create(cls) -> Workspace:
        """Create a new workspace and return a Workspace instance."""
        workspace_id = create_workspace()
        return cls(workspace_id)

    def exec(self, command: str) -> dict:
        """Execute a command in this workspace."""
        return exec_command(self.workspace_id, command)

    def delete(self) -> dict:
        """Delete this workspace."""
        return delete_workspace(self.workspace_id)

    def create_snapshot_trigger(self) -> dict:
        """Create a snapshot trigger for this workspace."""
        return create_snapshot_trigger(self.workspace_id)

    @classmethod
    def fork(
        cls,
        source: Workspace,
        *,
        num_of_workspace: int = 1,
        timeout: float = 60.0,
    ) -> list[Workspace]:
        """Fork a workspace by snapshotting it and restoring into new ones.

        Args:
            source: The workspace to fork from.
            num_of_workspace: Number of new workspaces to create from the snapshot.
            timeout: Maximum seconds to wait for the snapshot to become ready.

        Returns:
            A list of new Workspaces restored from the snapshot of the source.

        Raises:
            TimeoutError: If the snapshot is not ready within the timeout.
        """
        trigger = source.create_snapshot_trigger()
        trigger_name = trigger["name"]

        deadline = time.monotonic() + timeout
        while True:
            try:
                status = get_snapshot_status(trigger_name)
            except httpx.HTTPStatusError:
                status = {}
            if status.get("ready"):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Snapshot {trigger_name!r} not ready after {timeout}s"
                )
            time.sleep(1)

        snapshot_name = status["snapshot_name"]
        workspaces = []
        for _ in range(num_of_workspace):
            result = restore_from_snapshot(snapshot_name)
            workspaces.append(cls(result["workspace_id"]))
        return workspaces

    def __repr__(self) -> str:
        return f"Workspace({self.workspace_id!r})"
