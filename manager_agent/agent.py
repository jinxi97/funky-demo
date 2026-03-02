import asyncio
import atexit
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from google.adk.tools import ToolContext
from google.genai import types

from manager_agent.workspace_utils import Workspace

workspace = Workspace.create()
atexit.register(workspace.delete)


def _make_sub_execute_command(ws: Workspace):
    """Create an execute_command tool function bound to a specific workspace."""
    def execute_command(command: str) -> dict:
        """Execute a shell command in this agent's workspace.

        Args:
            command: The shell command to run (e.g. "ls -la", "python script.py").

        Returns:
            A dictionary containing the command execution result.
        """
        return ws.exec(command)
    return execute_command


def _create_sub_agent(
    ws: Workspace,
    session_service: BaseSessionService,
    app_name: str,
    agent_name: str,
) -> tuple[Runner, str]:
    """Create a sub-agent that shares the root agent's session service and app_name."""
    agent = Agent(
        model="gemini-3-flash-preview",
        name=agent_name,
        description="A sub-agent that executes tasks in its own Funky workspace.",
        instruction=(
            "You are a helpful assistant that can handle user requests. "
            "You have access to the 'execute_command' tool to run shell "
            "commands in your workspace."
        ),
        tools=[_make_sub_execute_command(ws)],
    )

    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )
    session_id = f"session_{agent_name}_{uuid.uuid4().hex[:8]}"
    return runner, session_id


async def _run_sub_agent(
    task_prompt: str,
    ws: Workspace,
    session_service: BaseSessionService,
    app_name: str,
    user_id: str,
    agent_name: str,
) -> str:
    """Run a single sub-agent to completion and return its final response."""
    runner, session_id = _create_sub_agent(
        ws, session_service, app_name, agent_name
    )

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    response_parts: list[str] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.UserContent(
            parts=[types.Part(text=task_prompt)]
        ),
    ):
        if not event.partial and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else "(no response)"


async def spawn_sub_agents(
    task_prompts: list[dict], tool_context: ToolContext
) -> dict:
    """Spawn multiple sub-agents, each running independently in its own ADK environment.

    Each sub-agent runs in a separate session within the same ADK app, so its
    activities are visible in the ADK web UI.

    Args:
        task_prompts: A list of task objects. Each object has a 'name' key for
                      the sub-agent's name (used as its app_name in the UI) and
                      a 'prompt' key for the task description.
        tool_context: Injected by ADK. Used to share the root agent's session
                      service so sub-agent sessions appear in the UI.

    Returns:
        A dictionary with a 'results' key containing a list of dicts, each with
        the original 'prompt', 'name', and the sub-agent's 'response'.
    """
    invocation_ctx = tool_context._invocation_context
    session_service = invocation_ctx.session_service
    app_name = invocation_ctx.app_name
    user_id = invocation_ctx.user_id

    forked_workspaces = Workspace.fork(
        workspace, num_of_workspace=len(task_prompts))
    tasks = [
        _run_sub_agent(task["prompt"], ws, session_service,
                       app_name, user_id, task["name"])
        for task, ws in zip(task_prompts, forked_workspaces)
    ]
    responses = await asyncio.gather(*tasks)

    for ws in forked_workspaces:
        ws.delete()

    results = []
    for task, response in zip(task_prompts, responses):
        results.append(
            {"name": task["name"], "prompt": task["prompt"], "response": response})

    return {"results": results}


def execute_command(command: str) -> dict:
    """Execute a shell command in the workspace.

    Args:
        command: The shell command to run (e.g. "ls -la", "python script.py").

    Returns:
        A dictionary containing the command execution result.
    """
    return workspace.exec(command)


root_agent = Agent(
    model="gemini-3-flash-preview",
    name="manager_agent",
    description=(
        "A manager agent that gets workspace context from a GitHub repository "
        "and coordinates work by spawning sub-agents when useful."
    ),
    instruction=(
        "You are a helpful agent with capability to spawn more sub-agents to do tasks if needed.\n\n"
        "Always ask the user for a GitHub repository URL first. After receiving "
        "the URL, clone the repository into the workspace and start by reading "
        "README.md to understand the project context.\n\n"
        "Use 'execute_command' for direct workspace operations. If the user's "
        "request can be split into independent tasks, decompose it into clear, "
        "self-contained prompts and call 'spawn_sub_agents' with a list of task "
        "objects, each with a 'name' (a descriptive agent name) and a 'prompt'. "
        "Each sub-agent runs in its own workspace.\n\n"
        "After sub-agents finish, review their responses, synthesize the results, "
        "and provide one unified answer."
    ),
    tools=[spawn_sub_agents, execute_command],
)
