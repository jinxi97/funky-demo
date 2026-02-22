import asyncio
import atexit
import uuid

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
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


def _create_sub_agent(task_prompt: str, ws: Workspace) -> tuple[InMemoryRunner, str, str]:
    """Create an independent ADK environment for a single sub-agent task."""
    agent = Agent(
        model="gemini-3-flash-preview",
        name="sub_agent",
        description="A sub-agent that executes tasks in its own Funky workspace.",
        instruction=(
            "You are a helpful assistant that can handle user requests. "
            "You have access to the 'execute_command' tool to run shell "
            "commands in your workspace."
        ),
        tools=[_make_sub_execute_command(ws)],
    )

    runner = InMemoryRunner(agent=agent)
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    return runner, user_id, session_id


async def _run_sub_agent(task_prompt: str, ws: Workspace) -> str:
    """Run a single sub-agent to completion and return its final response."""
    runner, user_id, session_id = _create_sub_agent(task_prompt, ws)

    session = await runner.session_service.create_session(
        app_name=runner.app_name,
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


async def spawn_sub_agents(task_prompts: list[str]) -> dict:
    """Spawn multiple sub-agents, each running independently in its own ADK environment.

    Args:
        task_prompts: A list of task descriptions. Each string becomes the prompt
                      for one sub-agent. The number of strings determines how many
                      sub-agents are spawned.

    Returns:
        A dictionary with a 'results' key containing a list of dicts, each with
        the original 'prompt' and the sub-agent's 'response'.
    """
    forked_workspaces = Workspace.fork(
        workspace, num_of_workspace=len(task_prompts))
    tasks = [_run_sub_agent(prompt, ws)
             for prompt, ws in zip(task_prompts, forked_workspaces)]
    responses = await asyncio.gather(*tasks)

    for ws in forked_workspaces:
        ws.delete()

    results = []
    for prompt, response in zip(task_prompts, responses):
        results.append({"prompt": prompt, "response": response})

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
        "A manager agent that coordinates work by spawning multiple sub-agents. "
        "Each sub-agent runs in its own independent environment."
    ),
    instruction=(
        "You are a manager agent. Your primary capability is delegating tasks to "
        "sub-agents using the 'spawn_sub_agents' tool.\n\n"
        "When you receive a request that can be broken into independent tasks, "
        "decompose it into clear, self-contained task prompts and pass them as a "
        "list to 'spawn_sub_agents'. Each sub-agent gets its own workspace and "
        "can execute shell commands independently.\n\n"
        "After the sub-agents complete, review their responses, synthesize the "
        "results, and present a unified answer to the user."
    ),
    tools=[spawn_sub_agents, execute_command],
)
