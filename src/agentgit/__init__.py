"""agentgit - Convert agent transcripts to git repositories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentgit.core import (
    AssistantContext,
    AssistantTurn,
    DiscoveredTranscript,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
    SourceCommit,
    Transcript,
    TranscriptEntry,
)
from agentgit.git_builder import (
    GitRepoBuilder,
    format_commit_message,
    format_prompt_merge_message,
    format_turn_commit_message,
)
from agentgit.plugins import get_configured_plugin_manager, hookimpl, hookspec

if TYPE_CHECKING:
    from git import Repo

    from agentgit.enhance import EnhanceConfig

__version__ = "0.1.0"

# Format identifier for merged transcripts
FORMAT_MERGED = "merged"

__all__ = [
    # Core types
    "FileOperation",
    "Prompt",
    "AssistantContext",
    "AssistantTurn",
    "PromptResponse",
    "TranscriptEntry",
    "Transcript",
    "OperationType",
    "SourceCommit",
    "DiscoveredTranscript",
    # Plugin system
    "hookspec",
    "hookimpl",
    # Main functions
    "parse_transcript",
    "parse_transcripts",
    "build_repo",
    "build_repo_grouped",
    "transcript_to_repo",
    "format_commit_message",
    "format_turn_commit_message",
    "format_prompt_merge_message",
    "discover_transcripts",
    "discover_transcripts_enriched",
    "find_git_root",
]


def parse_transcript(path: Path | str, plugin_type: str | None = None) -> Transcript:
    """Parse a transcript file into structured data.

    Args:
        path: Path to the transcript file.
        plugin_type: Optional plugin type to use (e.g., 'claude_code').
            If not specified, auto-detects the format.

    Returns:
        Parsed Transcript object.

    Raises:
        ValueError: If the transcript format cannot be detected or plugin not found.
    """
    path = Path(path)

    pm = get_configured_plugin_manager()

    # Determine format - try auto-detection first, fall back to plugin_type
    format = pm.hook.agentgit_detect_format(path=path)
    if not format:
        if plugin_type:
            format = plugin_type
        else:
            raise ValueError(f"Could not detect transcript format for: {path}")

    transcript = pm.hook.agentgit_parse_transcript(path=path, format=format)
    if not transcript:
        raise ValueError(f"Failed to parse transcript: {path}")

    operations = []
    for ops in pm.hook.agentgit_extract_operations(transcript=transcript):
        operations.extend(ops)

    enriched_operations = []
    for op in operations:
        for enriched_op in pm.hook.agentgit_enrich_operation(operation=op, transcript=transcript):
            enriched_operations.append(enriched_op)
            break

    transcript.operations = enriched_operations

    # Build prompt_responses structure for grouped git history
    for prompt_responses in pm.hook.agentgit_build_prompt_responses(transcript=transcript):
        transcript.prompt_responses.extend(prompt_responses)

    # Build conversation_rounds structure for conversational git history
    for conversation_rounds in pm.hook.agentgit_build_conversation_rounds(transcript=transcript):
        transcript.conversation_rounds.extend(conversation_rounds)

    return transcript


def parse_transcripts(
    paths: list[Path | str],
    plugin_type: str | None = None,
) -> Transcript:
    """Parse multiple transcript files and merge their operations.

    Operations from all transcripts are merged and sorted by timestamp.

    Args:
        paths: List of paths to transcript files.
        plugin_type: Optional plugin type to use (e.g., 'claude_code').
            If not specified, auto-detects the format for each file.

    Returns:
        Merged Transcript object with operations sorted by timestamp.
    """
    if not paths:
        return Transcript()

    all_entries = []
    all_prompts = []
    all_operations = []
    all_prompt_responses = []

    for path in paths:
        transcript = parse_transcript(path, plugin_type=plugin_type)
        all_entries.extend(transcript.entries)
        all_prompts.extend(transcript.prompts)
        all_operations.extend(transcript.operations)
        all_prompt_responses.extend(transcript.prompt_responses)

    # Sort by timestamp
    all_entries.sort(key=lambda e: e.timestamp)
    all_prompts.sort(key=lambda p: p.timestamp)
    all_operations.sort(key=lambda op: op.timestamp)
    all_prompt_responses.sort(key=lambda pr: pr.prompt.timestamp)

    return Transcript(
        entries=all_entries,
        prompts=all_prompts,
        operations=all_operations,
        prompt_responses=all_prompt_responses,
        source_format=FORMAT_MERGED,
    )


def build_repo(
    operations: list[FileOperation],
    output_dir: Path | None = None,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    source_repo: Path | None = None,
    enhance_config: "EnhanceConfig | None" = None,
) -> tuple[Repo, Path, dict[str, str]]:
    """Build a git repository from file operations.

    Args:
        operations: List of FileOperation objects.
        output_dir: Directory for the git repo. If None, creates a temp dir.
        author_name: Name for git commits.
        author_email: Email for git commits.
        source_repo: Optional source repository to interleave commits from.
        enhance_config: Optional configuration for generating commit messages.

    Returns:
        Tuple of (repo, repo_path, path_mapping).
    """
    builder = GitRepoBuilder(
        output_dir=output_dir,
        enhance_config=enhance_config,
    )
    return builder.build(
        operations=operations,
        author_name=author_name,
        author_email=author_email,
        source_repo=source_repo,
    )


def build_repo_grouped(
    conversation_rounds: list["ConversationRound"],
    transcript: "Transcript",
    output_dir: Path | None = None,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    enhance_config: "EnhanceConfig | None" = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    incremental: bool = True,
    file_ops_only: bool = False,
) -> tuple[Repo, Path, dict[str, str]]:
    """Build a git repository using conversational structure.

    Each conversation round becomes a feature branch with commits for every entry,
    then merges back to the session branch.

    Args:
        conversation_rounds: List of ConversationRound objects from transcript.
        transcript: The full transcript (needed for operation lookup).
        output_dir: Directory for the git repo. If None, creates a temp dir.
        author_name: Name for git commits.
        author_email: Email for git commits.
        enhance_config: Optional configuration for generating commit messages.
        session_id: Optional session identifier. If provided, creates a session branch.
        agent_name: Optional agent/format name for branch naming (e.g., 'claude-code').
        incremental: If True, skip already-processed entries. Default True.
        file_ops_only: If True, only commit file creation/edit/deletion events
            (linear history, one commit per operation). Default False.

    Returns:
        Tuple of (repo, repo_path, path_mapping).
    """
    # Generate session branch name if session_id is provided
    session_branch_name = None
    if session_id and conversation_rounds:
        from agentgit.enhance import generate_session_branch_name
        # Use conversation_rounds to get prompts for session naming
        # Convert to PromptResponse format for compatibility with existing function
        prompt_responses_for_naming = []
        for round in conversation_rounds:
            from agentgit.core import PromptResponse
            prompt_responses_for_naming.append(PromptResponse(prompt=round.prompt, turns=[]))

        session_branch_name = generate_session_branch_name(
            prompt_responses_for_naming, session_id, enhance_config, agent_name
        )

    builder = GitRepoBuilder(
        output_dir=output_dir,
        enhance_config=enhance_config,
        session_branch_name=session_branch_name,
        session_id=session_id,
    )
    return builder.build_from_conversation_rounds(
        conversation_rounds=conversation_rounds,
        transcript=transcript,
        author_name=author_name,
        author_email=author_email,
        incremental=incremental,
        file_ops_only=file_ops_only,
    )


def transcript_to_repo(
    transcript_path: Path | str,
    output_dir: Path | None = None,
    author_name: str = "Agent",
    author_email: str = "agent@local",
    source_repo: Path | str | None = None,
    plugin_type: str | None = None,
    use_grouped: bool = True,
) -> tuple[Repo, Path, Transcript]:
    """Parse a transcript and build a git repository.

    Convenience function combining parse_transcript and build_repo.

    Args:
        transcript_path: Path to the transcript file.
        output_dir: Directory for the git repo. If None, creates a temp dir.
        author_name: Name for git commits.
        author_email: Email for git commits.
        source_repo: Optional source repository to interleave commits from.
        plugin_type: Optional plugin type to use (e.g., 'claude_code').
            If not specified, auto-detects the format.
        use_grouped: If True (default), use merge-based grouped structure.
            If False, use flat structure with one commit per operation.

    Returns:
        Tuple of (repo, repo_path, transcript).
    """
    transcript = parse_transcript(transcript_path, plugin_type=plugin_type)

    # Use grouped build if we have conversation_rounds and it's enabled
    if use_grouped and transcript.conversation_rounds:
        # Extract agent name from format (e.g., "claude_code_jsonl" -> "claude-code")
        agent_name = None
        if transcript.source_format:
            agent_name = transcript.source_format.replace("_jsonl", "").replace("_web", "")

        # Use session_id from transcript, or filename as fallback
        session_id = transcript.session_id or Path(transcript_path).stem

        repo, repo_path, _ = build_repo_grouped(
            conversation_rounds=transcript.conversation_rounds,
            transcript=transcript,
            output_dir=output_dir,
            author_name=author_name,
            author_email=author_email,
            session_id=session_id,
            agent_name=agent_name,
        )
    else:
        # Fall back to flat structure
        source_repo_path = Path(source_repo) if source_repo else None
        repo, repo_path, _ = build_repo(
            operations=transcript.operations,
            output_dir=output_dir,
            author_name=author_name,
            author_email=author_email,
            source_repo=source_repo_path,
        )
    return repo, repo_path, transcript


def find_git_root(start_path: Path | str | None = None) -> Path | None:
    """Find the root of the git repository containing start_path.

    Args:
        start_path: Path to start searching from. Defaults to current directory.

    Returns:
        Path to git root, or None if not in a git repository.
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    current = start_path.resolve()

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    # Check root directory
    if (current / ".git").exists():
        return current

    return None


def discover_transcripts(project_path: Path | str | None = None) -> list[Path]:
    """Discover transcript files for a project.

    If no project_path is provided, attempts to find the git root of the
    current directory.

    Args:
        project_path: Path to the project. Defaults to current git repo root.

    Returns:
        List of paths to discovered transcript files, sorted by modification
        time (most recent first).

    Raises:
        ValueError: If no project_path provided and not in a git repository.
    """
    if project_path is None:
        project_path = find_git_root()
        if project_path is None:
            raise ValueError(
                "Not in a git repository. Please specify a project path or transcript file."
            )
    else:
        project_path = Path(project_path)

    pm = get_configured_plugin_manager()

    # Collect transcripts from all plugins
    all_transcripts = []
    for transcripts in pm.hook.agentgit_discover_transcripts(project_path=project_path):
        all_transcripts.extend(transcripts)

    # Sort by modification time, most recent first
    all_transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_transcripts


def discover_transcripts_enriched(
    project_path: Path | str | None = None,
    all_projects: bool = False,
) -> list[DiscoveredTranscript]:
    """Discover transcript files with enriched metadata.

    Returns DiscoveredTranscript objects containing file metadata and
    detected format type, grouped by plugin.

    Args:
        project_path: Path to the project. Defaults to current git repo root.
            Ignored if all_projects is True.
        all_projects: If True, returns all transcripts from all projects.

    Returns:
        List of DiscoveredTranscript objects, sorted by modification time
        (most recent first).

    Raises:
        ValueError: If no project_path provided and not in a git repository
            (unless all_projects is True).
    """
    if all_projects:
        # Discover all transcripts regardless of project
        project_path = None
    elif project_path is None:
        project_path = find_git_root()
        if project_path is None:
            raise ValueError(
                "Not in a git repository. Please specify a project path or transcript file."
            )
    else:
        project_path = Path(project_path)

    pm = get_configured_plugin_manager()

    # Collect transcripts from all plugins
    all_paths: list[Path] = []
    for transcripts in pm.hook.agentgit_discover_transcripts(project_path=project_path):
        all_paths.extend(transcripts)

    # Enrich each transcript with format detection and project name
    enriched: list[DiscoveredTranscript] = []
    for path in all_paths:
        # Detect format type
        format_type = pm.hook.agentgit_detect_format(path=path)
        if not format_type:
            format_type = "unknown"

        # Create plugin name from format: "claude_code_jsonl" -> "Claude Code"
        plugin_key = format_type.replace("_jsonl", "")
        plugin_name = plugin_key.replace("_", " ").title()

        # Get project name and display name from plugins
        project_name = pm.hook.agentgit_get_project_name(transcript_path=path)
        display_name = pm.hook.agentgit_get_display_name(transcript_path=path)

        # Get last timestamp from transcript content (if available)
        # Falls back to file modification time if plugin doesn't provide one
        last_timestamp = pm.hook.agentgit_get_last_timestamp(transcript_path=path)
        if last_timestamp is None:
            last_timestamp = path.stat().st_mtime

        stat = path.stat()
        enriched.append(
            DiscoveredTranscript(
                path=path,
                format_type=format_type,
                plugin_name=plugin_name,
                mtime=last_timestamp,
                size_bytes=stat.st_size,
                project_name=project_name,
                display_name=display_name,
            )
        )

    # Sort by modification time, most recent first
    enriched.sort(key=lambda t: t.mtime, reverse=True)
    return enriched
