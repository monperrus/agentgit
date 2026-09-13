"""Git repository builder for agentgit."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from agentgit.core import (
    AssistantTurn,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
    SourceCommit,
)
from agentgit.plugins import get_configured_plugin_manager

if TYPE_CHECKING:
    from agentgit.enhance import EnhanceConfig

logger = logging.getLogger(__name__)


@dataclass
class CommitMetadata:
    """Metadata extracted from a commit message."""

    tool_id: str | None = None
    timestamp: str | None = None
    operation: str | None = None
    file_path: str | None = None
    prompt_id: str | None = None
    source_commit: str | None = None
    entry_id: str | None = None


def format_git_date(iso_timestamp: str) -> str:
    """Format ISO 8601 timestamp for git environment variables.

    Git expects dates in format: "YYYY-MM-DD HH:MM:SS +0000"

    Args:
        iso_timestamp: ISO 8601 timestamp (e.g., "2025-01-15T14:30:45Z")

    Returns:
        Git-formatted date string.
    """
    from datetime import datetime

    # Parse ISO 8601 timestamp
    # Handle both Z suffix and timezone offsets
    if iso_timestamp.endswith('Z'):
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime.fromisoformat(iso_timestamp)

    # Format for git: "YYYY-MM-DD HH:MM:SS +HHMM"
    # Use strftime with timezone
    return dt.strftime("%Y-%m-%d %H:%M:%S %z")


def parse_commit_trailers(message: str) -> CommitMetadata:
    """Parse git trailers from a commit message.

    Args:
        message: The full commit message.

    Returns:
        CommitMetadata with extracted values.
    """
    metadata = CommitMetadata()

    # Parse trailers (Key: Value format)
    trailer_patterns = {
        "tool_id": r"^Tool-Id:\s*(.+)$",
        "timestamp": r"^Timestamp:\s*(.+)$",
        "operation": r"^Operation:\s*(.+)$",
        "file_path": r"^File:\s*(.+)$",
        "prompt_id": r"^Prompt-Id:\s*(.+)$",
        "source_commit": r"^Source-Commit:\s*(.+)$",
        "entry_id": r"^Entry-Id:\s*(.+)$",
    }

    for field, pattern in trailer_patterns.items():
        match = re.search(pattern, message, re.MULTILINE)
        if match:
            setattr(metadata, field, match.group(1).strip())

    return metadata


def get_processed_operations(repo: Repo) -> set[str]:
    """Get the set of already-processed operation identifiers from a repo.

    Uses Tool-Id from commit trailers to identify processed operations.
    Falls back to Timestamp if Tool-Id is not available.

    Args:
        repo: The git repository to scan.

    Returns:
        Set of operation identifiers (tool_id or timestamp).
    """
    processed = set()

    try:
        for commit in repo.iter_commits():
            metadata = parse_commit_trailers(commit.message)
            if metadata.tool_id:
                processed.add(metadata.tool_id)
            elif metadata.timestamp:
                # Fallback to timestamp if no tool_id
                processed.add(f"ts:{metadata.timestamp}")
    except ValueError:
        # Empty repository has no commits
        pass
    except GitCommandError as e:
        logger.warning("Failed to read commits from repository: %s", e)

    return processed


def format_commit_message(operation: FileOperation) -> str:
    """Format a rich commit message for a file operation.

    Structure:
    - Subject line: operation type and file
    - Blank line
    - User prompt (the "why") - full text, no truncation
    - Blank line
    - Assistant context if available
    - Blank line
    - Git trailers for machine parsing

    Args:
        operation: The file operation to format.
    """
    op_verb = {
        OperationType.WRITE: "Create",
        OperationType.EDIT: "Edit",
        OperationType.DELETE: "Delete",
    }.get(operation.operation_type, "Modify")
    subject = f"{op_verb} {operation.filename}"

    body_parts = []

    # User prompt (the "why") - full text, no truncation
    if operation.prompt:
        body_parts.append(f"Prompt #{operation.prompt.short_id}:\n{operation.prompt.text}")

    # Assistant context (the reasoning)
    if operation.assistant_context and operation.assistant_context.summary:
        context = operation.assistant_context.summary
        if context:
            body_parts.append(f"Context:\n{context}")

    body = "\n\n".join(body_parts) if body_parts else ""

    # Trailers
    trailers = []
    if operation.prompt:
        trailers.append(f"Prompt-Id: {operation.prompt.prompt_id}")
    trailers.append(f"Operation: {operation.operation_type.value}")
    trailers.append(f"File: {operation.file_path}")
    trailers.append(f"Timestamp: {operation.timestamp}")
    if operation.tool_id:
        trailers.append(f"Tool-Id: {operation.tool_id}")

    trailers_str = "\n".join(trailers)

    if body:
        return f"{subject}\n\n{body}\n\n{trailers_str}"
    else:
        return f"{subject}\n\n{trailers_str}"


def format_turn_commit_message(
    turn: AssistantTurn,
    enhance_config: Optional["EnhanceConfig"] = None,
    prompt: Optional[Prompt] = None,
) -> str:
    """Format a commit message for an assistant turn (grouped operations).

    Structure:
    - Subject line: summary of what was done (or AI-generated)
    - Blank line
    - Files modified/created/deleted
    - Blank line
    - Context if available
    - Blank line
    - Trailers

    Args:
        turn: The assistant turn to format.
        enhance_config: Optional AI config for generating smarter commit messages.
        prompt: The prompt that triggered this turn (for including in empty commits).
    """
    # Try AI-generated subject if configured
    subject = None
    if enhance_config and enhance_config.enabled:
        from agentgit.enhance import generate_turn_summary

        # Get the prompt from the first operation if available, or use the provided prompt
        prompt_for_summary = prompt
        if turn.operations and turn.operations[0].prompt:
            prompt_for_summary = turn.operations[0].prompt
        subject = generate_turn_summary(turn, prompt_for_summary, enhance_config)

    # Fall back to default format
    if not subject:
        subject = turn.summary_line

    body_parts = []

    # List of files changed
    file_lists = []
    if turn.files_created:
        file_lists.append(f"Created: {', '.join(turn.files_created)}")
    if turn.files_modified:
        file_lists.append(f"Modified: {', '.join(turn.files_modified)}")
    if turn.files_deleted:
        file_lists.append(f"Deleted: {', '.join(turn.files_deleted)}")
    if file_lists:
        body_parts.append("\n".join(file_lists))

    # Assistant context (the reasoning) - use curated version if available
    context = None
    if enhance_config and enhance_config.enabled:
        from agentgit.enhance import curate_turn_context

        context = curate_turn_context(turn, enhance_config)

    # Fall back to raw context
    if context is None and turn.context and turn.context.summary:
        context = turn.context.summary

    if context:
        body_parts.append(f"Context:\n{context}")

    body = "\n\n".join(body_parts) if body_parts else ""

    # Trailers
    trailers = []
    # Get prompt from first operation or use the provided prompt
    prompt_for_trailer = None
    if turn.operations and turn.operations[0].prompt:
        prompt_for_trailer = turn.operations[0].prompt
    elif prompt:
        prompt_for_trailer = prompt

    if prompt_for_trailer:
        trailers.append(f"Prompt-Id: {prompt_for_trailer.prompt_id}")

    trailers.append(f"Timestamp: {turn.timestamp}")
    for op in turn.operations:
        if op.tool_id:
            trailers.append(f"Tool-Id: {op.tool_id}")

    trailers_str = "\n".join(trailers)

    if body:
        return f"{subject}\n\n{body}\n\n{trailers_str}"
    else:
        return f"{subject}\n\n{trailers_str}"


def format_prompt_merge_message(
    prompt: Prompt,
    turns: list[AssistantTurn],
    enhance_config: Optional["EnhanceConfig"] = None,
) -> str:
    """Format a merge commit message for a prompt.

    Structure:
    - Subject line: first line of prompt (truncated if needed) or AI-generated
    - Blank line
    - Full prompt text
    - Blank line
    - Trailers

    Args:
        prompt: The user prompt.
        turns: All assistant turns that responded to the prompt.
        enhance_config: Optional AI config for generating smarter commit messages.
    """
    # Try AI-generated subject if configured
    subject = None
    if enhance_config and enhance_config.enabled:
        from agentgit.enhance import generate_prompt_summary

        subject = generate_prompt_summary(prompt, turns, enhance_config)

    # Fall back to default format
    if not subject:
        first_line = prompt.text.split("\n")[0].strip()
        if len(first_line) > 72:
            subject = first_line[:69] + "..."
        else:
            subject = first_line

    body_parts = []

    # Full prompt text
    body_parts.append(f"Prompt #{prompt.short_id}:\n{prompt.text}")

    body = "\n\n".join(body_parts)

    # Trailers
    trailers = [f"Prompt-Id: {prompt.prompt_id}"]

    trailers_str = "\n".join(trailers)

    return f"{subject}\n\n{body}\n\n{trailers_str}"


def normalize_file_paths(operations: list[FileOperation]) -> tuple[str, dict[str, str]]:
    """Normalize file paths by finding common prefix.

    Returns:
        Tuple of (common_prefix, path_mapping) where path_mapping
        maps original paths to relative paths.
    """
    paths = [op.file_path for op in operations if op.file_path]
    if not paths:
        return "", {}

    # Deduplicate paths while preserving order
    unique_paths = list(dict.fromkeys(paths))

    # Normalize all paths to absolute to handle mix of absolute/relative
    abs_paths = [str(Path(p).resolve()) for p in unique_paths]

    if len(unique_paths) == 1:
        # Single unique file - use its parent as prefix
        common_prefix = str(Path(abs_paths[0]).parent)
    else:
        common_prefix = os.path.commonpath(abs_paths)
        # If common_prefix is a file path (not a directory), use its parent
        # This happens when one path is a prefix of another
        if not os.path.isdir(common_prefix) and common_prefix in abs_paths:
            common_prefix = str(Path(common_prefix).parent)

    path_mapping = {}
    for orig_path, abs_path in zip(unique_paths, abs_paths):
        rel_path = os.path.relpath(abs_path, common_prefix)
        path_mapping[orig_path] = rel_path

    return common_prefix, path_mapping


def get_commits_in_timeframe(
    repo_path: Path,
    start_time: str,
    end_time: str,
) -> list[SourceCommit]:
    """Get commits from a repository within a timeframe.

    Args:
        repo_path: Path to the git repository.
        start_time: ISO format start timestamp.
        end_time: ISO format end timestamp.

    Returns:
        List of SourceCommit objects.
    """
    repo = Repo(repo_path)
    commits = []

    for commit in repo.iter_commits():
        commit_time = commit.committed_datetime.isoformat()

        if commit_time < start_time:
            break
        if commit_time > end_time:
            continue

        files_changed = list(commit.stats.files.keys())

        commits.append(
            SourceCommit(
                sha=commit.hexsha,
                message=commit.message.strip(),
                timestamp=commit_time,
                author=commit.author.name,
                author_email=commit.author.email,
                files_changed=files_changed,
            )
        )

    commits.reverse()
    return commits


def merge_timeline(
    operations: list[FileOperation],
    source_commits: list[SourceCommit],
) -> list[Union[FileOperation, SourceCommit]]:
    """Merge operations and commits by timestamp."""
    combined: list[Union[FileOperation, SourceCommit]] = [*operations, *source_commits]
    return sorted(combined, key=lambda x: x.timestamp)


class GitRepoBuilder:
    """Builds a git repository from file operations."""

    def __init__(
        self,
        output_dir: Path | None = None,
        enhance_config: Optional["EnhanceConfig"] = None,
        code_repo: Path | None = None,
        session_branch_name: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize the builder.

        Args:
            output_dir: Directory for the git repo. If None, creates a temp dir.
            enhance_config: Optional AI configuration for generating commit messages.
            code_repo: Path to the code repository. If provided, sets up git alternates
                to share objects with the code repo. If None, auto-detects from cwd.
            session_branch_name: Optional branch name for session-based workflow. If provided,
                all commits go to this branch (never merged to main).
            session_id: Optional session identifier to include in commit trailers.
        """
        self.output_dir = output_dir
        self.enhance_config = enhance_config
        self.code_repo = code_repo
        self.session_branch_name = session_branch_name
        self.session_id = session_id
        self.repo: Repo | None = None
        self.path_mapping: dict[str, str] = {}
        self.file_states: dict[str, str] = {}
        self._processed_ops: set[str] = set()

    def _get_code_repo_path(self) -> Path | None:
        """Get the code repository path.

        Returns:
            Path to code repo, or None if not found/not a git repo.
        """
        if self.code_repo:
            return self.code_repo

        # Auto-detect from current directory
        from agentgit import find_git_root

        return find_git_root()

    def _get_code_repo_author(self) -> tuple[str, str]:
        """Get the author name and email from the code repository.

        Returns:
            Tuple of (name, email) from code repo git config, or defaults.
        """
        code_repo_path = self._get_code_repo_path()
        if not code_repo_path:
            return ("User", "user@local")

        try:
            code_repo = Repo(code_repo_path)
            reader = code_repo.config_reader()
            name = reader.get_value("user", "name", default="User")
            email = reader.get_value("user", "email", default="user@local")
            return (name, email)
        except Exception:
            return ("User", "user@local")

    def _setup_alternates(self, code_repo: Path) -> None:
        """Set up git alternates to share objects with code repo.

        Args:
            code_repo: Path to the code repository.
        """
        if not self.repo:
            return

        # Path to code repo's objects directory
        code_objects = code_repo / ".git" / "objects"
        if not code_objects.exists():
            logger.warning(
                "Code repo objects directory not found: %s", code_objects
            )
            return

        # Create alternates file
        alternates_file = self.output_dir / ".git" / "objects" / "info" / "alternates"
        alternates_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the path to code repo's objects
        alternates_file.write_text(str(code_objects) + "\n")
        logger.info("Set up git alternates to share objects with: %s", code_repo)

    def _is_operation_processed(self, operation: FileOperation) -> bool:
        """Check if an operation has already been processed."""
        if operation.tool_id and operation.tool_id in self._processed_ops:
            return True
        if operation.timestamp and f"ts:{operation.timestamp}" in self._processed_ops:
            return True
        return False

    def _get_operation_id(self, operation: FileOperation) -> str:
        """Get the identifier for an operation."""
        if operation.tool_id:
            return operation.tool_id
        return f"ts:{operation.timestamp}"

    def _setup_repo(self, incremental: bool) -> bool:
        """Initialize or load the git repository.

        Args:
            incremental: If True and repo exists, load processed operations.

        Returns:
            True if existing repo was found, False if new repo was created.
        """
        if self.output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix="agentgit_"))
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.repo = Repo(self.output_dir)
            if incremental:
                self._processed_ops = get_processed_operations(self.repo)
                self._load_file_states_from_repo()
            return True
        except InvalidGitRepositoryError:
            self.repo = Repo.init(self.output_dir)

            # Set up git alternates to share objects with code repo
            code_repo_path = self._get_code_repo_path()
            if code_repo_path:
                self._setup_alternates(code_repo_path)

            # Disable commit signing for agentgit-created repos
            with self.repo.config_writer() as config:
                config.set_value("commit", "gpgsign", "false")
                # Store code repo path for reference
                if code_repo_path:
                    config.set_value("agentgit", "coderepo", str(code_repo_path))

            return False

    def _get_merged_timeline(
        self,
        operations: list[FileOperation],
        source_repo: Path | None,
    ) -> list[Union[FileOperation, SourceCommit]]:
        """Merge operations with source repo commits by timestamp.

        Args:
            operations: List of operations to include.
            source_repo: Optional source repository to include commits from.

        Returns:
            Combined list of operations and commits sorted by timestamp.
        """
        if not source_repo or not operations:
            return list(operations)

        start_time = min(op.timestamp for op in operations)
        end_time = max(op.timestamp for op in operations)
        source_commits = get_commits_in_timeframe(source_repo, start_time, end_time)
        timeline = merge_timeline(operations, source_commits)
        timeline.sort(key=lambda x: x.timestamp)
        return timeline

    def build(
        self,
        operations: list[FileOperation],
        author_name: str = "Agent",
        author_email: str = "agent@local",
        source_repo: Path | None = None,
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build a git repository from file operations.

        If the output directory already contains a git repository and
        incremental=True, only new operations will be added.

        Args:
            operations: List of FileOperation objects in chronological order.
            author_name: Name for git commits.
            author_email: Email for git commits.
            source_repo: Optional source repository to interleave commits from.
            incremental: If True, skip already-processed operations.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        existing_repo = self._setup_repo(incremental)

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        non_delete_ops = [op for op in operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(non_delete_ops)

        # Filter out already-processed operations if incremental
        if incremental and existing_repo:
            operations = [op for op in operations if not self._is_operation_processed(op)]

        if not operations:
            return self.repo, self.output_dir, self.path_mapping

        timeline = self._get_merged_timeline(operations, source_repo)

        for item in timeline:
            if isinstance(item, FileOperation):
                self._apply_operation(item)
                self._processed_ops.add(self._get_operation_id(item))
            elif isinstance(item, SourceCommit):
                self._apply_source_commit(item, source_repo)

        return self.repo, self.output_dir, self.path_mapping

    def build_from_prompt_responses(
        self,
        prompt_responses: list[PromptResponse],
        author_name: str = "Agent",
        author_email: str = "agent@local",
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build a git repository using merge-based structure.

        If session_branch_name is set, all commits go to that branch (never merged).
        Otherwise, each prompt becomes a merge commit on main with feature branches.

        Args:
            prompt_responses: List of PromptResponse objects.
            author_name: Name for git commits.
            author_email: Email for git commits.
            incremental: If True, skip already-processed operations.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        existing_repo = self._setup_repo(incremental)

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        # Collect all operations for path normalization
        all_operations = []
        for pr in prompt_responses:
            for turn in pr.turns:
                all_operations.extend(turn.operations)

        non_delete_ops = [op for op in all_operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(non_delete_ops)

        if not prompt_responses:
            return self.repo, self.output_dir, self.path_mapping

        # Ensure we have an initial commit on main
        if not self._has_commits():
            self._create_initial_commit()

        # SESSION MODE: Create or checkout the session branch
        if self.session_branch_name:
            if self.session_branch_name in self.repo.heads:
                # Branch exists, check it out
                self.repo.heads[self.session_branch_name].checkout()
            else:
                # Create new session branch from main
                main_ref = self.repo.head.commit
                session_branch = self.repo.create_head(self.session_branch_name, main_ref)
                session_branch.checkout()
                logger.info("Created session branch: %s", self.session_branch_name)

        # Pre-process batch enhancement for AI enhancers (much more efficient)
        # Note: If already done globally, the cache will be hit and this is fast
        if self.enhance_config and self.enhance_config.enabled:
            from agentgit.enhance import preprocess_batch_enhancement

            preprocess_batch_enhancement(prompt_responses, self.enhance_config)

        for pr in prompt_responses:
            self._process_prompt_response(pr, incremental)

        # Create a session ref if session_id is provided
        # This creates a symbolic ref pointing to the session branch
        if self.session_id and self.session_branch_name:
            session_ref_path = self.output_dir / ".git" / "refs" / "sessions" / self.session_id
            session_ref_path.parent.mkdir(parents=True, exist_ok=True)
            # Write a symbolic ref pointing to the branch
            session_ref_path.write_text(f"ref: refs/heads/{self.session_branch_name}\n")
            logger.info("Created session ref: %s -> %s", f"refs/sessions/{self.session_id}", self.session_branch_name)

        return self.repo, self.output_dir, self.path_mapping

    def build_from_conversation_rounds(
        self,
        conversation_rounds: list["ConversationRound"],
        transcript: "Transcript",
        author_name: str = "Agent",
        author_email: str = "agent@local",
        incremental: bool = True,
    ) -> tuple[Repo, Path, dict[str, str]]:
        """Build git repo with conversational structure.

        Each conversation round becomes:
        1. A feature branch: {sequence:03d}-{summary}
        2. One commit per entry on that branch
        3. A merge commit back to session branch (authored by the human user)

        Args:
            conversation_rounds: List of ConversationRound objects.
            transcript: The full transcript for operation lookup.
            author_name: Name for agent commits.
            author_email: Email for agent commits.
            incremental: If True, skip already-processed entries.

        Returns:
            Tuple of (repo, repo_path, path_mapping).
        """
        existing_repo = self._setup_repo(incremental)

        # Get author info from plugin if available
        pm = get_configured_plugin_manager()
        plugin_author_info = pm.hook.agentgit_get_author_info(transcript=transcript)

        if plugin_author_info:
            author_name = plugin_author_info.get("name", author_name)
            author_email = plugin_author_info.get("email", author_email)

        # Store author info for agent commits
        self._agent_author_name = author_name
        self._agent_author_email = author_email

        # Get user author info from code repo for merge commits
        self._user_author_name, self._user_author_email = self._get_code_repo_author()

        with self.repo.config_writer() as config:
            config.set_value("user", "name", author_name)
            config.set_value("user", "email", author_email)

        # Collect all operations for path normalization
        all_operations = [op for op in transcript.operations if op.operation_type != OperationType.DELETE]
        _, self.path_mapping = normalize_file_paths(all_operations)

        if not conversation_rounds:
            return self.repo, self.output_dir, self.path_mapping

        # Ensure we have an initial commit on main
        if not self._has_commits():
            self._create_initial_commit()

        # Create or checkout session branch
        if self.session_branch_name:
            if self.session_branch_name in self.repo.heads:
                self.repo.heads[self.session_branch_name].checkout()
            else:
                main_ref = self.repo.head.commit
                session_branch = self.repo.create_head(self.session_branch_name, main_ref)
                session_branch.checkout()
                logger.info("Created session branch: %s", self.session_branch_name)

        # Process each conversation round
        for round in conversation_rounds:
            if incremental and self._is_round_processed(round):
                continue
            self._process_conversation_round(round, transcript)

        # Create session ref if session_id is provided
        if self.session_id and self.session_branch_name:
            session_ref_path = self.output_dir / ".git" / "refs" / "sessions" / self.session_id
            session_ref_path.parent.mkdir(parents=True, exist_ok=True)
            session_ref_path.write_text(f"ref: refs/heads/{self.session_branch_name}\n")
            logger.info("Created session ref: %s -> %s", f"refs/sessions/{self.session_id}", self.session_branch_name)

        return self.repo, self.output_dir, self.path_mapping

    def _has_commits(self) -> bool:
        """Check if the repository has any commits."""
        try:
            _ = self.repo.head.commit
            return True
        except ValueError:
            return False

    def _create_initial_commit(self, timestamp: str = "1970-01-01T00:00:00Z") -> None:
        """Create an initial empty commit.

        Args:
            timestamp: ISO 8601 timestamp for the commit. Defaults to epoch.
        """
        self._commit_with_date("Initial commit", timestamp, allow_empty=True)

    def _process_prompt_response(
        self, pr: PromptResponse, incremental: bool
    ) -> None:
        """Process a single prompt response.

        If session_branch_name is set, all commits go directly to the session branch.
        Otherwise, creates a temporary feature branch and merges to main.
        """
        if not pr.turns:
            return

        # Filter turns to only those with unprocessed operations OR no operations at all
        # (we want to preserve conversational turns that don't modify files)
        turns_to_process = []
        for turn in pr.turns:
            if not turn.operations:
                # Turn with no operations - preserve it as an empty commit
                turns_to_process.append(turn)
            else:
                # Turn with operations - check if any are unprocessed
                has_unprocessed = False
                for op in turn.operations:
                    if not self._is_operation_processed(op):
                        has_unprocessed = True
                        break
                if has_unprocessed:
                    turns_to_process.append(turn)

        if not turns_to_process:
            return

        # SESSION MODE: All commits go directly to the session branch, no merging
        if self.session_branch_name:
            # Apply each turn as a commit on the session branch
            for turn in turns_to_process:
                self._apply_turn(turn, pr.prompt)
                # Mark all operations in this turn as processed
                for op in turn.operations:
                    self._processed_ops.add(self._get_operation_id(op))
            return

        # LEGACY MODE: Feature branches that merge to main
        # Remember the current main branch position
        main_ref = self.repo.head.commit

        # Create a feature branch for this prompt
        branch_name = f"prompt-{pr.prompt.short_id}"
        feature_branch = self.repo.create_head(branch_name, main_ref)
        feature_branch.checkout()

        # Apply each turn as a commit on the feature branch
        for turn in turns_to_process:
            self._apply_turn(turn, pr.prompt)
            # Mark all operations in this turn as processed
            for op in turn.operations:
                self._processed_ops.add(self._get_operation_id(op))

        # Switch back to main
        self.repo.heads.main.checkout() if "main" in self.repo.heads else self.repo.heads.master.checkout()

        # Merge the feature branch with a merge commit using the prompt timestamp
        merge_message = format_prompt_merge_message(pr.prompt, pr.turns)
        git_date = format_git_date(pr.prompt.timestamp)
        old_author_date = os.environ.get('GIT_AUTHOR_DATE')
        old_committer_date = os.environ.get('GIT_COMMITTER_DATE')
        try:
            os.environ['GIT_AUTHOR_DATE'] = git_date
            os.environ['GIT_COMMITTER_DATE'] = git_date
            try:
                self.repo.git.merge(branch_name, "--no-ff", "-m", merge_message)
            except GitCommandError as e:
                logger.warning("Merge failed for prompt %s: %s", pr.prompt.short_id, e)
                # Try to continue despite merge issues
                try:
                    self.repo.git.merge("--abort")
                except GitCommandError:
                    pass
        finally:
            # Restore old env vars
            if old_author_date is not None:
                os.environ['GIT_AUTHOR_DATE'] = old_author_date
            else:
                os.environ.pop('GIT_AUTHOR_DATE', None)
            if old_committer_date is not None:
                os.environ['GIT_COMMITTER_DATE'] = old_committer_date
            else:
                os.environ.pop('GIT_COMMITTER_DATE', None)

        # Delete the feature branch
        try:
            self.repo.delete_head(branch_name, force=True)
        except GitCommandError:
            pass

    def _apply_turn(self, turn: AssistantTurn, prompt: Optional[Prompt] = None) -> None:
        """Apply all operations in a turn and create a single commit.

        Args:
            turn: The assistant turn to apply.
            prompt: The prompt that triggered this turn (for empty commits).
        """
        files_changed = []
        files_deleted = []

        # Apply all file operations if there are any
        for operation in turn.operations:
            changed = self._apply_operation_no_commit(operation)
            if changed:
                rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
                if operation.operation_type == OperationType.DELETE:
                    files_deleted.append(rel_path)
                else:
                    files_changed.append(rel_path)

        # Create commit message
        commit_msg = format_turn_commit_message(turn, self.enhance_config, prompt)

        if files_changed or files_deleted:
            # Add changed files (not deleted ones)
            for rel_path in files_changed:
                try:
                    self.repo.index.add([rel_path])
                except GitCommandError:
                    pass

            # Create regular commit with file changes
            try:
                self._commit_with_date(commit_msg, turn.timestamp)
            except GitCommandError as e:
                logger.warning("Failed to commit turn: %s", e)
        else:
            # No file changes - create an empty commit to preserve the conversation
            try:
                self._commit_with_date(commit_msg, turn.timestamp, allow_empty=True)
            except GitCommandError as e:
                logger.warning("Failed to create empty commit for turn: %s", e)

    def _apply_operation_no_commit(self, operation: FileOperation) -> bool:
        """Apply a single operation without creating a commit.

        Returns:
            True if file was changed, False otherwise.
        """
        if operation.operation_type == OperationType.DELETE:
            return self._apply_delete_no_commit(operation)
        elif operation.operation_type == OperationType.WRITE:
            return self._apply_write_no_commit(operation)
        elif operation.operation_type == OperationType.EDIT:
            return self._apply_edit_no_commit(operation)
        return False

    def _apply_write_no_commit(self, operation: FileOperation) -> bool:
        """Apply a write operation without committing."""
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(operation.content or "")

        self.file_states[operation.file_path] = operation.content or ""
        return True

    def _ensure_initial_state_for_edit(self, operation: FileOperation) -> str | None:
        """Ensure file exists and return its content for editing.

        Args:
            operation: The edit operation.

        Returns:
            File content if available, None if file cannot be prepared.
        """
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        if full_path.exists():
            return full_path.read_text()
        elif operation.file_path in self.file_states:
            return self.file_states[operation.file_path]
        elif operation.original_content:
            # Create initial state commit with the original content
            content = operation.original_content
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            self.repo.index.add([rel_path])
            # Use the operation timestamp for the initial state commit
            self._commit_with_date("Initial state (pre-session)", operation.timestamp)
            return content
        else:
            return None

    def _apply_edit_no_commit(self, operation: FileOperation) -> bool:
        """Apply an edit operation without committing."""
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        full_path = self.output_dir / rel_path

        # Get or create initial content
        content = self._ensure_initial_state_for_edit(operation)
        if content is None:
            return False

        old_str = operation.old_string or ""
        new_str = operation.new_string or ""

        if old_str in content:
            if operation.replace_all:
                content = content.replace(old_str, new_str)
            else:
                content = content.replace(old_str, new_str, 1)

            full_path.write_text(content)
            self.file_states[operation.file_path] = content
            return True

        return False

    def _apply_delete_no_commit(self, operation: FileOperation) -> bool:
        """Apply a delete operation without committing."""
        delete_path = operation.file_path
        files_removed = False

        if operation.recursive:
            delete_prefix = delete_path.rstrip("/") + "/"
            for orig_path, rel_path in self.path_mapping.items():
                if orig_path.startswith(delete_prefix) or orig_path == delete_path:
                    file_abs = self.output_dir / rel_path
                    if file_abs.exists():
                        file_abs.unlink()
                        try:
                            self.repo.index.remove([rel_path])
                        except GitCommandError:
                            pass
                        files_removed = True
        else:
            if delete_path in self.path_mapping:
                rel_path = self.path_mapping[delete_path]
                file_abs = self.output_dir / rel_path
                if file_abs.exists():
                    file_abs.unlink()
                    try:
                        self.repo.index.remove([rel_path])
                    except GitCommandError:
                        pass
                    files_removed = True

        return files_removed

    def _commit_with_date(self, commit_msg: str, timestamp: str, allow_empty: bool = False) -> None:
        """Create a commit with a specific timestamp.

        Args:
            commit_msg: The commit message.
            timestamp: ISO 8601 timestamp to use for commit date.
            allow_empty: Whether to allow empty commits.
        """
        git_date = format_git_date(timestamp)
        old_author_date = os.environ.get('GIT_AUTHOR_DATE')
        old_committer_date = os.environ.get('GIT_COMMITTER_DATE')
        try:
            os.environ['GIT_AUTHOR_DATE'] = git_date
            os.environ['GIT_COMMITTER_DATE'] = git_date
            if allow_empty:
                self.repo.git.commit("--allow-empty", "-m", commit_msg)
            else:
                self.repo.index.commit(commit_msg)
        finally:
            # Restore old env vars
            if old_author_date is not None:
                os.environ['GIT_AUTHOR_DATE'] = old_author_date
            else:
                os.environ.pop('GIT_AUTHOR_DATE', None)
            if old_committer_date is not None:
                os.environ['GIT_COMMITTER_DATE'] = old_committer_date
            else:
                os.environ.pop('GIT_COMMITTER_DATE', None)

    def _load_file_states_from_repo(self) -> None:
        """Load current file states from existing repo."""
        if not self.repo or self.repo.bare:
            return

        try:
            tree = self.repo.head.commit.tree
        except ValueError:
            # Empty repository has no HEAD commit
            return

        for item in tree.traverse():
            if item.type == "blob":
                rel_path = item.path
                full_path = self.output_dir / rel_path
                if full_path.exists():
                    try:
                        self.file_states[str(full_path)] = full_path.read_text()
                    except (OSError, UnicodeDecodeError) as e:
                        logger.debug("Could not read file %s: %s", full_path, e)

    def _apply_operation(self, operation: FileOperation) -> None:
        """Apply a single operation and create a commit."""
        if operation.operation_type == OperationType.DELETE:
            self._apply_delete(operation)
        elif operation.operation_type == OperationType.WRITE:
            self._apply_write(operation)
        elif operation.operation_type == OperationType.EDIT:
            self._apply_edit(operation)

    def _apply_write(self, operation: FileOperation) -> None:
        """Apply a write operation."""
        # Delegate to no_commit version for file operations
        if not self._apply_write_no_commit(operation):
            return

        # Stage and commit the changes
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        self.repo.index.add([rel_path])
        commit_msg = format_commit_message(operation)
        self._commit_with_date(commit_msg, operation.timestamp)

    def _apply_edit(self, operation: FileOperation) -> None:
        """Apply an edit operation."""
        # Delegate to no_commit version for file operations
        if not self._apply_edit_no_commit(operation):
            return

        # Stage and commit the changes
        rel_path = self.path_mapping.get(operation.file_path, operation.file_path)
        self.repo.index.add([rel_path])
        commit_msg = format_commit_message(operation)
        self._commit_with_date(commit_msg, operation.timestamp)

    def _apply_delete(self, operation: FileOperation) -> None:
        """Apply a delete operation."""
        # Delegate to no_commit version for file operations
        if not self._apply_delete_no_commit(operation):
            return

        # Create commit for the deletion
        commit_msg = format_commit_message(operation)
        try:
            self._commit_with_date(commit_msg, operation.timestamp)
        except GitCommandError as e:
            # May fail if nothing to commit
            logger.debug("Could not commit delete operation: %s", e)

    def _apply_source_commit(
        self,
        commit: SourceCommit,
        source_repo: Path | None,
    ) -> None:
        """Apply a commit from the source repository."""
        if not source_repo:
            return

        source = Repo(source_repo)
        source_commit = source.commit(commit.sha)

        # Get the diff and apply changes
        if source_commit.parents:
            parent = source_commit.parents[0]
            diffs = parent.diff(source_commit)
        else:
            diffs = source_commit.diff(None)

        for diff in diffs:
            if diff.a_path:
                rel_path = diff.a_path
                full_path = self.output_dir / rel_path

                if diff.deleted_file:
                    if full_path.exists():
                        full_path.unlink()
                        try:
                            self.repo.index.remove([rel_path])
                        except GitCommandError:
                            # File may not be tracked in git
                            logger.debug("Could not remove %s from index", rel_path)
                elif diff.new_file or diff.a_blob:
                    if diff.b_blob:
                        content = diff.b_blob.data_stream.read().decode("utf-8", errors="replace")
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        full_path.write_text(content)
                        self.repo.index.add([rel_path])

        # Create commit with original metadata and timestamp
        message = f"{commit.message}\n\nSource-Commit: {commit.sha}\nSource-Author: {commit.author} <{commit.author_email}>"
        try:
            self._commit_with_date(message, commit.timestamp)
        except GitCommandError as e:
            # May fail if nothing to commit
            logger.debug("Could not commit source commit %s: %s", commit.sha[:8], e)

    # Conversation round processing methods

    def _process_conversation_round(
        self,
        round: "ConversationRound",
        transcript: "Transcript",
    ) -> None:
        """Process a single conversation round: branch, commits, merge."""
        # Remember current session branch position
        session_ref = self.repo.head.commit

        # Create prompt branch
        try:
            prompt_branch = self.repo.create_head(round.branch_name, session_ref)
            prompt_branch.checkout()
        except GitCommandError as e:
            logger.warning("Failed to create prompt branch %s: %s", round.branch_name, e)
            return

        # Create commits for each entry (skip the user prompt - it's in the merge commit)
        for entry in round.entries:
            # Skip user entries - they're already represented in the merge commit
            if entry.entry_type == "user" and not entry.is_meta:
                continue
            self._commit_entry(entry, round, transcript)

        # Switch back to session branch
        if self.session_branch_name and self.session_branch_name in self.repo.heads:
            self.repo.heads[self.session_branch_name].checkout()
        else:
            try:
                self.repo.heads.main.checkout()
            except AttributeError:
                # No main branch, stay where we are
                pass

        # Merge the prompt branch (authored by the human user, not the agent)
        merge_message = self._format_round_merge_message(round)

        # Temporarily set user author for the merge commit
        original_name = self.repo.config_reader().get_value("user", "name")
        original_email = self.repo.config_reader().get_value("user", "email")

        # Set timestamp for merge commit using the prompt timestamp
        git_date = format_git_date(round.prompt.timestamp)
        old_author_date = os.environ.get('GIT_AUTHOR_DATE')
        old_committer_date = os.environ.get('GIT_COMMITTER_DATE')

        try:
            with self.repo.config_writer() as config:
                config.set_value("user", "name", self._user_author_name)
                config.set_value("user", "email", self._user_author_email)

            os.environ['GIT_AUTHOR_DATE'] = git_date
            os.environ['GIT_COMMITTER_DATE'] = git_date

            try:
                self.repo.git.merge(round.branch_name, "--no-ff", "-m", merge_message)
            except GitCommandError as e:
                logger.warning("Merge failed for round %s: %s", round.sequence, e)
                # Try to recover
                try:
                    self.repo.git.merge("--abort")
                except GitCommandError:
                    pass
        finally:
            # Restore agent author for subsequent commits
            with self.repo.config_writer() as config:
                config.set_value("user", "name", original_name)
                config.set_value("user", "email", original_email)

            # Restore old env vars
            if old_author_date is not None:
                os.environ['GIT_AUTHOR_DATE'] = old_author_date
            else:
                os.environ.pop('GIT_AUTHOR_DATE', None)
            if old_committer_date is not None:
                os.environ['GIT_COMMITTER_DATE'] = old_committer_date
            else:
                os.environ.pop('GIT_COMMITTER_DATE', None)

        # Delete the prompt branch
        try:
            self.repo.delete_head(round.branch_name, force=True)
        except GitCommandError:
            pass

    def _commit_entry(
        self,
        entry: "TranscriptEntry",
        round: "ConversationRound",
        transcript: "Transcript",
    ) -> None:
        """Create a commit for a single transcript entry."""
        # Find any file operations for this entry
        operations = self._find_operations_for_entry(entry, transcript)

        # Apply file operations if any
        files_changed = []
        for op in operations:
            self._current_operation = op
            if self._apply_operation_no_commit(op):
                rel_path = self.path_mapping.get(op.file_path, op.file_path)
                files_changed.append(rel_path)
        self._current_operation = operations[-1] if operations else None

        # Generate entry ID for tracking
        entry_id = self._get_entry_id(entry)

        # Create commit message
        commit_msg = self._format_entry_commit_message(entry, round, operations)
        commit_msg = self._apply_subject_rules(commit_msg)

        # Create commit (may be empty if no file changes)
        if files_changed:
            for rel_path in files_changed:
                try:
                    self.repo.index.add([rel_path])
                except GitCommandError:
                    pass
            try:
                self._commit_with_date(commit_msg, entry.timestamp)
                # Mark as processed
                if not hasattr(self, '_processed_entries'):
                    self._processed_entries = set()
                self._processed_entries.add(entry_id)
            except GitCommandError as e:
                logger.warning("Failed to commit entry: %s", e)
        else:
            # Empty commit to preserve conversation
            try:
                self._commit_with_date(commit_msg, entry.timestamp, allow_empty=True)
                # Mark as processed
                if not hasattr(self, '_processed_entries'):
                    self._processed_entries = set()
                self._processed_entries.add(entry_id)
            except GitCommandError as e:
                logger.warning("Failed to create empty commit: %s", e)

    def _find_operations_for_entry(
        self,
        entry: "TranscriptEntry",
        transcript: "Transcript",
    ) -> list["FileOperation"]:
        """Find file operations associated with an entry."""
        operations = []

        # For assistant entries, look for tool_use blocks and match with operations
        if entry.entry_type == "assistant":
            content = entry.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        if tool_id:
                            # Find operation with this tool_id
                            for op in transcript.operations:
                                if op.tool_id == tool_id:
                                    operations.append(op)

        return operations

    def _get_entry_id(self, entry: "TranscriptEntry") -> str:
        """Generate stable ID for a transcript entry."""
        import hashlib
        import json

        content_str = json.dumps(entry.message, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]
        return f"{entry.timestamp}:{entry.entry_type}:{content_hash}"

    def _is_round_processed(self, round: "ConversationRound") -> bool:
        """Check if all entries in a round have been processed."""
        if not hasattr(self, '_processed_entries'):
            # Load from existing commits
            self._processed_entries = self._get_processed_entries()

        for entry in round.entries:
            # Skip user entries - they're not committed individually
            if entry.entry_type == "user" and not entry.is_meta:
                continue
            entry_id = self._get_entry_id(entry)
            if entry_id not in self._processed_entries:
                return False
        return True

    def _get_processed_entries(self) -> set[str]:
        """Extract Entry-Id values from existing commits."""
        processed = set()

        if not self.repo or not self._has_commits():
            return processed

        try:
            for commit in self.repo.iter_commits():
                # Parse commit message for Entry-Id trailer
                metadata = parse_commit_trailers(commit.message)
                if metadata.entry_id:
                    processed.add(metadata.entry_id)
        except GitCommandError:
            pass

        return processed

    # Markdown formatters

    def _format_entry_commit_message(
        self,
        entry: "TranscriptEntry",
        round: "ConversationRound",
        operations: list["FileOperation"],
    ) -> str:
        """Format commit message for a transcript entry."""
        if entry.entry_type == "user":
            return self._format_user_entry(entry, round)
        elif entry.entry_type == "assistant":
            return self._format_assistant_entry(entry, round, operations)
        else:
            # Generic format for other entry types
            return self._format_generic_entry(entry, round)

    def _format_user_entry(
        self,
        entry: "TranscriptEntry",
        round: "ConversationRound",
    ) -> str:
        """Format user prompt entry."""
        text = self._extract_text_from_content(entry.message.get("content", ""))

        # First line as subject
        first_line = text.split('\n')[0].strip()
        if len(first_line) > 72:
            subject = first_line[:69] + "..."
        else:
            subject = first_line or "User prompt"

        # Build body
        body = text

        # Trailers
        entry_id = self._get_entry_id(entry)
        trailers = [
            "---",
            "",
            f"Entry-Id: {entry_id}",
            f"Timestamp: {entry.timestamp}",
            f"Prompt-Id: {round.prompt.prompt_id}",
        ]

        trailers_str = "\n".join(trailers)
        return f"{subject}\n\n{body}\n\n{trailers_str}"

    def _format_assistant_entry(
        self,
        entry: "TranscriptEntry",
        round: "ConversationRound",
        operations: list["FileOperation"],
    ) -> str:
        """Format assistant response entry with tool calls."""
        import json

        content = entry.message.get("content", [])

        # Extract blocks
        text_blocks = []
        thinking_blocks = []
        tool_calls = []

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text_blocks.append(block.get("text", ""))
                    elif block_type == "thinking":
                        thinking_blocks.append(block.get("thinking", ""))
                    elif block_type == "tool_use":
                        tool_calls.append(block)

        # Generate subject
        if text_blocks:
            first_text = text_blocks[0].split('\n')[0].strip()
            subject = first_text[:72] if len(first_text) > 72 else first_text
        elif tool_calls:
            if len(tool_calls) == 1:
                subject = f"Call {tool_calls[0].get('name', 'tool')}"
            else:
                subject = f"Call {len(tool_calls)} tools"
        elif thinking_blocks:
            first_thinking = thinking_blocks[0].split('\n')[0].strip()
            if first_thinking:
                subject = first_thinking[:72] if len(first_thinking) > 72 else first_thinking
            else:
                subject = "Assistant response"
        else:
            subject = "Assistant response"

        # Build body
        body_parts = []

        if text_blocks:
            body_parts.append("\n\n".join(text_blocks))

        if thinking_blocks:
            thinking_md = "## Extended Thinking\n\n" + "\n\n".join(thinking_blocks)
            body_parts.append(thinking_md)

        # Only include tool call details if this commit has no file operations
        # (the diff shows what happened for commits with code changes)
        if tool_calls and not operations:
            tools_md = "## Tool Calls\n\n"
            for call in tool_calls:
                tool_name = call.get("name", "unknown")
                tool_id = call.get("id", "")
                tool_input = call.get("input", {})
                input_json = json.dumps(tool_input, indent=2)
                tools_md += f"- **{tool_name}** (`{tool_id}`)\n  ```json\n  {input_json}\n  ```\n\n"
            body_parts.append(tools_md)

        body = "\n\n".join(body_parts) if body_parts else ""

        # Trailers
        entry_id = self._get_entry_id(entry)
        trailers = [
            "---",
            "",
            f"Entry-Id: {entry_id}",
            f"Timestamp: {entry.timestamp}",
        ]
        for call in tool_calls:
            tool_id = call.get("id")
            if tool_id:
                trailers.append(f"Tool-Id: {tool_id}")

        trailers_str = "\n".join(trailers)
        return f"{subject}\n\n{body}\n\n{trailers_str}"

    def _apply_subject_rules(self, message: str) -> str:
        """Post-process a commit message subject for readability.

        Turns mechanical placeholders into informative subjects:
        - ``Call Edit`` / ``Call Write`` / ``Call Bash`` (single tool entries)
          become ``Edit <filename>``, using assistant context or the
          operation's prompt as a fallback description.
        - Empty or whitespace-only subjects fall back to the first line of
          the body.

        Messages whose subject already carries information are returned
        unchanged.
        """
        import re as re_module

        # Split subject from the rest (subject = up to first blank line)
        parts = message.split("\n\n", 1)
        subject = parts[0].strip()
        rest = parts[1] if len(parts) > 1 else ""

        m = re_module.match(r"^Call (\w+)$", subject)
        if m:
            tool = m.group(1)
            # Filename of the first file operation in this commit, if any:
            # the trailers carry Tool-Id lines; the diff itself is not
            # available here, so we look the message's file list via callers.
            replacement = self._subject_for_tool_entry(tool, rest)
            if replacement:
                subject = replacement

        if not subject and rest:
            first_line = next(
                (ln.strip() for ln in rest.splitlines() if ln.strip()), ""
            )
            if first_line and not first_line.startswith("#"):
                subject = first_line[:72]

        if not rest:
            return subject
        return f"{subject}\n\n{rest}"

    def _subject_for_tool_entry(self, tool: str, rest: str) -> str | None:
        """Build a better subject for a 'Call <Tool>' entry.

        Prefers the current operation's file (Write/Edit/NotebookEdit), then
        the assistant context summary, then the Bash command description.
        """
        op = getattr(self, "_current_operation", None)
        if op is not None:
            verb = {"write": "Add", "edit": "Update", "delete": "Remove"}.get(
                str(op.operation_type).rsplit(".", 1)[-1].lower()
                if hasattr(op.operation_type, "name")
                else str(op.operation_type).lower(),
                None,
            )
            if verb:
                return f"{verb} {op.filename}"
        if tool == "Bash":
            # Use the command's description field when present in the body
            import re as re_module

            m = re_module.search(r'"description":\s*"([^"]+)"', rest)
            if m:
                desc = m.group(1)[:72]
                return f"Bash: {desc}"
        return None

    def _format_generic_entry(
        self,
        entry: "TranscriptEntry",
        round: "ConversationRound",
    ) -> str:
        """Format generic entry (system, tool results, etc.)."""
        import json

        # Try to extract meaningful subject
        subject = f"{entry.entry_type.title()} message"

        # Try to get content as text
        content = entry.message.get("content", "")
        if isinstance(content, str):
            body = content
        else:
            body = json.dumps(content, indent=2)

        # Trailers
        entry_id = self._get_entry_id(entry)
        trailers = [
            "---",
            "",
            f"Entry-Id: {entry_id}",
            f"Timestamp: {entry.timestamp}",
            f"Type: {entry.entry_type}",
        ]

        trailers_str = "\n".join(trailers)
        return f"{subject}\n\n{body}\n\n{trailers_str}"

    def _format_round_merge_message(
        self,
        round: "ConversationRound",
    ) -> str:
        """Format merge commit message for a conversation round."""
        # Subject: first line of the prompt (truncated), prefixed by the
        # sequence number so the timeline stays ordered
        first_line = round.prompt.text.split("\n")[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        subject = f"#{round.sequence}: {first_line}" if first_line else f"User Prompt #{round.sequence}"
        # Body: Full prompt text (not truncated)
        body = round.prompt.text
        # Trailers
        trailers = [
            "---",
            "",
            f"Prompt-Id: {round.prompt.prompt_id}",
            f"Round-Sequence: {round.sequence}",
        ]

        trailers_str = "\n".join(trailers)
        return f"{subject}\n\n{body}\n\n{trailers_str}"

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from message content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(text_parts)
        return ""
