"""Core dataclasses and interfaces for agentgit."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class OperationType(str, Enum):
    """Types of file operations."""

    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"


@dataclass
class Prompt:
    """Represents a user prompt in a transcript.

    The prompt_id is computed as md5(text) for stable, content-based identification
    that persists across re-processing of the same transcript.
    """

    text: str
    timestamp: str
    raw_entry: dict[str, Any] = field(default_factory=dict)

    _prompt_id: Optional[str] = field(default=None, repr=False)

    @property
    def prompt_id(self) -> str:
        """MD5 hash of the prompt text for stable identification."""
        if self._prompt_id is None:
            self._prompt_id = hashlib.md5(self.text.encode("utf-8")).hexdigest()
        return self._prompt_id

    @property
    def short_id(self) -> str:
        """First 8 characters of the prompt_id for display."""
        return self.prompt_id[:8]


@dataclass
class AssistantContext:
    """Context from assistant messages preceding a file operation.

    Captures the reasoning/thinking that explains why a change was made,
    plus the explanation the assistant gave right after the change.
    """

    thinking: Optional[str] = None
    text: Optional[str] = None
    timestamp: str = ""
    # Explanation text produced by the assistant after the operation
    after: Optional[str] = None

    @property
    def summary(self) -> str:
        """Get a summary of the context for commit messages."""
        if self.thinking:
            return self.thinking
        if self.text:
            return self.text
        return ""


@dataclass
class FileOperation:
    """Represents a single file operation extracted from a transcript.

    Contains all metadata needed to create a rich git commit.
    """

    file_path: str
    operation_type: OperationType
    timestamp: str

    tool_id: str = ""

    # Content for operations
    content: Optional[str] = None  # For WRITE operations
    old_string: Optional[str] = None  # For EDIT operations
    new_string: Optional[str] = None  # For EDIT operations
    replace_all: bool = False  # For EDIT: replace all occurrences vs first
    recursive: bool = False  # For DELETE: recursive directory delete
    original_content: Optional[str] = None  # Pre-edit content from tool result

    # Rich metadata for commit messages
    prompt: Optional[Prompt] = None
    assistant_context: Optional[AssistantContext] = None

    # Raw data for extensibility
    raw_tool_use: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Extract just the filename from the path."""
        return self.file_path.rsplit("/", 1)[-1] if "/" in self.file_path else self.file_path


@dataclass
class TranscriptEntry:
    """A single entry from an agent transcript.

    Generic representation that plugins parse into.
    """

    entry_type: str  # "user", "assistant", "system", etc.
    timestamp: str
    message: dict[str, Any]
    raw_entry: dict[str, Any] = field(default_factory=dict)

    is_continuation: bool = False
    is_meta: bool = False  # e.g., skill expansions


@dataclass
class ConversationRound:
    """A user prompt and all entries until the next prompt.

    This becomes a feature branch in the git structure, with each entry
    becoming a commit on that branch, then merged back to the session branch.
    """

    prompt: Prompt
    entries: list[TranscriptEntry] = field(default_factory=list)
    sequence: int = 0  # 1-based sequence number for branch naming

    @property
    def short_summary(self) -> str:
        """Generate short summary for branch name (50 chars max)."""
        # Extract first line of prompt
        text = self.prompt.text.split('\n')[0].strip()
        # Sanitize for git branch (remove special chars, spaces to dashes)
        safe_text = re.sub(r'[^\w\s-]', '', text.lower())
        safe_text = re.sub(r'[\s_]+', '-', safe_text)
        return safe_text[:50].strip('-')

    @property
    def branch_name(self) -> str:
        """Full branch name: {sequence:03d}-{short-summary}"""
        return f"{self.sequence:03d}-{self.short_summary}"


@dataclass
class AssistantTurn:
    """A single assistant response containing grouped file operations.

    All operations in a turn are from the same assistant message and
    should be committed together as one logical change.
    """

    operations: list[FileOperation] = field(default_factory=list)
    context: Optional[AssistantContext] = None
    timestamp: str = ""
    raw_entry: dict[str, Any] = field(default_factory=dict)

    @property
    def files_modified(self) -> list[str]:
        """List of files modified in this turn."""
        return [op.filename for op in self.operations if op.operation_type == OperationType.EDIT]

    @property
    def files_created(self) -> list[str]:
        """List of files created in this turn."""
        return [op.filename for op in self.operations if op.operation_type == OperationType.WRITE]

    @property
    def files_deleted(self) -> list[str]:
        """List of files deleted in this turn."""
        return [op.filename for op in self.operations if op.operation_type == OperationType.DELETE]

    @property
    def summary_line(self) -> str:
        """Generate a summary line for commit message."""
        if self.context and self.context.text:
            # Use first line of assistant text as summary
            first_line = self.context.text.split("\n")[0].strip()
            if len(first_line) > 72:
                return first_line[:69] + "..."
            return first_line
        # Fall back to describing the operations
        if len(self.operations) == 1:
            op = self.operations[0]
            verb = {"write": "Create", "edit": "Edit", "delete": "Delete"}.get(
                op.operation_type.value, "Modify"
            )
            return f"{verb} {op.filename}"
        return f"Update {len(self.operations)} files"


@dataclass
class PromptResponse:
    """A user prompt and all assistant turns that responded to it.

    This represents one "unit of work" that becomes a merge commit.
    """

    prompt: Prompt
    turns: list[AssistantTurn] = field(default_factory=list)

    @property
    def all_operations(self) -> list[FileOperation]:
        """All operations across all turns for this prompt."""
        ops = []
        for turn in self.turns:
            ops.extend(turn.operations)
        return ops


@dataclass
class Transcript:
    """A complete parsed transcript with all entries and extracted operations."""

    entries: list[TranscriptEntry] = field(default_factory=list)
    prompts: list[Prompt] = field(default_factory=list)
    operations: list[FileOperation] = field(default_factory=list)

    # Grouped structures for git history
    prompt_responses: list[PromptResponse] = field(default_factory=list)  # Legacy
    conversation_rounds: list[ConversationRound] = field(default_factory=list)  # New

    source_path: Optional[str] = None
    source_format: str = ""

    session_id: Optional[str] = None
    session_cwd: Optional[str] = None

    def get_prompt_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Look up a prompt by its md5-based ID."""
        for prompt in self.prompts:
            if prompt.prompt_id == prompt_id or prompt.short_id == prompt_id:
                return prompt
        return None

    @property
    def all_turns(self) -> list[AssistantTurn]:
        """All assistant turns across all prompts."""
        turns = []
        for pr in self.prompt_responses:
            turns.extend(pr.turns)
        return turns


@dataclass
class SourceCommit:
    """A commit from the source repository."""

    sha: str
    message: str
    timestamp: str
    author: str
    author_email: str = ""
    files_changed: list[str] = field(default_factory=list)


@dataclass
class DiscoveredTranscript:
    """A transcript file discovered by plugins.

    Contains metadata about the transcript without fully parsing it.
    """

    path: Path
    format_type: str  # e.g., "claude_code_jsonl", "codex_jsonl"
    plugin_name: str  # Short name for display, e.g., "Claude Code", "Codex"
    mtime: float  # Modification time as timestamp
    size_bytes: int
    project_name: Optional[str] = None  # Project name from plugin
    display_name: Optional[str] = None  # Display name from plugin

    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        if self.size_bytes >= 1024 * 1024:
            return f"{self.size_bytes / (1024 * 1024):.1f} MB"
        elif self.size_bytes >= 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        return f"{self.size_bytes} B"

    @property
    def mtime_formatted(self) -> str:
        """Human-readable modification time."""
        from datetime import datetime

        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def name(self) -> str:
        """Get display name, falling back to filename."""
        return self.display_name or self.path.name
