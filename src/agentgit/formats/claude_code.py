"""Claude Code JSONL format parser plugin for agentgit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentgit.core import (
    AssistantContext,
    AssistantTurn,
    ConversationRound,
    FileOperation,
    OperationType,
    Prompt,
    PromptResponse,
    Transcript,
    TranscriptEntry,
)
from agentgit.plugins import hookimpl
from agentgit.utils import extract_deleted_paths

# Format identifier for Claude Code JSONL transcripts
FORMAT_CLAUDE_CODE_JSONL = "claude_code_jsonl"


def get_last_timestamp_from_jsonl(file_path: Path) -> float | None:
    """Extract the last timestamp from a JSONL file.

    Reads the last 8KB of the file to find the most recent entry's timestamp.
    This is more efficient than reading the entire file and more accurate than
    using file modification time.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Unix timestamp (seconds since epoch) of the last entry, or None if not found.
    """
    try:
        with open(file_path, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            if file_size == 0:
                return None

            # Read last 8KB (or entire file if smaller)
            chunk_size = min(8192, file_size)
            f.seek(file_size - chunk_size)
            chunk = f.read(chunk_size).decode('utf-8', errors='ignore')

            # Split into lines and process in reverse
            # Skip first line as it's likely truncated (unless we read the whole file)
            lines = chunk.split('\n')
            if chunk_size < file_size and len(lines) > 1:
                lines = lines[1:]  # Discard potentially truncated first line

            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    timestamp_str = obj.get("timestamp")
                    if timestamp_str:
                        # Parse ISO timestamp and convert to unix timestamp
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        return dt.timestamp()
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        pass

    return None


def _reconstruct_from_structured_patch(patch: Any) -> str | None:
    """Reconstruct pre-edit content from a Claude Code structuredPatch.

    The patch only covers the changed hunks with limited context, so the
    result is partial — but it lets an Edit on a never-Read file produce a
    best-effort baseline instead of being silently skipped. Lines are
    renumbered from oldStart to preserve approximate positions; missing
    ranges show up as gaps, which is more honest than inventing content.
    """
    if not isinstance(patch, list) or not patch:
        return None
    lines: list[str | None] = []

    def ensure(length: int) -> None:
        while len(lines) < length:
            lines.append(None)

    for hunk in patch:
        if not isinstance(hunk, dict):
            continue
        old_start = hunk.get("oldStart", 1) or 1
        hunk_lines = hunk.get("lines", [])
        ensure(old_start - 1)
        # Running old-file line number for this hunk; '+' lines don't
        # consume an old line, everything else does.
        old_line = old_start - 1
        for raw in hunk_lines:
            marker = raw[0] if raw else " "
            text = raw[1:] if marker in "+- " else raw
            if marker in "- ":
                # Pre-edit content: removed and context lines
                ensure(old_line + 1)
                lines[old_line] = text
            old_line += 0 if marker == "+" else 1
    if not any(l is not None for l in lines):
        return None
    return "\n".join(l if l is not None else "" for l in lines)


def _is_contentless_entry(entry: Any) -> bool:
    """Check if an assistant entry carries no usable content.

    Streaming padding entries contain a single ``thinking`` block whose
    ``thinking`` text is empty (only the cryptographic signature is set).
    They contribute nothing to commits or conversation context.
    """
    content = entry.message.get("content", [])
    if not isinstance(content, list) or not content:
        return True
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking" and block.get("thinking", "").strip():
            return False
        if block_type == "text" and block.get("text", "").strip():
            return False
        if block_type == "tool_use":
            return False
    return True


def _is_pure_tool_result_entry(entry: Any) -> bool:
    """Check if a user entry only carries tool results (no human text).

    Claude Code feeds tool output back to the model as ``user`` entries whose
    content is a list of ``tool_result`` blocks. These are not human prompts
    and must not start a new conversation round — otherwise every assistant
    entry between a tool call and its result is dropped from the round.
    """
    content = entry.message.get("content", [])
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


class ClaudeCodePlugin:
    """Plugin for parsing Claude Code JSONL transcripts."""

    @hookimpl
    def agentgit_get_plugin_info(self) -> dict[str, str]:
        """Return plugin identification info."""
        return {
            "name": "claude_code",
            "description": "Claude Code JSONL transcripts",
        }

    @hookimpl
    def agentgit_detect_format(self, path: Path) -> str | None:
        """Detect Claude Code JSONL format."""
        if path.suffix != ".jsonl":
            return None

        try:
            with open(path, encoding="utf-8") as f:
                saw_user_or_assistant = False
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    # Skip non-interactive sessions (from claude-cli --print mode)
                    # These are created when tools like llm make API calls
                    if not saw_user_or_assistant and obj.get("type") == "queue-operation" and obj.get("operation") == "dequeue":
                        return None

                    if obj.get("type") in ("user", "assistant", "summary"):
                        return FORMAT_CLAUDE_CODE_JSONL

                    # Claude Code >= 2.1 sessions can open with many
                    # mode/permission-mode/file-history-snapshot lines before
                    # the first user/assistant entry; keep scanning instead of
                    # giving up after 5 lines.
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_parse_transcript(self, path: Path, format: str) -> Transcript | None:
        """Parse Claude Code JSONL transcript."""
        if format != FORMAT_CLAUDE_CODE_JSONL:
            return None

        entries = []
        prompts = []
        session_id = None
        session_cwd = None

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type")
                if entry_type not in ("user", "assistant"):
                    continue

                timestamp = obj.get("timestamp", "")
                message = obj.get("message", {})

                if not session_id:
                    session_id = obj.get("sessionId")
                if not session_cwd:
                    session_cwd = obj.get("cwd")

                entry = TranscriptEntry(
                    entry_type=entry_type,
                    timestamp=timestamp,
                    message=message,
                    raw_entry=obj,
                    is_continuation=obj.get("isCompactSummary", False),
                    is_meta=obj.get("isMeta", False),
                )
                # Skip assistant entries with no usable content: streaming
                # padding entries only carry an empty thinking block with a
                # signature. Committing them adds noise ("Assistant response"
                # empty commits) without information.
                if entry_type == "assistant" and _is_contentless_entry(entry):
                    continue
                entries.append(entry)

                if entry_type == "user" and not entry.is_meta:
                    text = self._extract_text_from_content(message.get("content", ""))
                    if text:
                        prompts.append(
                            Prompt(
                                text=text,
                                timestamp=timestamp,
                                raw_entry=obj,
                            )
                        )

        return Transcript(
            entries=entries,
            prompts=prompts,
            source_path=str(path),
            source_format=FORMAT_CLAUDE_CODE_JSONL,
            session_id=session_id,
            session_cwd=session_cwd,
        )

    @hookimpl
    def agentgit_extract_operations(self, transcript: Transcript) -> list[FileOperation]:
        """Extract file operations from Claude Code transcript."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return []

        operations = []

        # First pass: collect originalFile content from tool results
        tool_id_to_original: dict[str, str] = {}
        for entry in transcript.entries:
            tool_use_result = entry.raw_entry.get("toolUseResult", {})
            if tool_use_result and "originalFile" in tool_use_result:
                original = tool_use_result.get("originalFile")
                if not isinstance(original, str) or not original:
                    # originalFile can be None when the transcript was
                    # compacted/resumed; structuredPatch still carries the
                    # hunks (with context lines), the best available
                    # pre-edit reconstruction.
                    original = _reconstruct_from_structured_patch(
                        tool_use_result.get("structuredPatch")
                    )
                content = entry.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "")
                            if tool_id and isinstance(original, str) and original:
                                tool_id_to_original[tool_id] = original

        # Second pass: extract operations
        for entry in transcript.entries:
            if entry.entry_type != "assistant":
                continue

            content = entry.message.get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue

                tool_name = block.get("name", "")
                tool_id = block.get("id", "")
                tool_input = block.get("input", {})

                if tool_name == "Write":
                    file_path = tool_input.get("file_path", "")
                    if file_path:  # Skip if file_path is empty or missing
                        operations.append(
                            FileOperation(
                                file_path=file_path,
                                operation_type=OperationType.WRITE,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                content=tool_input.get("content", ""),
                                raw_tool_use=block,
                            )
                        )

                elif tool_name == "Edit":
                    file_path = tool_input.get("file_path", "")
                    if file_path:  # Skip if file_path is empty or missing
                        operations.append(
                            FileOperation(
                                file_path=file_path,
                                operation_type=OperationType.EDIT,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                old_string=tool_input.get("old_string", ""),
                                new_string=tool_input.get("new_string", ""),
                                replace_all=tool_input.get("replace_all", False),
                                original_content=tool_id_to_original.get(tool_id),
                                raw_tool_use=block,
                            )
                        )

                elif tool_name == "Bash":
                    command = tool_input.get("command", "")
                    deleted_paths = extract_deleted_paths(command)
                    is_recursive = "-r" in command

                    for path in deleted_paths:
                        operations.append(
                            FileOperation(
                                file_path=path,
                                operation_type=OperationType.DELETE,
                                timestamp=entry.timestamp,
                                tool_id=tool_id,
                                recursive=is_recursive,
                                raw_tool_use=block,
                            )
                        )

        operations.sort(key=lambda op: op.timestamp)
        return operations

    @hookimpl
    def agentgit_enrich_operation(
        self,
        operation: FileOperation,
        transcript: Transcript,
    ) -> FileOperation:
        """Enrich operation with prompt and assistant context."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return operation

        # Find the prompt that triggered this operation
        current_prompt = None
        for prompt in transcript.prompts:
            if prompt.timestamp <= operation.timestamp:
                current_prompt = prompt
            else:
                break

        if current_prompt:
            operation.prompt = current_prompt

        # Find assistant context (thinking/text) preceding this tool use.
        # A different tool_use in between does NOT invalidate the context:
        # agents routinely think, run a Bash probe, then edit a file — the
        # reasoning still explains the edit. Context is only bounded by the
        # start of the enclosing user turn (a real human prompt), so we keep
        # the most recent non-empty thinking/text seen since then.
        context = AssistantContext()
        for entry in transcript.entries:
            if entry.timestamp > operation.timestamp:
                break

            if entry.entry_type == "user" and not entry.is_meta:
                # A genuine human prompt (tool_result carriers are filtered
                # out) starts a new reasoning context.
                content = entry.message.get("content", [])
                is_tool_result_only = isinstance(content, list) and bool(content) and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in content
                )
                if not is_tool_result_only:
                    context = AssistantContext()

            elif entry.entry_type == "assistant":
                content = entry.message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking" and block.get("thinking"):
                                context.thinking = block.get("thinking", "")
                                context.timestamp = entry.timestamp
                            elif block.get("type") == "text" and block.get("text"):
                                context.text = block.get("text", "")
                                context.timestamp = entry.timestamp
                            elif block.get("type") == "tool_use":
                                if block.get("id") == operation.tool_id:
                                    if context.thinking or context.text:
                                        operation.assistant_context = context
                                    return operation

        return operation

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract plain text from message content."""
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts).strip()
        return ""

    def _add_context_if_needed(
        self, prompt: Prompt, assistant_texts: list[str]
    ) -> Prompt:
        """Add assistant context to a prompt if it appears to need it.

        Short prompts like "yes", "do it", or "let's do 2, 3, 4" need the
        preceding assistant text to make sense.
        """
        if not self._prompt_needs_context(prompt.text):
            return prompt

        # Combine assistant texts, most recent last
        context = "\n\n".join(assistant_texts)

        # Truncate if too long (keep last 1500 chars which likely has the question)
        if len(context) > 1500:
            context = "..." + context[-1500:]

        # Format with clear delimiter
        combined_text = f"[Assistant context:\n{context}]\n\n{prompt.text}"

        return Prompt(
            text=combined_text,
            timestamp=prompt.timestamp,
            raw_entry=prompt.raw_entry,
        )

    def _prompt_needs_context(self, text: str) -> bool:
        """Determine if a prompt is too short/referential to stand alone.

        Returns True if the prompt likely needs assistant context to make sense.
        """
        text = text.strip()

        # Very short prompts almost always need context
        if len(text) < 50:
            return True

        # Common affirmative/directive responses that need context
        contextual_starters = [
            "yes",
            "no",
            "ok",
            "okay",
            "sure",
            "do it",
            "go ahead",
            "let's do",
            "let's go",
            "sounds good",
            "that works",
            "perfect",
            "great",
            "please do",
            "go for it",
            "make it so",
            "proceed",
            "continue",
            "approved",
            "confirmed",
            "agreed",
            "correct",
            "right",
            "exactly",
            "yep",
            "yup",
            "nope",
            "skip",
            "ignore",
            "both",
            "neither",
            "all of",
            "none of",
            "the first",
            "the second",
            "the last",
            "option",
        ]

        text_lower = text.lower()
        for starter in contextual_starters:
            if text_lower.startswith(starter):
                return True

        # Check for numbered references like "1, 2, 3" or "2 and 4"
        # These typically reference a list from the assistant
        if re.search(r"\b\d+\s*(,|and|&|\+)\s*\d+", text_lower):
            return True

        # Check for "that", "this", "it", "those" at start - referential
        referential_starts = ["that", "this", "it ", "its ", "those", "these", "the "]
        for ref in referential_starts:
            if text_lower.startswith(ref):
                return True

        return False

    @hookimpl
    def agentgit_get_project_name(self, transcript_path: Path) -> str | None:
        """Get the project name from a Claude Code transcript.

        Reads the transcript to find the first cwd, then finds the git root
        (tracing upward if needed) and returns its directory name.
        """
        from agentgit import find_git_root

        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Check if this transcript is in ~/.claude/projects/
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        # Extract cwd from the transcript
        cwd = self._extract_cwd_from_transcript(transcript_path)
        if not cwd:
            return None

        # Find git root (may be cwd itself or a parent directory)
        project_root = find_git_root(cwd)
        return project_root.name if project_root else Path(cwd).name

    def _extract_cwd_from_transcript(self, transcript_path: Path) -> str | None:
        """Extract the first cwd from a Claude Code transcript."""
        try:
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        cwd = obj.get("cwd")
                        if cwd:
                            return cwd
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return None

    @hookimpl
    def agentgit_get_display_name(self, transcript_path: Path) -> str | None:
        """Get display name for a Claude Code transcript.

        Returns the filename as-is without truncation.
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Only handle Claude Code transcripts
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        return transcript_path.name

    @hookimpl
    def agentgit_get_last_timestamp(self, transcript_path: Path) -> float | None:
        """Get the last timestamp from a Claude Code transcript.

        Reads the end of the JSONL file to find the last entry's timestamp
        instead of using file modification time.
        """
        transcript_abs = transcript_path.resolve()
        claude_projects_dir = Path.home() / ".claude" / "projects"

        # Only handle Claude Code transcripts
        try:
            transcript_abs.relative_to(claude_projects_dir)
        except ValueError:
            return None

        return get_last_timestamp_from_jsonl(transcript_path)

    @hookimpl
    def agentgit_get_author_info(self, transcript: Transcript) -> dict[str, str] | None:
        """Get git author information for Claude Code transcripts.

        Returns Claude as the author. In the future, this could extract
        specific model information from the transcript if available.
        """
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return None

        # Default author for Claude Code transcripts
        # TODO: Extract actual model info from transcript when available
        return {
            "name": "Claude",
            "email": "claude@anthropic.com",
        }

    @hookimpl
    def agentgit_build_prompt_responses(
        self, transcript: Transcript
    ) -> list[PromptResponse]:
        """Build prompt-response structure grouping operations by assistant message."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return []

        # Build a mapping of tool_id -> operation for quick lookup
        tool_id_to_op: dict[str, FileOperation] = {}
        for op in transcript.operations:
            if op.tool_id:
                tool_id_to_op[op.tool_id] = op

        prompt_responses: list[PromptResponse] = []
        current_prompt: Prompt | None = None
        current_turns: list[AssistantTurn] = []
        # Track assistant text between user prompts for context
        pending_assistant_text: list[str] = []

        for entry in transcript.entries:
            # New user prompt starts a new PromptResponse
            if entry.entry_type == "user" and not entry.is_meta:
                # Find or create the Prompt object for this entry
                text = self._extract_text_from_content(entry.message.get("content", ""))
                new_prompt: Prompt | None = None
                for p in transcript.prompts:
                    if p.timestamp == entry.timestamp:
                        new_prompt = p
                        break
                if new_prompt is None and text:
                    new_prompt = Prompt(
                        text=text, timestamp=entry.timestamp, raw_entry=entry.raw_entry
                    )

                # Add assistant context if the prompt needs it
                if new_prompt and pending_assistant_text:
                    new_prompt = self._add_context_if_needed(
                        new_prompt, pending_assistant_text
                    )

                if current_prompt is not None:
                    if current_turns:
                        # Previous prompt had operations - save it and start fresh
                        prompt_responses.append(
                            PromptResponse(prompt=current_prompt, turns=current_turns)
                        )
                        current_prompt = new_prompt
                        current_turns = []
                        pending_assistant_text = []
                    elif new_prompt:
                        # No operations between prompts - concatenate them
                        combined_text = current_prompt.text + "\n\n" + new_prompt.text
                        current_prompt = Prompt(
                            text=combined_text,
                            timestamp=current_prompt.timestamp,  # Keep original timestamp
                            raw_entry=current_prompt.raw_entry,
                        )
                else:
                    current_prompt = new_prompt
                    current_turns = []
                    pending_assistant_text = []

            # Assistant message - track text and create turns
            elif entry.entry_type == "assistant" and current_prompt is not None:
                # Collect assistant text for potential context
                assistant_text = self._extract_text_from_content(
                    entry.message.get("content", [])
                )
                if assistant_text:
                    pending_assistant_text.append(assistant_text)

                turn = self._build_assistant_turn(entry, tool_id_to_op)
                if turn.operations:  # Only add turns that have file operations
                    current_turns.append(turn)
                    # Clear pending text after operations - context is less relevant
                    pending_assistant_text = []

        # Don't forget the last prompt response
        if current_prompt is not None and current_turns:
            prompt_responses.append(
                PromptResponse(prompt=current_prompt, turns=current_turns)
            )

        return prompt_responses

    @hookimpl
    def agentgit_build_conversation_rounds(
        self, transcript: Transcript
    ) -> list[ConversationRound]:
        """Build conversation rounds grouping ALL entries by user prompt."""
        if transcript.source_format != FORMAT_CLAUDE_CODE_JSONL:
            return []

        rounds: list[ConversationRound] = []
        current_prompt: Prompt | None = None
        current_entries: list[TranscriptEntry] = []
        sequence = 0

        def flush_round() -> None:
            nonlocal current_prompt, current_entries
            if current_prompt is not None and current_entries:
                rounds.append(ConversationRound(
                    prompt=current_prompt,
                    entries=current_entries,
                    sequence=sequence
                ))
            current_prompt = None
            current_entries = []

        for entry in transcript.entries:
            # Only genuine human prompts (with text, not tool results) start
            # a new round. User entries carrying pure tool_result content are
            # the harness feeding tool output back to the model — they belong
            # to the ongoing round, not a new one.
            if (
                entry.entry_type == "user"
                and not entry.is_meta
                and not _is_pure_tool_result_entry(entry)
            ):
                flush_round()

                # Start new round
                sequence += 1

                # Find or create the Prompt object for this entry
                text = self._extract_text_from_content(entry.message.get("content", ""))
                current_prompt = None
                for p in transcript.prompts:
                    if p.timestamp == entry.timestamp:
                        current_prompt = p
                        break

                if current_prompt is None and text:
                    # Create Prompt if not found in transcript.prompts
                    current_prompt = Prompt(
                        text=text,
                        timestamp=entry.timestamp,
                        raw_entry=entry.raw_entry
                    )

                # Initialize with this user entry
                current_entries = [entry]
            else:
                # Add all other entries to current round
                if current_prompt is not None:
                    current_entries.append(entry)

        # Don't forget the last round
        flush_round()

        return rounds

    def _build_assistant_turn(
        self, entry: TranscriptEntry, tool_id_to_op: dict[str, FileOperation]
    ) -> AssistantTurn:
        """Build an AssistantTurn from an assistant entry."""
        operations: list[FileOperation] = []
        context = AssistantContext()

        content = entry.message.get("content", [])
        if not isinstance(content, list):
            return AssistantTurn(timestamp=entry.timestamp, raw_entry=entry.raw_entry)

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "thinking":
                context.thinking = block.get("thinking", "")
                context.timestamp = entry.timestamp
            elif block_type == "text":
                context.text = block.get("text", "")
                context.timestamp = entry.timestamp
            elif block_type == "tool_use":
                tool_id = block.get("id", "")
                if tool_id in tool_id_to_op:
                    operations.append(tool_id_to_op[tool_id])

        return AssistantTurn(
            operations=operations,
            context=context if (context.thinking or context.text) else None,
            timestamp=entry.timestamp,
            raw_entry=entry.raw_entry,
        )

    @hookimpl
    def agentgit_discover_transcripts(
        self, project_path: Path | None
    ) -> list[Path]:
        """Discover Claude Code transcripts for a project.

        Looks in ~/.claude/projects/ for directories matching the project path.
        Claude Code stores projects with paths like:
        ~/.claude/projects/-Users-username-path-to-project/

        Args:
            project_path: Path to the project. If None, returns all transcripts
                from all projects.
        """
        claude_projects_dir = Path.home() / ".claude" / "projects"
        if not claude_projects_dir.exists():
            return []

        transcripts = []

        if project_path is None:
            # Return all transcripts from all projects
            for project_dir in claude_projects_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                for jsonl_file in project_dir.glob("*.jsonl"):
                    # Skip agent sub-transcripts (they start with "agent-")
                    if jsonl_file.name.startswith("agent-"):
                        continue
                    # Skip empty files (0 bytes)
                    if jsonl_file.stat().st_size == 0:
                        continue
                    # Skip non-interactive API call sessions (queue-operation dequeue)
                    if self._is_api_call_session(jsonl_file):
                        continue
                    transcripts.append(jsonl_file)
        else:
            # Convert project path to Claude's format: /path/to/project -> -path-to-project
            project_path = project_path.resolve()
            encoded_path = str(project_path).replace("/", "-")

            # Look for matching project directory
            project_dir = claude_projects_dir / encoded_path
            if not project_dir.exists():
                return []

            # Find all JSONL files in the project directory
            for jsonl_file in project_dir.glob("*.jsonl"):
                # Skip agent sub-transcripts (they start with "agent-")
                if jsonl_file.name.startswith("agent-"):
                    continue
                # Skip empty files (0 bytes)
                if jsonl_file.stat().st_size == 0:
                    continue
                # Skip non-interactive API call sessions (queue-operation dequeue)
                if self._is_api_call_session(jsonl_file):
                    continue
                transcripts.append(jsonl_file)

        # Sort by modification time, most recent first
        transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return transcripts

    def _is_api_call_session(self, path: Path) -> bool:
        """Check if a transcript file is a non-interactive API call session.

        These are created when tools like llm use claude-cli with --print mode.
        They start with a queue-operation dequeue entry.

        Args:
            path: Path to the transcript file.

        Returns:
            True if this is an API call session, False otherwise.
        """
        try:
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                obj = json.loads(first_line)
                return (
                    obj.get("type") == "queue-operation"
                    and obj.get("operation") == "dequeue"
                )
        except Exception:
            return False
