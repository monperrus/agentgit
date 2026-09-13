"""Tests for agentgit.formats.claude_code module."""

import json
import tempfile
from pathlib import Path

import pytest

from agentgit.core import OperationType, Transcript
from agentgit.formats.claude_code import ClaudeCodePlugin, get_last_timestamp_from_jsonl


@pytest.fixture
def plugin():
    """Create a ClaudeCodePlugin instance."""
    return ClaudeCodePlugin()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file."""
    content = [
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Create a hello world function"},
            "sessionId": "test-session",
            "cwd": "/test/project",
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll create a hello world function."},
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/project/hello.py",
                            "content": "def hello():\n    return 'Hello, World!'",
                        },
                    },
                ]
            },
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:10.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Now I'll add a greeting parameter."},
                    {
                        "type": "tool_use",
                        "id": "toolu_002",
                        "name": "Edit",
                        "input": {
                            "file_path": "/test/project/hello.py",
                            "old_string": "def hello():",
                            "new_string": "def hello(name='World'):",
                        },
                    },
                ]
            },
        },
    ]

    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


@pytest.fixture
def sample_jsonl_with_tool_results(tmp_path):
    """Create a sample JSONL file with tool_result entries between tool calls.

    Mirrors real Claude Code transcripts: tool output is fed back as ``user``
    entries carrying pure ``tool_result`` content blocks. A regression in
    round-building treated these as new human prompts, dropping every
    assistant entry between a tool call and its result.
    """
    content = [
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:00.000Z",
            "message": {"content": "Add a greeting module"},
            "sessionId": "test-session",
            "cwd": "/test/project",
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:05.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_001",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/project/greet.py",
                            "content": "def greet():\n    return 'hi'",
                        },
                    },
                ]
            },
        },
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:06.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_001",
                        "content": "File created successfully",
                    }
                ]
            },
            "toolUseResult": {
                "type": "text",
                "filePath": "/test/project/greet.py",
                "originalFile": "",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2025-01-01T10:00:10.000Z",
            "message": {
                "content": [
                    {"type": "text", "text": "Now add a farewell module."},
                    {
                        "type": "tool_use",
                        "id": "toolu_002",
                        "name": "Write",
                        "input": {
                            "file_path": "/test/project/bye.py",
                            "content": "def bye():\n    return 'bye'",
                        },
                    },
                ]
            },
        },
        {
            "type": "user",
            "timestamp": "2025-01-01T10:00:11.000Z",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_002",
                        "content": "File created successfully",
                    }
                ]
            },
            "toolUseResult": {
                "type": "text",
                "filePath": "/test/project/bye.py",
                "originalFile": "",
            },
        },
    ]

    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for line in content:
            f.write(json.dumps(line) + "\n")

    return jsonl_path


class TestClaudeCodePlugin:
    """Tests for ClaudeCodePlugin."""

    def test_detect_format_jsonl(self, plugin, sample_jsonl):
        """Should detect Claude Code JSONL format."""
        result = plugin.agentgit_detect_format(sample_jsonl)
        assert result == "claude_code_jsonl"

    def test_detect_format_non_jsonl(self, plugin, tmp_path):
        """Should return None for non-JSONL files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("just some text")
        result = plugin.agentgit_detect_format(txt_file)
        assert result is None

    def test_detect_format_invalid_jsonl(self, plugin, tmp_path):
        """Should return None for invalid JSONL."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text("not valid json\n")
        result = plugin.agentgit_detect_format(jsonl_file)
        assert result is None

    def test_parse_transcript(self, plugin, sample_jsonl):
        """Should parse transcript correctly."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")

        assert transcript is not None
        assert transcript.source_format == "claude_code_jsonl"
        assert transcript.session_id == "test-session"
        assert transcript.session_cwd == "/test/project"
        assert len(transcript.entries) == 3
        assert len(transcript.prompts) == 1
        assert transcript.prompts[0].text == "Create a hello world function"

    def test_parse_transcript_wrong_format(self, plugin, sample_jsonl):
        """Should return None for wrong format."""
        result = plugin.agentgit_parse_transcript(sample_jsonl, "other_format")
        assert result is None

    def test_extract_operations(self, plugin, sample_jsonl):
        """Should extract file operations."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        assert len(operations) == 2

        # First operation: Write
        assert operations[0].operation_type == OperationType.WRITE
        assert operations[0].file_path == "/test/project/hello.py"
        assert "Hello, World!" in operations[0].content
        assert operations[0].tool_id == "toolu_001"

        # Second operation: Edit
        assert operations[1].operation_type == OperationType.EDIT
        assert operations[1].file_path == "/test/project/hello.py"
        assert operations[1].old_string == "def hello():"
        assert operations[1].new_string == "def hello(name='World'):"
        assert operations[1].tool_id == "toolu_002"

    def test_extract_operations_wrong_format(self, plugin):
        """Should return empty list for wrong format."""
        transcript = Transcript(source_format="other_format")
        operations = plugin.agentgit_extract_operations(transcript)
        assert operations == []

    def test_extract_operations_skips_empty_file_path(self, plugin, tmp_path):
        """Should skip operations with empty or missing file_path."""
        content = [
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "",  # Empty path
                                "content": "should be skipped",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                # Missing file_path
                                "content": "should also be skipped",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_003",
                            "name": "Write",
                            "input": {
                                "file_path": "/valid/path.py",
                                "content": "valid",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        # Should only have the valid operation
        assert len(operations) == 1
        assert operations[0].file_path == "/valid/path.py"

    def test_enrich_operation(self, plugin, sample_jsonl):
        """Should enrich operations with prompt context."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")
        operations = plugin.agentgit_extract_operations(transcript)

        enriched = plugin.agentgit_enrich_operation(operations[0], transcript)

        assert enriched.prompt is not None
        assert enriched.prompt.text == "Create a hello world function"

    def test_extract_text_from_string_content(self, plugin):
        """Should extract text from string content."""
        text = plugin._extract_text_from_content("Hello world")
        assert text == "Hello world"

    def test_extract_text_from_array_content(self, plugin):
        """Should extract text from array content."""
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        text = plugin._extract_text_from_content(content)
        assert text == "First part Second part"

    def test_extract_text_ignores_non_text_blocks(self, plugin):
        """Should ignore non-text blocks."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "123"},
            {"type": "text", "text": "World"},
        ]
        text = plugin._extract_text_from_content(content)
        assert text == "Hello World"


class TestClaudeCodeProjectName:
    """Tests for Claude Code project name extraction from cwd."""

    def test_get_project_name_from_cwd(self, plugin, tmp_path, monkeypatch):
        """Should extract project name from cwd field in transcript."""
        # Create .claude/projects directory
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        # Create transcript with cwd field
        transcript = claude_project_dir / "session.jsonl"
        transcript.write_text(
            json.dumps({"type": "user", "cwd": "/tmp/myproject"}) + "\n"
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        # Should return directory name from cwd
        assert result == "myproject"

    def test_get_project_name_returns_none_for_non_claude_transcript(self, plugin, tmp_path):
        """Should return None for transcripts not in ~/.claude/projects/."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type": "user", "cwd": "/tmp/test"}\n')

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None

    def test_get_project_name_returns_none_without_cwd(self, plugin, tmp_path, monkeypatch):
        """Should return None if transcript has no cwd field."""
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        transcript = claude_project_dir / "session.jsonl"
        transcript.write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_project_name(transcript)
        assert result is None


class TestClaudeCodeDiscovery:
    """Tests for Claude Code transcript discovery."""

    def test_discover_returns_empty_if_no_claude_dir(self, plugin, tmp_path, monkeypatch):
        """Should return empty if ~/.claude doesn't exist."""
        # Point home to a temp directory without .claude
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_returns_empty_if_no_matching_project(self, plugin, tmp_path, monkeypatch):
        """Should return empty if no matching project directory."""
        # Create .claude/projects but no matching project
        claude_dir = tmp_path / ".claude" / "projects"
        claude_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(tmp_path / "some-project")
        assert result == []

    def test_discover_finds_transcripts(self, plugin, tmp_path, monkeypatch):
        """Should find transcript files in matching project directory."""
        # Create project directory
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        # Create .claude/projects with matching directory
        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create transcript files
        (claude_project_dir / "session1.jsonl").write_text('{"type": "user"}\n')
        (claude_project_dir / "session2.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 2
        assert all(p.suffix == ".jsonl" for p in result)

    def test_discover_excludes_agent_files(self, plugin, tmp_path, monkeypatch):
        """Should exclude agent-*.jsonl files."""
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create regular and agent transcript files
        (claude_project_dir / "session.jsonl").write_text('{"type": "user"}\n')
        (claude_project_dir / "agent-abc123.jsonl").write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 1
        assert result[0].name == "session.jsonl"

    def test_discover_sorts_by_mtime(self, plugin, tmp_path, monkeypatch):
        """Should sort results by modification time, most recent first."""
        import time

        project_path = tmp_path / "myproject"
        project_path.mkdir()

        encoded_path = str(project_path.resolve()).replace("/", "-")
        claude_project_dir = tmp_path / ".claude" / "projects" / encoded_path
        claude_project_dir.mkdir(parents=True)

        # Create files with different modification times
        old_file = claude_project_dir / "old.jsonl"
        old_file.write_text('{"type": "user"}\n')

        time.sleep(0.01)  # Ensure different mtime

        new_file = claude_project_dir / "new.jsonl"
        new_file.write_text('{"type": "user"}\n')

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_discover_transcripts(project_path)
        assert len(result) == 2
        assert result[0].name == "new.jsonl"  # Most recent first
        assert result[1].name == "old.jsonl"


class TestBuildPromptResponses:
    """Tests for agentgit_build_prompt_responses hook."""

    def test_concatenates_consecutive_prompts_without_operations(self, plugin, tmp_path):
        """Should concatenate prompts when no operations between them."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "First part of the request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {"content": [{"type": "text", "text": "Let me think..."}]},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Second part - please also add tests"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "I'll create the file."},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/hello.py",
                                "content": "def hello(): pass",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have only 1 prompt response with concatenated text
        assert len(prompt_responses) == 1
        assert "First part of the request" in prompt_responses[0].prompt.text
        assert "Second part - please also add tests" in prompt_responses[0].prompt.text
        # Should use the first prompt's timestamp
        assert prompt_responses[0].prompt.timestamp == "2025-01-01T10:00:00.000Z"

    def test_separate_prompts_when_operations_between(self, plugin, tmp_path):
        """Should keep prompts separate when there are operations between them."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "First request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/file1.py",
                                "content": "# file 1",
                            },
                        },
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Second request"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/file2.py",
                                "content": "# file 2",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have 2 separate prompt responses
        assert len(prompt_responses) == 2
        assert prompt_responses[0].prompt.text == "First request"
        assert prompt_responses[1].prompt.text == "Second request"

    def test_concatenates_multiple_consecutive_prompts(self, plugin, tmp_path):
        """Should concatenate more than two consecutive prompts."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Part one"},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {"content": "Part two"},
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "Part three"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/hello.py",
                                "content": "# code",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Should have only 1 prompt response with all three parts
        assert len(prompt_responses) == 1
        assert "Part one" in prompt_responses[0].prompt.text
        assert "Part two" in prompt_responses[0].prompt.text
        assert "Part three" in prompt_responses[0].prompt.text

    def test_adds_context_for_short_prompt(self, plugin, tmp_path):
        """Should add assistant context when user gives short response like 'yes'.

        When there are no operations between prompts, they get concatenated.
        The short follow-up prompt gets context added before concatenation.
        """
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Help me refactor this code"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "I can help refactor this. Should I also add unit tests?",
                        }
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "yes"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/code.py",
                                "content": "# refactored",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Since no operations between prompts, they are concatenated into one
        # The short "yes" prompt gets context added
        assert len(prompt_responses) == 1
        prompt_text = prompt_responses[0].prompt.text
        assert "Help me refactor this code" in prompt_text
        assert "[Assistant context:" in prompt_text
        assert "Should I also add unit tests?" in prompt_text
        assert "yes" in prompt_text

    def test_adds_context_for_numbered_selection(self, plugin, tmp_path):
        """Should add context when user selects numbered items like 'let's do 2, 3, 4'.

        When there are no operations between prompts, they get concatenated.
        The numbered selection gets context added before concatenation.
        """
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Review the code quality"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": """Here are my recommendations:
1. Split the large function
2. Remove duplicate imports
3. Add proper field instead of reusing
4. Consolidate blob reading
5. Extract repeated patterns""",
                        }
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {"content": "let's do 2, 3, 4"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Edit",
                            "input": {
                                "file_path": "/test/code.py",
                                "old_string": "import re",
                                "new_string": "",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # Since no operations between prompts, they are concatenated into one
        # The "let's do 2, 3, 4" gets context with the numbered list
        assert len(prompt_responses) == 1
        prompt_text = prompt_responses[0].prompt.text
        assert "Review the code quality" in prompt_text
        assert "[Assistant context:" in prompt_text
        assert "1. Split the large function" in prompt_text
        assert "2. Remove duplicate imports" in prompt_text
        assert "let's do 2, 3, 4" in prompt_text

    def test_no_context_for_detailed_prompt(self, plugin, tmp_path):
        """Should NOT add context when user gives a detailed, standalone prompt."""
        content = [
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "message": {"content": "Start a new project"},
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "What kind of project?"},
                        {
                            "type": "tool_use",
                            "id": "toolu_001",
                            "name": "Write",
                            "input": {"file_path": "/test/init.py", "content": ""},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "content": "Please create a Python web server using FastAPI with endpoints for user authentication including login, logout, and password reset functionality"
                },
            },
            {
                "type": "assistant",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_002",
                            "name": "Write",
                            "input": {
                                "file_path": "/test/server.py",
                                "content": "# server",
                            },
                        },
                    ]
                },
            },
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        transcript = plugin.agentgit_parse_transcript(jsonl_path, "claude_code_jsonl")
        ops = plugin.agentgit_extract_operations(transcript)
        transcript.operations = ops

        prompt_responses = plugin.agentgit_build_prompt_responses(transcript)

        # The detailed prompt should NOT have context added
        assert len(prompt_responses) == 2
        second_prompt = prompt_responses[1].prompt.text
        assert "[Assistant context:" not in second_prompt
        assert "FastAPI" in second_prompt


class TestPromptNeedsContext:
    """Tests for _prompt_needs_context helper."""

    def test_short_prompts_need_context(self, plugin):
        """Short prompts should need context."""
        assert plugin._prompt_needs_context("yes") is True
        assert plugin._prompt_needs_context("no") is True
        assert plugin._prompt_needs_context("ok do it") is True
        assert plugin._prompt_needs_context("sure, go ahead") is True

    def test_numbered_selections_need_context(self, plugin):
        """Prompts selecting numbered items need context."""
        assert plugin._prompt_needs_context("let's do 2, 3, 4") is True
        assert plugin._prompt_needs_context("do 1 and 3") is True
        assert plugin._prompt_needs_context("items 2 & 5") is True
        assert plugin._prompt_needs_context("just 1, 2") is True

    def test_referential_prompts_need_context(self, plugin):
        """Prompts that reference something need context."""
        assert plugin._prompt_needs_context("that sounds good") is True
        assert plugin._prompt_needs_context("this is what I want") is True
        assert plugin._prompt_needs_context("the first option") is True

    def test_detailed_prompts_dont_need_context(self, plugin):
        """Detailed standalone prompts don't need context."""
        assert (
            plugin._prompt_needs_context(
                "Create a new Python class called UserManager with methods for CRUD operations"
            )
            is False
        )
        assert (
            plugin._prompt_needs_context(
                "Please refactor the authentication module to use JWT tokens instead of sessions"
            )
            is False
        )


class TestGetLastTimestamp:
    """Tests for get_last_timestamp_from_jsonl helper function."""

    def test_extracts_last_timestamp(self, tmp_path):
        """Should extract the timestamp from the last entry."""
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": "first"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": "second"},
            {"type": "user", "timestamp": "2025-01-01T10:00:10.000Z", "message": "third"},
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the last timestamp (2025-01-01T10:00:10)
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:10.000+00:00").timestamp()
        assert result == expected

    def test_skips_entries_without_timestamp(self, tmp_path):
        """Should skip entries that don't have a timestamp field."""
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z", "message": "first"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z", "message": "second"},
            {"type": "other", "message": "no timestamp"},  # No timestamp
        ]

        jsonl_path = tmp_path / "session.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the second timestamp since last entry has none
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:05.000+00:00").timestamp()
        assert result == expected

    def test_handles_empty_file(self, tmp_path):
        """Should return None for empty file."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        result = get_last_timestamp_from_jsonl(jsonl_path)
        assert result is None

    def test_handles_truncated_first_line(self, tmp_path):
        """Should handle files larger than 8KB by skipping truncated first line."""
        # Create a file with many entries to exceed 8KB
        content = []
        for i in range(200):
            # Use i as seconds (0-59 valid), cycle through minutes
            minute = i // 60
            second = i % 60
            content.append({
                "type": "user",
                "timestamp": f"2025-01-01T10:{minute:02d}:{second:02d}.000Z",
                "message": f"message {i}" * 100,  # Make it large
            })

        jsonl_path = tmp_path / "large.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should still extract the last timestamp correctly
        from datetime import datetime
        # Last entry was i=199 -> minute=3, second=19
        expected = datetime.fromisoformat("2025-01-01T10:03:19.000+00:00").timestamp()
        assert result == expected

    def test_handles_invalid_json_lines(self, tmp_path):
        """Should skip invalid JSON lines and continue searching."""
        jsonl_path = tmp_path / "mixed.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"type": "assistant", "timestamp": "2025-01-01T10:00:05.000Z"}) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)

        # Should return the last valid timestamp
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:05.000+00:00").timestamp()
        assert result == expected

    def test_returns_none_if_no_timestamps(self, tmp_path):
        """Should return None if file has no entries with timestamps."""
        content = [
            {"type": "other", "message": "no timestamp"},
            {"type": "other", "data": "also no timestamp"},
        ]

        jsonl_path = tmp_path / "no_timestamps.jsonl"
        with open(jsonl_path, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        result = get_last_timestamp_from_jsonl(jsonl_path)
        assert result is None


class TestClaudeCodeGetLastTimestamp:
    """Tests for agentgit_get_last_timestamp hook."""

    def test_returns_timestamp_for_claude_code_transcript(self, plugin, tmp_path, monkeypatch):
        """Should return last timestamp for Claude Code transcripts."""
        # Create .claude/projects directory
        claude_project_dir = tmp_path / ".claude" / "projects" / "-tmp-myproject"
        claude_project_dir.mkdir(parents=True)

        # Create transcript with timestamps
        transcript = claude_project_dir / "session.jsonl"
        content = [
            {"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"},
            {"type": "assistant", "timestamp": "2025-01-01T10:00:10.000Z"},
        ]
        with open(transcript, "w") as f:
            for line in content:
                f.write(json.dumps(line) + "\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = plugin.agentgit_get_last_timestamp(transcript)

        # Should return the last timestamp
        from datetime import datetime
        expected = datetime.fromisoformat("2025-01-01T10:00:10.000+00:00").timestamp()
        assert result == expected

    def test_returns_none_for_non_claude_transcript(self, plugin, tmp_path):
        """Should return None for transcripts not in ~/.claude/projects/."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text('{"type": "user", "timestamp": "2025-01-01T10:00:00.000Z"}\n')

        result = plugin.agentgit_get_last_timestamp(transcript)
        assert result is None


class TestClaudeCodeAuthorInfo:
    """Tests for agentgit_get_author_info hook."""

    def test_returns_author_info_for_claude_code_transcript(self, plugin, sample_jsonl):
        """Should return Claude author info for Claude Code transcripts."""
        transcript = plugin.agentgit_parse_transcript(sample_jsonl, "claude_code_jsonl")

        result = plugin.agentgit_get_author_info(transcript)

        assert result is not None
        assert result["name"] == "Claude"
        assert result["email"] == "claude@anthropic.com"

    def test_returns_none_for_non_claude_transcript(self, plugin):
        """Should return None for non-Claude Code transcripts."""
        transcript = Transcript(source_format="other_format")

        result = plugin.agentgit_get_author_info(transcript)

        assert result is None


class TestConversationRoundsWithToolResults:
    """Tests for round building with tool_result user entries.

    Real Claude Code transcripts interleave assistant tool_use entries with
    user entries carrying tool_result content. These must stay inside the
    ongoing round, not start a new one.
    """

    def test_tool_result_entries_do_not_start_new_rounds(
        self, plugin, sample_jsonl_with_tool_results
    ):
        """All entries after a tool_result should stay in the same round."""
        transcript = plugin.agentgit_parse_transcript(
            sample_jsonl_with_tool_results, "claude_code_jsonl"
        )

        rounds = plugin.agentgit_build_conversation_rounds(transcript)

        assert len(rounds) == 1

    def test_assistant_entries_after_tool_result_are_kept(
        self, plugin, sample_jsonl_with_tool_results
    ):
        """Assistant Write/Edit after a tool_result must remain in the round."""
        transcript = plugin.agentgit_parse_transcript(
            sample_jsonl_with_tool_results, "claude_code_jsonl"
        )

        rounds = plugin.agentgit_build_conversation_rounds(transcript)

        assert len(rounds) == 1
        tool_use_ids_in_round = [
            block.get("id")
            for e in rounds[0].entries
            if isinstance(e.message.get("content"), list)
            for block in e.message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        assert "toolu_001" in tool_use_ids_in_round
        assert "toolu_002" in tool_use_ids_in_round

    def test_mixed_content_user_entry_with_text_starts_new_round(
        self, plugin, sample_jsonl_with_tool_results
    ):
        """A user entry with both tool_result and text blocks (e.g. an
        interruption with commentary) still counts as a human prompt."""
        import json as json_module

        with open(sample_jsonl_with_tool_results) as f:
            lines = [json_module.loads(line) for line in f]

        # Append a follow-up human message after the tool results
        lines.append(
            {
                "type": "user",
                "timestamp": "2025-01-01T10:01:00.000Z",
                "message": {"content": "Now add tests"},
            }
        )
        sample_jsonl_with_tool_results.write_text(
            "".join(json_module.dumps(line) + "\n" for line in lines)
        )

        transcript = plugin.agentgit_parse_transcript(
            sample_jsonl_with_tool_results, "claude_code_jsonl"
        )
        rounds = plugin.agentgit_build_conversation_rounds(transcript)

        assert len(rounds) == 2
        assert rounds[1].prompt.text == "Now add tests"
