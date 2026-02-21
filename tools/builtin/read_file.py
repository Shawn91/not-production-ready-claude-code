from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import is_binary_file, resolve_path
from utils.text import count_tokens, truncate_text


class ReadFileParams(BaseModel):
    path: str = Field(
        ...,
        description="The path to the file to read (relative to working directory or absolute path)",
    )
    offset: int = Field(
        1,
        ge=1,
        description="Line number to start reading from (1-based). Defaults to 1.",
    )
    limit: int | None = Field(
        None,
        ge=1,
        description="Maximum number of lines to read. If not provided, all lines from the offset will be read.",
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Return the file content with line numbers. "
        "For large files, use offset and limit to read only a portion of the file. "
        "Cannot read binary files (images, executables, etc.)."
    )
    kind = ToolKind.READ
    MAX_FILE_SIZE = 1024 * 1024 * 10  # 10 MB
    MAX_TOKEN_OUTPUT = 25000

    @property
    def schema(self):
        return ReadFileParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)
        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.error_result(f"Not a file: {path}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(
                f"File too large ({file_size / 1024 / 1024:.1f} MB): {path}"
            )

        if is_binary_file(path):
            return ToolResult.error_result(f"Cannot read binary files: {path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.error_result(
                f"File contains invalid UTF-8 characters: {path}"
            )
        try:
            lines = content.splitlines()
            if not lines:
                return ToolResult.success_result("File is empty", metadata={"lines": 0})
            if params.limit is not None:
                end_line_idx = params.offset + params.limit - 1
            else:
                end_line_idx = len(lines)
            selected_lines = lines[params.offset - 1 : end_line_idx]

            # 将行号写在每一行开头
            formatted_lines = []
            for i, line in enumerate(selected_lines, start=params.offset - 1):
                formatted_lines.append(f"{i:6}|{line}")

            output = "\n".join(formatted_lines)
            token_count = count_tokens(output)
            need_truncation = token_count > self.MAX_TOKEN_OUTPUT
            if need_truncation:
                output = truncate_text(
                    text=output,
                    max_tokens=self.MAX_TOKEN_OUTPUT,
                    suffix=f"\n... [truncated. Total line count {len(lines)}]",
                )

            metadata_lines = []
            if params.offset > 1 or end_line_idx < len(lines):
                metadata_lines.append(
                    f"SHowing lines {params.offset}-{end_line_idx} of {len(lines)}"
                )
            if metadata_lines:
                header = " | ".join(metadata_lines) + "\n\n"
                output = header + output
            return ToolResult.success_result(
                output,
                truncated=need_truncation,
                metadata={
                    "total_lines": len(lines),
                    "path": str(path),
                    "shown_start": params.offset,
                    "shown_end": end_line_idx,
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to read file: {e}")
