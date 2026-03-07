from pydantic import BaseModel, Field

from tools.base import (
    FileDiff,
    Tool,
    ToolConfirmation,
    ToolInvocation,
    ToolKind,
    ToolResult,
)
from utils.paths import ensure_parent_directory, resolve_path


class WriteFileParams(BaseModel):
    path: str = Field(
        ...,
        description="Path to the file to write (relative to working directory or absolute)",
    )
    content: str = Field(..., description="Content to write to the file")
    create_directories: bool = Field(
        True, description="Create parent directories if they don't exist"
    )


class WriteFileTool(Tool):
    _name = "write_file"
    _description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "or overwrites if it does. Parent directories are created automatically. "
        "Use this for creating new files or completely replacing file contents. "
        "For partial modifications, use the edit tool instead."
    )

    kind = ToolKind.WRITE

    @property
    def schema(self):
        return WriteFileParams

    async def get_confirmation(
        self, invocation: ToolInvocation
    ) -> ToolConfirmation | None:
        params = WriteFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        is_new_file = not path.exists()

        old_content = ""
        if not is_new_file:
            try:
                old_content = path.read_text(encoding="utf-8")
            except Exception:
                pass

        diff = FileDiff(
            path=path,
            old_content=old_content,
            new_content=params.content,
            is_new_file=is_new_file,
        )

        action = "Created" if is_new_file else "Updated"

        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"{action} file: {path}",
            diff=diff,
            affected_paths=[path],
            is_dangerous=not is_new_file,
        )

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WriteFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)
        old_content = ""  # 写入的文件的原始内容
        file_existed = path.exists()
        if file_existed:
            old_content = path.read_text(encoding="utf-8")

        try:
            if params.create_directories:
                ensure_parent_directory(path=path)
            elif not path.parent.exists():
                return ToolResult.error_result(
                    f"Parent directory does not exist: {path.parent}",
                )
            path.write_text(params.content, encoding="utf-8")

            # 当前是创建新文件，还是写入一个旧文件
            action = "Created" if not file_existed else "Updated"
            line_count = len(params.content.splitlines())

            return ToolResult.success_result(
                output=f"{action} {path} {line_count} lines",
                diff=FileDiff(
                    path=path,
                    old_content=old_content,
                    new_content=params.content,
                    is_new_file=not file_existed,
                ),
                metadata={
                    "path": str(path),
                    "is_new_file": not file_existed,
                    "lines": line_count,
                    "bytes": len(params.content.encode("utf-8")),
                },
            )
        except OSError as e:
            return ToolResult.error_result(f"Faild to write file: {e}")
