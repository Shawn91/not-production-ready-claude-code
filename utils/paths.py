from pathlib import Path


def resolve_path(base: str | Path, path: str | Path):
    """将给定的路径（可能是相对，可能是绝对路径）解析为绝对路径"""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base).resolve() / path


def is_binary_file(path: Path):
    """判断给定的文件是否为二进制文件"""
    try:
        with open(path, "rb") as file:
            return b"\x00" in file.read(8192)
    except (OSError, IOError):
        return False


def display_path_rel_to_cwd(path: str, cwd: str | Path | None) -> str:
    try:
        p = Path(path)
    except Exception:
        return path

    if cwd:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            pass

    return str(p)
