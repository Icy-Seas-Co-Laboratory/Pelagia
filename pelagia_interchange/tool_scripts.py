from pathlib import Path

_ROOT = Path(__file__).with_name("bundled_tools")
TOOL_FILES = {name: (_ROOT / name).read_text(encoding="utf-8") for name in ("extract.py", "inspect.py", "verify.py", "README.md")}

