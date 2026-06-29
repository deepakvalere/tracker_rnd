from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = UTILS_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent

DEFAULT_MODEL = PROJECT_ROOT / "models" / "player.pt"
DEFAULT_TRACKER = PROJECT_ROOT / "configs" / "botsort_reid.yaml"
DEFAULT_SOURCE = PROJECT_ROOT / "inputs" / "vid1.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLIPS_OUTPUT_DIR = PROJECT_ROOT / "vid_clips_extracted"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_outputs_dir(output_dir: str | Path | None = None) -> Path:
    out = resolve_path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
