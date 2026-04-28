from pathlib import Path


def get_path_to_configs() -> str:
    path = str(Path(__file__).parent.parent.parent / "configs")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_path_to_evals() -> str:
    path = str(Path(__file__).parent.parent.parent / "evals")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_path_model_checkpoints_in_modal_volume(experiment_name: str) -> str:
    """Returns path to the model checkpoints directory inside the Modal volume."""
    path = Path("/model_checkpoints") / experiment_name.replace("/", "--")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return str(path)
