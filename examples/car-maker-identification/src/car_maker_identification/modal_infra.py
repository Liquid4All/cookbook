import modal


def get_modal_app(name: str) -> modal.App:
    """
    Returns the Modal application object.
    """
    return modal.App(name)


def get_docker_image() -> modal.Image:
    """
    Returns a Modal Docker image with all the required Python dependencies installed.
    """
    docker_image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "datasets>=4.1.1",
            "modal>=1.1.4",
            "outlines>=1.2.7",
            "peft>=0.15.2",
            "pydantic-settings>=2.10.1",
            "tqdm>=4.67.1",
            "transformers==4.57.1",
            "trl==0.24.0",
            "pillow>=11.3.0",
            "matplotlib>=3.10.6",
            "torchao>=0.4.0",
            "wandb>=0.22.2",
            "torchvision==0.23.0",
            "bitsandbytes",
            "seaborn",
            "scikit-learn",
        )
        # .add_local_python_source(".")
        .env({"HF_HOME": "/model_cache"})
    )

    return docker_image


def get_volume(name: str) -> modal.Volume:
    """
    Returns a Modal volume object for the given name.
    """
    return modal.Volume.from_name(name, create_if_missing=True)


def get_retries(max_retries: int) -> modal.Retries:
    """
    Returns the retry policy for failed tasks.
    """
    return modal.Retries(initial_delay=0.0, max_retries=max_retries)


def get_secrets() -> list[modal.Secret]:
    """
    Returns the Weights & Biases secret.
    """
    wandb_secret = modal.Secret.from_name("wandb-secret")
    return [wandb_secret]
