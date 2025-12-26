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
        .apt_install(
            "git",
        )          
        .uv_pip_install(
            "trl[vllm]",
            "git+https://github.com/meta-pytorch/OpenEnv.git@bf5e968286e0d49cdc03fd904d48faff4b15a437",
            "openenv_core==0.1.1",
            "liger-kernel",
            "wandb",
            "peft>=0.13.0",
        )
        .env({"HF_HOME": "/hf_model_cache"})
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