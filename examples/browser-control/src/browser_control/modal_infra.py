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
            "curl",
            "ca-certificates",
            "gnupg",
            "lsb-release",
            "supervisor",
        )
        .run_commands(
            # Install Docker
            "curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
            'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null',
            "apt-get update",
            "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin",
            # Create Docker startup script
            "mkdir -p /usr/local/bin",
            'echo "#!/bin/bash\ndockerd --host=unix:///var/run/docker.sock --host=tcp://0.0.0.0:2376 --storage-driver=overlay2 &\nsleep 5\n" > /usr/local/bin/start-docker.sh',
            "chmod +x /usr/local/bin/start-docker.sh",
        )           
        .uv_pip_install(
            "datasets>=4.4.1",
            "numpy>=2.3.5",
            "openenv-core>=0.1.1",
            "requests>=2.32.5",
            "torchvision>=0.24.1",
            "gradio>=6.0.2",
            "trackio>=0.10.0",
            "transformers>=4.57.3",
            "trl>=0.25.1",
            "pillow>=11.3.0",
            "typer>=0.20.0",
            "git+https://github.com/meta-pytorch/OpenEnv.git",
            # "https://github.com/meta-pytorch/OpenEnv.git",
        )
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