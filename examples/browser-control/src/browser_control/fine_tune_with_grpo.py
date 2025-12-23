

from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = get_modal_app("browser-control-fine-tune-with-grpo")
image = get_docker_image()
volume = get_volume("hf-model-cache")


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": volume},
    secrets=get_secrets(),
    timeout=2 * 60 * 60,  # 2 hours timeout for training
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def main(config: Fine) -> None:

    # client to interact with the BrowserGym environment
    from envs.browsergym_env import BrowserGymEnv
    space_url = "https://burtenshaw-browsergym-v2.hf.space"
    client = BrowserGymEnv(base_url=space_url)






    


if __name__ == "__main__":
    main()