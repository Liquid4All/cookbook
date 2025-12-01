# GRPO training with BrowserGym

https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/browsergym.py

## Environment setup

Installation instructions for `OpenEnv` are [here](https://meta-pytorch.org/OpenEnv/quickstart/)
```sh
git clone https://github.com/meta-pytorch/OpenEnv
uv pip install -e OpenEnv

# not sure if needed
uv openenv-core
```

```python
# Update import in file OpenEnv/src/envs/browsergym_env/client.py:
# from browsergym_env.models import (
#     BrowserGymAction,
#     BrowserGymObservation,
#     BrowserGymState,
# )
from .models import (
    BrowserGymAction,
    BrowserGymObservation,
    BrowserGymState,
)
```

Start Docker container with the RL environment:

```sh
docker login registry.hf.space -u Paulescu
Password: <PASTE_YOUR_HF_TOKEN_HERE>
```

```sh
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  registry.hf.space/burtenshaw-browsergym-env-95313e2:latest
```




