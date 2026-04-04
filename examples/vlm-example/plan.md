# Goal

Let's fine-tune [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-new-chat-template-3) on the [MMAD dataset](https://huggingface.co/datasets/jiang-cc/MMAD) to detect defects in an object, given an image of this object.

I would like to try 2 ways to fine-tune the model:

1. Using TRL, Outlines and Modal as done in this [other example](https://docs.liquid.ai/examples/customize-models/car-maker-identification.md)

2. Using [leap-finetune](https://github.com/Liquid4All/leap-finetune)

I am not very sure if option 2 will work, so let's start with option 1. Once we are done with option 1, we can try option 2, to put the cherry on top of the cake.

Use Python and uv for packaging. Place all the code inside a `src/` folder. As done in the example shared in option 1, I want to use Modal, because I don't have NVIDIA at home.

## Data preparation

The exact task I want to solve from the MMAD dataset is the following:

- Input: (`query_image`, `input_prompt`) where `input_prompt` is a new column made from the concatenation of the already existing `question` and `options` columns from the dataset.

- Output: `answer`.

As a starter, I would like to use only samples where the `question` is "Is there any defect in the object".

The original MMAD dataset has only a `train` split. To make sure we always use the same training and evaluation datasets, please create new dataset called `Paulescu/defect-detection` that contains all rows form the original MMAD dataset where `question` is "Is there any defect in the object", but it contains to splits `train` and `test`, with 90% and 10% of the samples respectively. From now on we will use `Paulescu/MMAD` and not the original `MMAD` dataset. 

Also, please format the new dataset to have columns:

- `query_image`: path to the image
- `input_prompt`: always "Is there any defect in the object. Respond Yes or No.".
- `answer`: "Yes" / "No"

When you split into train and test splits use a balanced sample so that both splits (`train` and `test`) have the same distribution of "Yes" and "No".

You can encapsulate all this logic into a single python script, like `src/prepare_data.py` with arguments:

- to (str): HF dataset name.


## Benchmarking

As done in the example shared in option 1, before fine-tuning I want to establish a benchmark performance for the following models:

- [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-new-chat-template-3)
- [LFM2-VL-450](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

Create a benchmark script `src/benchmark.py` that runs on Modal, and accepts as input parameters:

- hf model name (str): e.g. `LiquidAI/LFM2.5-VL-450M-new-chat-template-3`
- samples (int): 10, meaning use only the first 10 samples to speed up. Useful for quick debugging.
- use-constrained-geneartion (bool): wheter we use `outlines` to constraing the model output to be "Yes" or "No".

## Fine-tuning with Modal and TRL

Create a `src/finetune.py` that follows a similar structure as [the example I shared with you](https://github.com/Liquid4All/cookbook/blob/main/examples/car-maker-identification/src/car_maker_identification/fine_tune.py).

Please use caching of HF model downloads, HF datasets, and intermediate checkpoints generated during training, as explained in the [Modal docs](https://modal.com/docs/examples/unsloth_finetune#efficient-llm-finetuning-with-unsloth).


