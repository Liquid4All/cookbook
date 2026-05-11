# End-2-end tutorial on how to build a home voice assistant

I want to build an end-2-end example of a home voice agent powered by LFM2.5-Audio-1.5B.

This tutorial covers the whole journey, meaning

- Explanation of the app we will build
- Evaluation of LFM2.5-Audio-1.5B out-of-the box
- Fine-tuning of this model, to boost performance
- Quantization and deployment using llama.cpp

I want to structre my repo to separate the main components

- `app/`: contains the on-device app that uses the fine-tuned LFM2.5-Audio-1.5B model to perform audio to function calling. The functions correspond to commands a user can invoke to change the state of their home, retrieve current data, etc. (more on this on the dataset section below)

There might be different app examples, like:
    - `app/web-gpu`
    - `app/ios`

At the moment we can leave this part empty, as I would like us to focus on the evaluation and fine-tuning steps first.

- `evaluation/`: contains Python package that runs evaluations using the LFM2.5 GGUFs and the llama-like runners we need that you can find [in this repo](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF).

- `finetuning/`: contains the finetuning logic.

## Evaluation

The data we will use is from [this HF dataset](https://huggingface.co/datasets/LiquidAI/OHF-Voice-audio-20260504)
It contains 55,302 samples of spoken voice commands mapped to Home Assistant function signatures.


| Category | Function | Samples |
|---|---|---|
| **Timer** | `HassDecreaseTimer` | 7,906 |
| | `HassStartTimer` | 7,615 |
| | `HassIncreaseTimer` | 4,676 |
| | `HassTimerStatus` | 2,063 |
| | `HassCancelTimer` | 1,672 |
| | `HassPauseTimer` | 1,478 |
| | `HassUnpauseTimer` | 1,377 |
| | `HassCancelAllTimers` | 385 |
| | **Subtotal** | **27,172 (49.1%)** |
| **Lighting & devices** | `HassLightSet` | 4,723 |
| | `HassTurnOn` | 1,573 |
| | `HassTurnOff` | 1,204 |
| | `HassSetPosition` | 531 |
| | **Subtotal** | **8,031 (14.5%)** |
| **Media** | `HassMediaSearchAndPlay` | 1,990 |
| | `HassSetVolumeRelative` | 1,615 |
| | `HassSetVolume` | 1,309 |
| | `HassMediaNext` | 1,010 |
| | `HassMediaPrevious` | 860 |
| | `HassMediaUnpause` | 583 |
| | `HassMediaPause` | 467 |
| | `HassMediaPlayerMute` | 463 |
| | `HassMediaPlayerUnmute` | 446 |
| | **Subtotal** | **8,743 (15.8%)** |
| **Climate** | `HassClimateSetTemperature` | 1,547 |
| | `HassFanSetSpeed` | 1,417 |
| | `HassClimateGetTemperature` | 1,107 |
| | **Subtotal** | **4,071 (7.4%)** |
| **Vacuum & lawn** | `HassVacuumReturnToBase` | 898 |
| | `HassVacuumStart` | 802 |
| | `HassVacuumCleanArea` | 725 |
| | `HassLawnMowerDock` | 606 |
| | `HassLawnMowerStartMowing` | 384 |
| | **Subtotal** | **3,415 (6.2%)** |
| **Lists & shopping** | `HassShoppingListCompleteItem` | 500 |
| | `HassListRemoveItem` | 477 |
| | `HassListCompleteItem` | 472 |
| | `HassListAddItem` | 382 |
| | `HassShoppingListAddItem` | 208 |
| | **Subtotal** | **2,039 (3.7%)** |
| **Info & utility** | `HassGetState` | 502 |
| | `HassBroadcast` | 421 |
| | `HassGetWeather` | 380 |
| | `HassNevermind` | 170 |
| | `HassGetCurrentTime` | 135 |
| | `HassGetCurrentDate` | 129 |
| | `HassRespond` | 94 |
| | **Subtotal** | **1,831 (3.3%)** |
| | **Total** | **55,302** |


## Fine-tuning

I would like us to use [liquid-audio](https://github.com/Liquid4All/liquid-audio) and follow the steps explained [in this document](https://github.com/Liquid4All/liquid-audio-staging/blob/examples/audio-to-function-calling/FINETUNING_EXAMPLE.md) for the audio to function calling example. Take a look at the [preprocessing](https://github.com/Liquid4All/liquid-audio-staging/blob/examples/audio-to-function-calling/examples/preprocess_ohf_voice.py) and the [fine-tuning](https://github.com/Liquid4All/liquid-audio-staging/blob/examples/audio-to-function-calling/examples/train.py) scripts, both of which use Modal.

In that repo I used [liquid-audio-staging](https://github.com/Liquid4All/liquid-audio-staging) project, because back then the fine-tuning logic did not yet exists in the original [liquid-audio](https://github.com/Liquid4All/liquid-audio). As per today, the fine-tuning logic has been integrated into liquid-audio, so please dont use at all liquid-audio-staging, as it is private and my reader wont have access to it.





