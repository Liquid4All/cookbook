# Fine-tuning LFM2-350M for browser control with GRPO and OpenEnv

### Table of contents
- [What is browser control?]()
- [Reinforcement Learning to the rescue]()
- [Architecture of our fine-tuning job]()
- [Results]()

## What is browser control?

Browser control is the ability of a language model to navigate and interact with websites by generating sequences of actions (clicking elements, typing text, scrolling) to accomplish user-specified tasks like booking flights, filling forms, or extracting information from web pages.

Browser control has many real-world applications including

- **Accessibility assistance**: A screen reader companion that navigates complex checkout flows, reads product descriptions, and completes purchases for visually impaired users on Amazon or grocery delivery sites

- **Healthcare appointment management**: An app that checks multiple clinic websites for appointment availability, books the earliest slot matching your insurance, and adds it to your calendar

- **Bill payment automation**: A monthly routine that visits utility company websites, verifies amounts, and schedules payments from your bank account

As any other powerful technology, it can also be misused, for example:

- **Review manipulation**: Bots that create fake accounts and post fraudulent reviews on Amazon, Yelp, or Google to artificially boost product ratings or damage competitors

Understanding how these systems are trained and deployed is crucial if we want to get the most out of the good uses, and minimize the impact of the bad use cases.

Now that we know what the problem is, let's see how we can solve it.


## Reinforcement learning to the rescue

B
Imagine the task your Language Model needs to solve is the following:

Example browser control task:
"Book a one-way flight from San Francisco (SFO) to Boston (BOS) on January 15, 2025 for 1 adult passenger on Google Flights"
Required action sequence:

Navigate to google.com/flights
Click "One-way" radio button
Type "San Francisco" in departure field, select "SFO" from dropdown
Type "Boston" in arrival field, select "BOS" from dropdown
Click date picker, select January 15, 2025
Click passenger selector, ensure "1 adult" is selected
Click "Search" button
Wait for results to load
Click "Select" on the cheapest direct flight option
Verify booking details and click "Continue to booking"





## The building blocks of a Reinforcement Learning problem

The 3 main components of a Reinforcement Learning problem:

### 1. The policy

This is a function (in our case a Language Model) that given the current state of the environment (context window) outputs the next action (text completion).

We will be using [LFM2-350M](https://huggingface.co/LiquidAI/LFM2-350M), which is small-yet-powerful-and-fast language model with
- knowledge (MMLU, GPQA)
- instruction following (IFEval, IFBench)
- mathematical reasoning (GSM8K, MGSM), and
- multimodal understanding (MMMU).


### 2. The environment

The environment is the component that
- provides the current state to the Language Model
- provides feedback to the Language Model in terms of rewards.
- Manages state transitions: once the policy chooses an action, the environment updates the state plus possibly provides a reward to the agent.



3. The algorithm that optimizes the policy. Given policy actions and rewards, the algorithm needs to update policy parameters to increase its performance on the task.
We will be using GRPO, a very popular algorithm for RL with Language Models, introduced by DeepSeek in 2024 [link to paper](https://arxiv.org/pdf/2402.03300).

[TODO: add diagram]



### 1. The policy -> LFM2-350M

### 2. The environment -> BrowserGym

BrowserGym is a collection of different browser automation benchmarks

- [Mini World of Bits++](https://miniwob.farama.org/)
- [WebArena](https://github.com/web-arena-x/webarena)
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena)
- [WorkArena](https://github.com/ServiceNow/WorkArena)

Each benchmark contains a collection of tasks and each task comes with different variations. If you let an agent (e.g. a Language Model) perform the task under enough variations, and update its parameters (more on this later) this agent will learn the task.

For example, we will be working with
- [Mini World of Bits++](https://miniwob.farama.org/index.html#) as the benchmark
- [click-test](https://miniwob.farama.org/environments/click-test/) as the task.

During fine-tuning we will ask the LM to solve up to N variations of this task. Once we are done fine-tuning we will evaluate the model to see it has actually learn where to click.

You can find examples of `click-test` tasks under [media/](.media/).

#### How to run BrowserGym env

The environment is basically a Docker container, that you can run either:

- Locally. In this case you need the Docker damon up and running on your system.
- As remote docker container running in your infra, for example as a pod in a Kubernetes cluster.
- Using a Hugging Face space. This option requires 0 infra setup, as the container runs on HF infrastructure and you can access it.

In this example we will be the public HF space `[SET_TO_PAULESCU_VERSION]`.


### 3. The algorithm GRPO









a collection of environment an agent (even a human) can interact with t

- 

We will be working with Mini World of Bits++ (aka MiniWoB).

OpenEnv provides a unified access to these environments through HTTP. The idea is that the RL environment runs in a Docker container either

- Locally
- Hugging Face Space
- Remotely inside your cloud/infrastructure

and your Language Model can interact with it.


MiniWoB contains over 100 web interaction environments. [Click here to see the full ist](https://miniwob.farama.org/environments/list/).

- `bisect-angle`
- `book-flight`
- `click-test`





## Out-of-the box performance

## Fine-tuning

```
make run config=functiongemma_270m.yaml
make run config=lfm2_350m.yaml
```

You can find the 2 checkpoints on Hugging Face

- [Paulescu/LFM2-350M-browsergym-20251224-013119](https://huggingface.co/Paulescu/LFM2-350M-browsergym-20251224-013119)
- [Paulescu/functiongemma-270m-it-browsergym-20251224-013151](https://huggingface.co/Paulescu/functiongemma-270m-it-browsergym-20251224-013151)


## Evaluation of the fine-tuned checkpoints


For this sample scenario, the 4 generations produced the correct response in less than 10 steps. Avg reward in this scenario is 1.
```
[DEBUG] Processing prompt 1/4
Step 2: click([13])
Step 3: noop()
Step 4: bind(this) event
Step 5: Click( parmist its child items that get fixed by FocusStatus2023914900625)'vection.xml
Step 6: click('Click Me!')
Step 7: noop()
Step 8: Click(Button)
Step 9: click('13')
[DEBUG] Processing prompt 2/4
Step 1: click('13')
[DEBUG] Processing prompt 3/4
Step 1: noop()
Step 2: noop()
Step 3: noop()
Step 4: click('13')
[DEBUG] Processing prompt 4/4
Step 1: fill('42' as Text, 'Click')
Step 2: click('Click Me!')
Step 3: click('13')
```

## TODOs
- [x] evaluate.py
- [ ] `book-flight` task [here](https://miniwob.farama.org/environments/book-flight/)


- [ ] slides


