#!/usr/bin/env python3
"""Generate a synthetic tool-call SFT dataset for an autonomous game agent.

Two regimes:
  * instruction  -- map a player command to one or more tool calls
  * AUTO         -- with no instruction, inspect state and decide proactively

Use --profile sleep to emit a small focused dataset for patching a single hard rule
("only sleep when the player sleeps") into an existing adapter.

All names, positions and states are generated at random; no real data is used.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SYSTEM = """You are a companion agent in a block-building survival game.
Responsibilities: keep the player company, follow them, gather materials, craft and
equip tools, and protect everyone when danger appears.

## Available actions (<tool_call> format; multiple calls allowed)
  chat              <parameter=message>reply text</parameter>
  follow_entity     <parameter=entity_id>1</parameter> <parameter=range>2</parameter>
  attack_nearest    attack the closest hostile mob
  flee              retreat from a threat
  gather            <parameter=block_name>oak_log</parameter> <parameter=count>4</parameter>
  craft_and_equip   <parameter=item_name>wooden_pickaxe</parameter>
  craft_bed         craft a bed from wool + planks in inventory
  sleep_in_bed      sleep in a nearby bed
  eat               <parameter=item_name>bread</parameter>
  move_to / look_at / jump / stop_moving / get_state / stop_attack

## Autonomous decision rules (act proactively; do not only talk)
- hostile mob + HP > 8  -> attack_nearest
- hostile mob + HP <= 8 -> flee
- low health            -> flee or eat
- no tool in inventory  -> gather logs/stone, then craft_and_equip
- no weapon             -> gather logs, then craft_and_equip a sword
- player asks for a material -> gather it
- night but players are still awake -> DO NOT sleep; stay and keep them company
- only when a nearby player is sleeping may you sleep
- idle                  -> gather materials, craft gear, or observe with the player

## Output rules
- Emit only <tool_call>...</tool_call>. No explanations, no thinking lines.
- Multiple tool calls per turn are allowed. Act on state signals before chatting.
"""

MOBS = ["zombie", "creeper", "skeleton", "spider", "husk", "drowned"]


def make_state(rng: random.Random, *, hostile=False, hp=None, night=False,
               bed=False, inv=None, players_sleeping=False) -> dict:
    x, y, z = rng.randint(-100, 100), 64, rng.randint(-100, 100)
    hp = hp if hp is not None else rng.randint(10, 20)
    inv = inv if inv is not None else rng.sample(
        ["oak_log", "stone", "bread", "wool", "oak_planks", "iron_ore"], rng.randint(0, 4))
    mob_line, danger = "", "NONE"
    if hostile:
        mid, mtype, dist = rng.randint(2, 9), rng.choice(MOBS), rng.randint(2, 12)
        mob_line = f", {mtype}(id={mid})@({x+dist},{y},{z})"
        danger = f"{mtype}(dist={dist})"
    bed_line = f"white_bed@({x+3},{y},{z+2})" if bed else "NONE"

    alerts = []
    if hostile:
        alerts.append("!! HOSTILE NEARBY - attack_nearest or flee !!")
    if hp <= 8:
        alerts.append("!! LOW HEALTH - flee or eat !!")
    if players_sleeping:
        alerts.append("!! PLAYERS SLEEPING - you may sleep too if you have a bed !!")
    elif night:
        alerts.append("!! NIGHT but players are still awake - do NOT sleep yet !!")
    alert = "\n" + "\n".join(alerts) if alerts else ""

    text = (f"Position: ({x},{y},{z})  Health: {hp}/20  Food: {rng.randint(6,20)}/20\n"
            f"Time: {18000 if night else rng.randint(1000,11000)}  "
            f"IsNight: {night}  Weather: clear\n"
            f"Inventory: {', '.join(inv) if inv else 'empty'}\n"
            f"Nearby entities: Player@({x+rng.randint(1,10)},{y},{z+rng.randint(1,10)}){mob_line}\n"
            f"DANGER (hostiles): {danger}\n"
            f"Nearby beds: {bed_line}{alert}")
    return {"x": x, "y": y, "z": z, "hp": hp, "inv": inv, "text": text}


def tc(fn: str, **params) -> str:
    inner = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params.items())
    return f"<tool_call><function={fn}>{inner}</function></tool_call>"


def reply(rng: random.Random, msg: str) -> str:
    return tc("chat", message=msg)


# Instruction regime: (player utterance, generator)
INSTRUCTIONS = [
    ("hello", lambda s, r: reply(r, "Hi! I'm right here with you.")),
    ("follow me", lambda s, r: reply(r, "Coming!") + tc("follow_entity", entity_id=1, range=2)),
    ("stop", lambda s, r: tc("stop_moving") + reply(r, "Stopped.")),
    ("cut some wood", lambda s, r: reply(r, "I'll chop some logs.") + tc("gather", block_name="oak_log", count=5)),
    ("get stone", lambda s, r: tc("gather", block_name="stone", count=5) + reply(r, "Here you go.")),
    ("find iron", lambda s, r: reply(r, "Looking for iron ore.") + tc("gather", block_name="iron_ore", count=4)),
    ("get wool", lambda s, r: reply(r, "I'll find some wool.") + tc("gather", block_name="white_wool", count=3)),
    ("make a pickaxe", lambda s, r: reply(r, "Crafting a pickaxe.") + tc("craft_and_equip", item_name="wooden_pickaxe")),
    ("make a sword", lambda s, r: reply(r, "Arming myself.") + tc("craft_and_equip", item_name="wooden_sword")),
    ("make a bed", lambda s, r: reply(r, "I'll make a bed.") + tc("gather", block_name="white_wool", count=3) + tc("craft_bed")),
    ("go to sleep", lambda s, r: reply(r, "I'll sleep now.") + tc("sleep_in_bed")),
    ("a zombie!", lambda s, r: reply(r, "Watch out, I'll handle it!") + tc("attack_nearest")),
    ("protect me", lambda s, r: reply(r, "I've got your back!") + tc("attack_nearest") + tc("follow_entity", entity_id=1, range=1)),
    ("run away", lambda s, r: reply(r, "Retreating!") + tc("flee")),
    ("eat something", lambda s, r: tc("eat", item_name="bread") + reply(r, "That was good.")),
]


def auto_example(rng: random.Random) -> tuple[dict, str]:
    kind = rng.random()
    if kind < 0.18:   # mob + high HP -> attack
        s = make_state(rng, hostile=True, hp=rng.randint(12, 20))
        return s, reply(rng, "A mob is near, I'll engage it.") + tc("attack_nearest")
    if kind < 0.30:   # mob + low HP -> flee
        s = make_state(rng, hostile=True, hp=rng.randint(4, 8))
        return s, reply(rng, "I'm low on health, retreating.") + tc("flee")
    if kind < 0.40:   # low HP -> eat
        s = make_state(rng, hp=rng.randint(4, 8))
        return s, reply(rng, "I need to eat.") + tc("eat", item_name="bread")
    if kind < 0.55:   # no tool -> gather + craft
        s = make_state(rng, inv=rng.sample(["oak_log", "stone", "bread"], rng.randint(1, 3)))
        return s, reply(rng, "No tool yet, let me gather and craft.") + tc("gather", block_name="oak_log", count=5) + tc("craft_and_equip", item_name="wooden_pickaxe")
    if kind < 0.68:   # no weapon -> craft sword
        s = make_state(rng, inv=["oak_log", "oak_planks"])
        return s, reply(rng, "I need a weapon.") + tc("gather", block_name="oak_log", count=4) + tc("craft_and_equip", item_name="wooden_sword")
    if kind < 0.78:   # proactive gather
        block = rng.choice(["oak_log", "stone", "white_wool", "iron_ore"])
        return make_state(rng), reply(rng, "Gathering supplies.") + tc("gather", block_name=block, count=rng.randint(3, 6))
    if kind < 0.88:   # night, players sleeping, bed available -> sleep
        s = make_state(rng, night=True, bed=True, players_sleeping=True)
        return s, reply(rng, "Everyone's asleep, I'll rest too.") + tc("sleep_in_bed")
    if kind < 0.96:   # night, players awake -> stay, do not sleep
        s = make_state(rng, night=True, bed=True, players_sleeping=False)
        return s, reply(rng, "Still awake, I'll stay with you.") + tc("follow_entity", entity_id=1, range=2)
    return make_state(rng), reply(rng, "Watching over you.")  # idle companionship


def sleep_override_example(rng: random.Random) -> tuple[dict, str]:
    sleeping = rng.random() < 0.5
    bed = rng.random() < 0.8
    s = make_state(rng, night=True, bed=bed, players_sleeping=sleeping,
                   hostile=(rng.random() < 0.15))
    if sleeping:
        if bed:
            return s, reply(rng, "The player is asleep, I'll sleep too.") + tc("sleep_in_bed")
        return s, reply(rng, "I'll make a bed and sleep.") + tc("gather", block_name="white_wool", count=3) + tc("craft_bed") + tc("sleep_in_bed")
    if s["hp"] <= 8:
        return s, reply(rng, "I'm hurt, retreating.") + tc("flee")
    return s, reply(rng, "Player is awake, I'll stay.") + tc("follow_entity", entity_id=1, range=2)


def build_example(rng: random.Random, *, profile: str) -> dict:
    if profile == "sleep":
        s, out = sleep_override_example(rng)
        user = "AUTO"
    elif rng.random() < 0.5:
        s, out = auto_example(rng)
        user = "AUTO"
    else:
        cmd, gen = rng.choice(INSTRUCTIONS)
        s = make_state(rng, hostile=(rng.random() < 0.25))
        user = f"Player says: {cmd}"
        out = gen(s, rng)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Current state:\n" + s["text"] + "\n\n" + user},
        {"role": "assistant", "content": out},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("./data/sft_dataset.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", choices=["default", "sleep"], default="default")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            ex = build_example(rng, profile=args.profile)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            for fn in ["attack_nearest", "flee", "gather", "craft_and_equip", "craft_bed",
                       "sleep_in_bed", "follow_entity", "eat", "chat"]:
                if fn in ex["messages"][2]["content"]:
                    counts[fn] += 1

    print(f"wrote {args.n} examples to {args.out}")
    print("action coverage:", dict(counts))


if __name__ == "__main__":
    main()
