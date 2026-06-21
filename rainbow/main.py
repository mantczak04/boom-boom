# -*- coding: utf-8 -*-
# Adapted from Kaixhin/Rainbow main.py (MIT).
# Rewired for ProbMinesweeper: masked actions, float replay, no Atari.

from __future__ import division

import argparse
import bz2
from datetime import datetime
import json
import os
import pickle

import numpy as np
import torch
from tqdm import trange

from .agent import Agent
from .env_adapter import Env
from .memory import ReplayMemory
from .test import test

OBS_CHANNELS = {"state": 3, "state+prob": 4}


def sample_masked_action(env: Env) -> int:
    mask = env.action_mask()
    valid = mask.nonzero(as_tuple=True)[0]
    pick = torch.randint(len(valid), (1,), device=valid.device)
    return int(valid[pick].item())


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description="Rainbow — ProbMinesweeper")
parser.add_argument("--id", type=str, default="default", help="Experiment ID")
parser.add_argument("--seed", type=int, default=123, help="Random seed")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--board-width", type=int, default=9, help="Board width")
parser.add_argument("--board-height", type=int, default=9, help="Board height")
parser.add_argument(
    "--distribution",
    type=str,
    default="correlated",
    choices=("correlated", "uniform", "constant"),
    help="Mine probability field generator",
)
parser.add_argument(
    "--obs-mode",
    type=str,
    default="state+prob",
    choices=("state", "state+prob"),
    help="Observation mode",
)
parser.add_argument("--T-max", type=int, default=int(50e6), metavar="STEPS", help="Number of training steps")
parser.add_argument("--hidden-size", type=int, default=256, metavar="SIZE", help="Network hidden size")
parser.add_argument("--noisy-std", type=float, default=0.1, metavar="σ", help="Initial standard deviation of noisy linear layers")
parser.add_argument("--atoms", type=int, default=51, metavar="C", help="Discretised size of value distribution")
parser.add_argument("--V-min", type=float, default=-5, metavar="V", help="Minimum of value distribution support")
parser.add_argument("--V-max", type=float, default=40, metavar="V", help="Maximum of value distribution support")
parser.add_argument("--model", type=str, metavar="PARAMS", help="Pretrained model (state dict)")
parser.add_argument("--memory-capacity", type=int, default=int(1e6), metavar="CAPACITY", help="Experience replay memory capacity")
parser.add_argument("--replay-frequency", type=int, default=1, metavar="k", help="Frequency of sampling from memory")
parser.add_argument("--priority-exponent", type=float, default=0.5, metavar="ω", help="Prioritised experience replay exponent (originally denoted α)")
parser.add_argument("--priority-weight", type=float, default=0.4, metavar="β", help="Initial prioritised experience replay importance sampling weight")
parser.add_argument("--multi-step", type=int, default=3, metavar="n", help="Number of steps for multi-step return")
parser.add_argument("--discount", type=float, default=0.99, metavar="γ", help="Discount factor")
parser.add_argument("--target-update", type=int, default=2000, metavar="τ", help="Number of steps after which to update target network")
parser.add_argument("--reward-clip", type=int, default=0, metavar="VALUE", help="Reward clipping (0 to disable)")
parser.add_argument("--learning-rate", type=float, default=1e-4, metavar="η", help="Learning rate")
parser.add_argument("--adam-eps", type=float, default=1.5e-4, metavar="ε", help="Adam epsilon")
parser.add_argument("--batch-size", type=int, default=32, metavar="SIZE", help="Batch size")
parser.add_argument("--norm-clip", type=float, default=10, metavar="NORM", help="Max L2 norm for gradient clipping")
parser.add_argument("--learn-start", type=int, default=int(20e3), metavar="STEPS", help="Number of steps before starting training")
parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
parser.add_argument("--evaluation-interval", type=int, default=100000, metavar="STEPS", help="Number of training steps between evaluations")
parser.add_argument("--evaluation-episodes", type=int, default=10, metavar="N", help="Number of evaluation episodes to average over")
parser.add_argument("--evaluation-size", type=int, default=500, metavar="N", help="Number of transitions to use for validating Q")
parser.add_argument("--render", action="store_true", help="Display screen (testing only)")
parser.add_argument("--enable-cudnn", action="store_true", help="Enable cuDNN (faster but nondeterministic)")
parser.add_argument("--checkpoint-interval", default=0, help="How often to checkpoint the model, defaults to 0 (never checkpoint)")
parser.add_argument("--memory", help="Path to save/load the memory from")
parser.add_argument("--disable-bzip-memory", action="store_true", help="Don't zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)")


def configure_args(args: argparse.Namespace) -> None:
    args.history_length = 1
    args.obs_channels = OBS_CHANNELS[args.obs_mode]
    args.distribution_kwargs = {}


def log(s: str) -> None:
    print("[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    with bz2.open(memory_path, "rb") as zipped_pickle_file:
        return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "wb") as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, "wb") as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def append_metrics(metrics_path: str, step: int, win_rate: float, mean_reward: float, mean_steps: float) -> None:
    record = {
        "step": step,
        "win_rate": win_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(dqn: Agent, checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "model.pt")
    dqn.save(checkpoint_dir, "model.pt")
    return path


def main(argv: list[str] | None = None) -> None:
    args = parser.parse_args(argv)
    configure_args(args)

    print(" " * 26 + "Options")
    for k, v in vars(args).items():
        print(" " * 26 + k + ": " + str(v))
    run_name = args.id
    results_dir = os.path.join("results", run_name)
    checkpoint_dir = os.path.join("checkpoints", run_name)
    runs_dir = os.path.join("runs", run_name)
    metrics_path = os.path.join(runs_dir, "metrics.jsonl")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    metrics = {"steps": [], "rewards": [], "Qs": [], "win_rates": [], "best_avg_reward": -float("inf")}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        try:
            torch.zeros(1, device="cuda")
            args.device = torch.device("cuda")
            torch.cuda.manual_seed(np.random.randint(1, 10000))
            torch.backends.cudnn.enabled = args.enable_cudnn
        except RuntimeError:
            log("CUDA unavailable or incompatible — falling back to CPU")
            args.device = torch.device("cpu")
    else:
        args.device = torch.device("cpu")

    env = Env(args)
    env.train()
    action_space = env.action_space()

    dqn = Agent(args, env)

    if args.model is not None and not args.evaluate and args.memory and os.path.exists(args.memory):
        mem = load_memory(args.memory, args.disable_bzip_memory)
    else:
        mem = ReplayMemory(args, args.memory_capacity)

    priority_weight_increase = (1 - args.priority_weight) / max(args.T_max - args.learn_start, 1)

    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state = env.reset()
        action = sample_masked_action(env)
        next_state, _, done = env.step(action)
        val_mem.append(state, -1, 0.0, done)
        state = next_state
        T += 1

    if args.evaluate:
        dqn.eval()
        avg_reward, avg_Q, win_rate, mean_steps = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)
        print(
            f"Avg. reward: {avg_reward} | Avg. Q: {avg_Q} | Win-rate: {100 * win_rate:.1f}% | Mean steps: {mean_steps:.1f}"
        )
    else:
        dqn.train()
        done = True
        for T in trange(1, args.T_max + 1):
            if done:
                state = env.reset()

            if T % args.replay_frequency == 0:
                dqn.reset_noise()

            action = dqn.act(state, action_mask=env.action_mask())
            next_state, reward, done = env.step(action)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            mem.append(state, action, reward, done)

            if T >= args.learn_start:
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)

                if T % args.evaluation_interval == 0:
                    dqn.eval()
                    avg_reward, avg_Q, win_rate, mean_steps = test(args, T, dqn, val_mem, metrics, results_dir)
                    save_checkpoint(dqn, checkpoint_dir)
                    append_metrics(metrics_path, T, win_rate, avg_reward, mean_steps)
                    log(
                        f"T = {T} / {args.T_max} | Avg. reward: {avg_reward} | Avg. Q: {avg_Q} | "
                        f"Win-rate: {100 * win_rate:.1f}% | Mean steps: {mean_steps:.1f}"
                    )
                    dqn.train()

                    if args.memory is not None:
                        save_memory(mem, args.memory, args.disable_bzip_memory)

                if T % args.target_update == 0:
                    dqn.update_target_net()

                if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                    save_checkpoint(dqn, checkpoint_dir)

            state = next_state

        dqn.eval()
        avg_reward, avg_Q, win_rate, mean_steps = test(args, T, dqn, val_mem, metrics, results_dir)
        save_checkpoint(dqn, checkpoint_dir)
        if T % args.evaluation_interval != 0:
            append_metrics(metrics_path, T, win_rate, avg_reward, mean_steps)
        log(
            f"Final | T = {T} | Avg. reward: {avg_reward} | Avg. Q: {avg_Q} | "
            f"Win-rate: {100 * win_rate:.1f}% | Mean steps: {mean_steps:.1f}"
        )
        log(f"Checkpoint: {os.path.join(checkpoint_dir, 'model.pt')}")
        log(f"Metrics: {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
