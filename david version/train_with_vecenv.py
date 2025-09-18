#!/usr/bin/env python3
"""
Example script showing how to use VecEnv with different numbers of environments
"""

from right_agent import train_right_agent

def main():
    print("=== VecEnv Training Examples ===\n")

    # Example 1: Single environment (original method)
    print("1. Training with single environment:")
    # train_right_agent(n_envs=1, use_vecenv=False)

    # Example 2: 2 parallel environments
    print("2. Training with 2 parallel environments:")
    # train_right_agent(n_envs=2, use_vecenv=True)

    # Example 3: 4 parallel environments (recommended)
    print("3. Training with 4 parallel environments (RECOMMENDED):")
    train_right_agent(n_envs=2, use_vecenv=True)

    # Example 4: 8 parallel environments (for powerful machines)
    print("4. Training with 8 parallel environments (for powerful machines):")
    # train_right_agent(n_envs=8, use_vecenv=True)

if __name__ == "__main__":
    print("\nStarting training...\n")

    main()
