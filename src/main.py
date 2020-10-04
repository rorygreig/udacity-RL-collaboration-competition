import argparse
from src.tennis_env import TennisMultiAgentEnv

from src.ddpg.ddpg import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train
    use_ppo = parsed_args.ppo

    env = TennisMultiAgentEnv("./Tennis_Linux/Tennis.x86_64")

    algo = DDPG(env)

    if train:
        scores = algo.train()
    else:
        algo.run_with_stored_weights()


if __name__ == "__main__":
    main()
