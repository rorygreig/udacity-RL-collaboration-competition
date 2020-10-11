import argparse
from src.tennis_env import TennisMultiAgentEnv

from src.maddpg.maddpg import MADDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train

    env = TennisMultiAgentEnv("./Tennis_Linux/Tennis.x86_64")

    maddpg = MADDPG(env)

    if train:
        scores = maddpg.train()
    else:
        maddpg.run_with_stored_weights()


if __name__ == "__main__":
    main()
