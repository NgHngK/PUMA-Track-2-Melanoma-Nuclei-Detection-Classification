import argparse
from .train import train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--plan", choices=("A", "B"), required=True)
    p.add_argument("--fold", type=int, choices=range(4), required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    a = p.parse_args()
    train_one(a.features, a.results, a.plan, a.fold, a.epochs, a.batch_size)


if __name__ == "__main__":
    main()
