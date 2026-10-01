import argparse
from qcl.config import load_config
from qcl.engine import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM quantum continual segmentation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Trusted project checkpoint; identical config required")
    args = parser.parse_args()
    print(train(load_config(args.config), args.resume))
