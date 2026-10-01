import argparse
from qcl.engine import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the saved held-out split")
    parser.add_argument("--checkpoint", required=True, help="Trusted checkpoint from this project")
    parser.add_argument("--output")
    args = parser.parse_args()
    test(args.checkpoint, args.output)
