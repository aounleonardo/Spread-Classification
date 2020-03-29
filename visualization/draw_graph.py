import argparse
from src.draw import draw
from spread_classification.utils import load_graph
import matplotlib.pyplot as plt

def main(args):
    graph = load_graph(args.graph.name)
    figure = draw(graph)
    plt.savefig(args.image_output.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes a graph file and draws its layout in a png file"
    )
    parser.add_argument("--graph", "-i", type=argparse.FileType("rb"), required=True)
    parser.add_argument(
        "--image-output", "-o", type=argparse.FileType("wb"), required=True
    )

    main(parser.parse_args())
