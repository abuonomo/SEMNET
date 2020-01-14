import argparse
import logging

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def create_ancient_network(all_networks, nn, all_KW):
    pass


def main(msg, feature):
    LOG.info(f'{msg} and feature is {feature}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input txt file')
    parser.add_argument('--feature', dest='feature', action='store_true')
    parser.add_argument('--no-feature', dest='feature', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.i, args.feature)
