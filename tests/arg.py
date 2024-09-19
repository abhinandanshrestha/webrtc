import argparse

def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--call', type=str, help='Number to call', required=True)

    # Parse the arguments
    args = parser.parse_args()
    print(args.call)

if __name__ == '__main__':
    main()