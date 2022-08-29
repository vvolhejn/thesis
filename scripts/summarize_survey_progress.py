# Used to quickly check how many people have filled out the survey.

import pandas as pd


def main(path):
    df = pd.read_csv(path)
    names = df["firstname"] + " " + df["lastname"]
    names = names.unique()

    print("Respondents:")
    print("\n".join(list(names)))
    print(f"\n{len(names)} respondents, {len(df)} lines")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")

    args = parser.parse_args()
    main(args.path)