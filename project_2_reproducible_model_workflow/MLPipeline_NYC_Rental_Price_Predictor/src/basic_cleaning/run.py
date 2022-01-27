#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""

import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    '''Download input artifact. 
    This will also log that this script is using this particular version of the artifact.
    '''

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info("Basic cleaning")

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Drop rows in the dataset that are not in the proper geolocation.
    idx = df['longitude'].between(-74.25, -
                                  73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned data as csv")
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset uploaded to wandb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="basic data cleaning")

    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="temporary directory for dataset storage",
        required=True
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="input value",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="input value",
        required=True
    )

    args = parser.parse_args()

    go(args)