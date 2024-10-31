#!/usr/bin/env python
"""Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact"""
import argparse
import logging
import wandb
import pandas as pd
import os

# DO NOT MODIFY
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
# DO NOT MODIFY

def go(args):
    
    logger.info('Starting wandb run.')
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info('Fetching raw dataset....')
    local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    # EDA with arguments passed into the step
    # Dropping the outliers 
    logger.info('Cleaning data.')
    idx = df['price'].between(float(args.min_price), float(args.max_price))
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    # TODO: add code to fix the issue happened when testing the model
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Restrict long and lat
    idx_coord = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx_coord].copy()

    # Convert results to CSV file
    logger.info("Saving everything to a csv file.")
    df.to_csv(args.output_artifact, index=False)

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()


    # Save the cleaned data
    logger.info('Creating the Artifact.')
    #df.to_csv('clean_sample.csv', index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)

    logger.info("Logging the artifact.")
    run.log_artifact(artifact)
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The output artifact",
        required=True
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="The output artifact type",
        required=True
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="The output artifact description",
        required=True
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price",
        required=True
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price",
        required=True
    )

    args = parser.parse_args()
    go(args)
