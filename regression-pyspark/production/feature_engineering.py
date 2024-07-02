"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os  # noqa
import os.path as op
import yaml
from pyspark.ml.feature import Imputer
from pyspark.sql import Window
from pyspark.sql import functions as F

from ta_lib.pyspark import dp
from ta_lib.pyspark.core import utils
from ta_lib.pyspark.core.pipelines.processors import register_processor 

# from pyspark.sql import types as DT


# from ta_lib.core.api import (
#     get_dataframe,
#     get_feature_names_from_column_transformer,
#     get_package_path,
#     load_dataset,
#     register_processor,
#     save_pipeline,
#     DEFAULT_ARTIFACTS_PATH
# )


logger = logging.getLogger(__name__)


HERE = op.dirname(op.abspath(__file__))
# TODO: Reference the below from cli.py if possible?
with open(op.join(HERE, "conf", "data_catalog", "local.yml"), "r") as fp:
    data_config = yaml.safe_load(fp)


@register_processor("feature-engineering", "transform-data")
def transform_data(context, params):
    """Average historical prices for route."""

    input_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["final_routes_data_path"]
    )

    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["trasnformed_routes_data_path"]
    )
    spark = context.CreateSparkSession  # noqa

    final_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["clean"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )
    w = (
        Window.partitionBy(["origin_zip", "destination_zip", "vehicle_type"])
        .orderBy("pickup_date")
        .rowsBetween(-1, 30)
    )
    final_df = final_df.withColumn("pastmonth_avg", F.avg("carrier_price").over(w))
    w = (
        Window.partitionBy(["origin_zip", "destination_zip", "vehicle_type"])
        .orderBy("pickup_date")
        .rowsBetween(-1, 7)
    )
    final_df = final_df.withColumn("pastweek_avg", F.avg("carrier_price").over(w))

    final_df = final_df.select(
        "trip_id",
        "pickup_date",
        "vehicle_type",
        "origin_zip",
        "destination_zip",
        "origin_market_id",
        "destination_market_id",
        "distance",
        "market_rate_per_mile",
        "national_price",
        "pastmonth_avg",
        "pastweek_avg",
        "carrier_price",
    )
    final_df = final_df.withColumn("vehicle_type", F.col("vehicle_type").cast("string"))

    # Impute missing values
    imputer = Imputer(inputCol="national_price", outputCol="national_price")
    imputer_transform = imputer.fit(final_df)
    imputed_data = imputer_transform.transform(final_df)

    # Save the dataset
    utils.save_data(imputed_data, path=output_dataset)

    return imputed_data


@register_processor("feature-engineering", "train_test_split")
def create_train_test_split(context, params):
    """Transform dataset to create training datasets."""

    input_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["trasnformed_routes_data_path"]
    )
    output_train_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["train"]
    )

    output_test_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["test"]
    )

    # artifacts_folder = DEFAULT_ARTIFACTS_PATH
    spark = context.CreateSparkSession  # noqa

    # load datasets
    final_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["clean"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    train_df, test_df = dp.test_train_split(
        spark,
        data=final_df,
        target_col="carrier_price",
        train_prop=0.7,
        random_seed=0,
        stratify=True,
        target_type="continuous",
    )
    # Save the dataset
    utils.save_data(train_df, path=output_train_dataset)
    utils.save_data(test_df, path=output_test_dataset)
