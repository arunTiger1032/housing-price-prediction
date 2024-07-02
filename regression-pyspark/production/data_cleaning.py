"""Processors for the data cleaning step of the worklow.

The processors in this step apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import os.path as op
import pandas as pd
import yaml
from pyspark.sql import Window  # noqa
from pyspark.sql import functions as F
from pyspark.sql import types as DT  # noqa

from ta_lib.pyspark.core import utils
from ta_lib.pyspark.core.pipelines.processors import register_processor 

HERE = op.dirname(op.abspath(__file__))

with open(op.join(HERE, "conf", "data_catalog", "local.yml"), "r") as fp:
    data_config = yaml.safe_load(fp)


@register_processor("data-cleaning", "carrier_data")
def clean_carrier_table(context, params):
    """
    Clean the carrier_data dataset.

    The  dataset contains information about  price for trips completed over routes.
     Column carrier_price contains price for the trip_id covering a certain distance between origin_zip and destination_zip with a specific vehicle_type)
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["carrier_data_path"]
    )
    # TODO: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['carrier_data_path']
    output_dataset = (
        data_config["clean"]["base_path"] + data_config["clean"]["carrier_data_path"]
    )

    spark = context.CreateSparkSession  # noqa

    carrier_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fmt="csv",
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    # Filter carrier data dataframe
    carrier_df_clean = carrier_df.withColumn(
        "pickup_date",
        F.to_date(
            F.unix_timestamp(F.col("pickup_date"), "dd/MM/yyyy").cast("timestamp")
        ),
    ).filter(F.col("pickup_date") <= reference_date)

    # Save the dataset
    utils.save_data(carrier_df_clean, path=output_dataset)

    return carrier_df_clean


@register_processor("data-cleaning", "fuelprice_data")
def clean_fuelprice_table(context, params):
    """
    Clean the fuel_prices dataset.

    The fuel prices dataset contains the prevailing market fuel price on a given date.
     Prices are reported on the first Monday of each week.
     For any trips scheduled during a given week, the reference fuel price would be the price prevailing the earlier Monday.
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["fuel_prices_data_path"]
    )
    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["fuel_prices_data_path"]
    )

    spark = context.CreateSparkSession  # noqa

    fuelprices_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fmt="csv",
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    fuelprices_df_clean = fuelprices_df.withColumn(
        "first_monday_of_week",
        F.to_date(F.unix_timestamp(F.col("date"), "dd-MM-yyyy").cast("timestamp")),
    ).filter(F.col("first_monday_of_week") <= reference_date)

    # Save the dataset
    utils.save_data(fuelprices_df_clean, path=output_dataset)

    return fuelprices_df_clean


@register_processor("data-cleaning", "market_carrier_rates_data")
def clean_market_carrier_rates_table(context, params):
    """
    Clean the market carrier prices dataset.

    This contains total cost and total distance serviced by each vehicle_type across all carrier providers in the geography
    """

    input_dataset = (
        data_config["raw"]["base_path"]
        + data_config["raw"]["market_carrier_rates_data_path"]
    )  # TODO: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['fuel_prices_data_path']
    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["market_carrier_rates_data_path"]
    )

    market_carrier_rates_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fmt="csv",
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    market_carrier_rates_clean = (
        market_carrier_rates_df.withColumn(
            "week_ending_date",
            F.to_date(
                F.unix_timestamp(F.col("week_ending_date"), "yyyy-MM-dd").cast(
                    "timestamp"
                )
            ),
        )
        .filter(F.col("week_ending_date") <= reference_date)
        .withColumn(
            "first_monday_of_week",
            F.to_date(
                F.unix_timestamp(F.col("first_monday_of_week"), "yyyy-MM-dd").cast(
                    "timestamp"
                )
            ),
        )
    )

    # Save the dataset
    utils.save_data(market_carrier_rates_clean, path=output_dataset)

    return market_carrier_rates_clean


@register_processor("data-cleaning", "final_routes_data")
def clean_final_routes_table(context, params):
    """
    Clean the final routes data.

    This contains the final result from merging all the data tables.
    """
    input_dataset_carrier = (
        data_config["clean"]["base_path"] + data_config["clean"]["carrier_data_path"]
    )
    input_dataset_fuelprice = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["fuel_prices_data_path"]
    )
    input_dataset_market_rates = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["market_carrier_rates_data_path"]
    )
    input_dataset_route_mapping = (
        data_config["raw"]["base_path"] + data_config["raw"]["route_mapping_data_path"]
    )
    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["final_routes_data_path"]
    )

    spark = context.CreateSparkSession  # noqa

    carrier_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset_carrier],
        fs=data_config["clean"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    fuelprices_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset_fuelprice],
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    market_carrier_rates_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset_market_rates],
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    route_mapping_df = utils.read_data(
        spark=context.spark,
        paths=[input_dataset_route_mapping],
        fmt="csv",
        fs=data_config["raw"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    print(f"{'**-----**' * 20} Trying the first join")

    carrier_df = (
        carrier_df.join(
            route_mapping_df.select(["zipcode", "market_id"]).withColumnRenamed(
                "market_id", "origin_market_id"
            ),
            carrier_df.origin_zip == route_mapping_df.zipcode,
            how="left",
        )
        .drop("zipcode")
        .join(
            route_mapping_df.select(["zipcode", "market_id"]).withColumnRenamed(
                "market_id", "destination_market_id"
            ),
            carrier_df.destination_zip == route_mapping_df.zipcode,
            how="left",
        )
        .drop("zipcode")
        .where(
            (F.col("origin_market_id").isNotNull())
            & (F.col("destination_market_id").isNotNull())
        )
    )

    droplist = [
        "origin_city",
        "origin_state",
        "origin_country",
        "destination_city",
        "destination_state",
        "destination_country",
    ]
    carrier_df = carrier_df.drop(*droplist)

    # Create a column for first Monday of the week in order to join with the weekly fuel prices data

    carrier_df = carrier_df.withColumn(
        "day_of_week", F.dayofweek(F.col("pickup_date")) - 2
    ).withColumn("first_monday_of_week", F.expr("date_sub(pickup_date, day_of_week)"))

    dropcols = ["sum_of_cost_all_providers", "sum_of_dist_all_providers"]
    market_carrier_rates_df = market_carrier_rates_df.join(
        route_mapping_df.select(["zipcode", "market_id"]).withColumnRenamed(
            "market_id", "origin_market_id"
        ),
        market_carrier_rates_df.origin_zip == route_mapping_df.zipcode,
        how="left",
    ).drop("zipcode")

    market_carrier_rates_df = (
        market_carrier_rates_df.withColumn(
            "first_monday_of_week",
            F.to_date(
                F.unix_timestamp(
                    F.col("first_monday_of_week"), "dd-MM-yyyy HH:mm:ss"
                ).cast("timestamp")
            ),
        )
        .join(
            route_mapping_df.select(["zipcode", "market_id"]).withColumnRenamed(
                "market_id", "destination_market_id"
            ),
            market_carrier_rates_df.destination_zip == route_mapping_df.zipcode,
            how="left",
        )
        .drop("zipcode")
    )
    market_carrier_rates_df = market_carrier_rates_df.groupby(
        [
            "first_monday_of_week",
            "origin_market_id",
            "destination_market_id",
            "vehicle_type",
        ]
    ).agg(
        F.sum("total_cost_all_providers").alias("sum_of_cost_all_providers"),
        F.sum("total_distance_all_providers").alias("sum_of_dist_all_providers"),
    )

    market_carrier_rates_df = market_carrier_rates_df.withColumn(
        "market_rate_per_mile",
        F.col("sum_of_cost_all_providers") / F.col("sum_of_dist_all_providers"),
    )
    dropcols = ["sum_of_cost_all_providers", "sum_of_dist_all_providers"]
    market_carrier_rates_df = market_carrier_rates_df.drop(*dropcols)

    final_df = (
        carrier_df.select(
            [
                "trip_id",
                "pickup_date",
                "first_monday_of_week",
                "origin_zip",
                "destination_zip",
                "origin_market_id",
                "destination_market_id",
                "vehicle_type",
                "distance",
                "carrier_price",
            ]
        )
        .join(
            market_carrier_rates_df,
            on=[
                "origin_market_id",
                "destination_market_id",
                "vehicle_type",
                "first_monday_of_week",
            ],
            how="left",
        )
        .where(F.col("market_rate_per_mile").isNotNull())
        .join(fuelprices_df, on=["first_monday_of_week"], how="left")
        .where(F.col("national_price").isNotNull())
    )

    # Save the dataset
    utils.save_data(final_df, path=output_dataset)
