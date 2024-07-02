"""Processors for the model training step of the worklow."""
import logging
import os  # noqa
import os.path as op
import yaml
from pyspark.ml.regression import RandomForestRegressor

from ta_lib.pyspark import dp, features
from ta_lib.pyspark.core import utils
from ta_lib.pyspark.core.constants import DEFAULT_ARTIFACTS_PATH
from ta_lib.pyspark.core.pipelines.processors import register_processor
from ta_lib.pyspark.core.utils import save_model

logger = logging.getLogger(__name__)


HERE = op.dirname(op.abspath(__file__))
# TODO: Reference the below from cli.py if possible?
with open(op.join(HERE, "conf", "data_catalog", "local.yml"), "r") as fp:
    data_config = yaml.safe_load(fp)
artifacts_folder = DEFAULT_ARTIFACTS_PATH


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    train_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["train"]
    )

    spark = context.CreateSparkSession

    # load datasets
    train_df = utils.read_data(
        spark=context.spark,
        paths=[train_dataset],
        fs=data_config["clean"][
            "filesystem"
        ],  # TODO: context.data_catalog['data']['raw']['filesystem']
    )

    cols_to_drop = ["pickup_date"]
    train_df = train_df.drop(*cols_to_drop)

    outlier = dp.Outlier_Treatment(
        cols=[
            "distance",
            "market_rate_per_mile",
            "carrier_price",
            "pastweek_avg",
            "pastmonth_avg",
        ],
        drop=True,
        cap=False,
        method="iqr",
        iqr_multiplier=1.5,
    )
    outlier.fit(train_df)
    train_df = outlier.transform(train_df)

    # Encoding features
    encoder = features.Encoder(
        cols=["vehicle_type", "origin_zip", "destination_zip"],
        rules={
            "vehicle_type": {"method": "onehot"},
            "origin_zip": {"method": "onehot"},
            "destination_zip": {"method": "onehot"},
        },
    )
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)

    num_cols = dp.list_numerical_columns(train_df)
    cat_cols = dp.list_categorical_columns(train_df)
    date_cols = dp.list_datelike_columns(train_df)
    bool_cols = dp.list_boolean_columns(train_df)

    target_col = "carrier_price"
    id_cols = "trip_id"
    # non_relevant_cat_cols = []
    non_relevant_num_cols = [x for x in num_cols if "index" in x]
    feature_cols = train_df.columns
    feature_cols = [
        x
        for x in feature_cols
        if x
        not in cat_cols
        + date_cols
        + bool_cols
        + [id_cols]
        + non_relevant_num_cols
        + [target_col]
    ]

    train_df = dp.generate_features_vector(
        spark, train_df, feature_cols, output_col="features"
    )

    # renaming target col as y
    train_df = train_df.withColumnRenamed(target_col, "y")

    # estimator declaration
    m = RandomForestRegressor(
        featuresCol="features", labelCol="y", predictionCol="yhat"
    )
    model = m.fit(train_df)

    # save fitted training pipeline
    save_model(model, artifacts_folder)
