import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect import signature
from operator import itemgetter
from pyspark.mllib.common import (  # common extention in handyspark
    JavaModelWrapper,
    _java2py,
    _py2java,
)
from pyspark.mllib.evaluation import (
    BinaryClassificationMetrics,
    MulticlassMetrics,
)
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql.types import DoubleType, StructField, StructType

from tigerml.pyspark.model_eval.handy_spark_cd import (
    call2,
    confusionMatrix,
    fMeasureByThreshold,
    getMetricsByThreshold,
    plot_pr_curve,
    plot_roc_curve,
    pr,
    pr_curve,
    precisionByThreshold,
    print_confusion_matrix,
    recallByThreshold,
    roc,
    roc_curve,
    thresholds,
)

mpl.rc("lines", markeredgewidth=0.5)
JavaModelWrapper.call2 = call2


def __init__(self, scoreAndLabels, scoreCol="score", labelCol="label"):
    if isinstance(scoreAndLabels, DataFrame):
        scoreAndLabels = scoreAndLabels.select(scoreCol, labelCol).rdd.map(
            lambda row: (float(row[scoreCol][1]), float(row[labelCol]))
        )

    sc = scoreAndLabels.ctx
    sql_ctx = SQLContext.getOrCreate(sc)
    df = sql_ctx.createDataFrame(
        scoreAndLabels,
        schema=StructType(
            [
                StructField("score", DoubleType(), nullable=False),
                StructField("label", DoubleType(), nullable=False),
            ]
        ),
    )

    java_class = sc._jvm.org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    java_model = java_class(df._jdf)
    super(BinaryClassificationMetrics, self).__init__(java_model)


BinaryClassificationMetrics.__init__ = __init__
BinaryClassificationMetrics.thresholds = thresholds
BinaryClassificationMetrics.roc = roc
BinaryClassificationMetrics.pr = pr
BinaryClassificationMetrics.fMeasureByThreshold = fMeasureByThreshold
BinaryClassificationMetrics.precisionByThreshold = precisionByThreshold
BinaryClassificationMetrics.recallByThreshold = recallByThreshold
BinaryClassificationMetrics.getMetricsByThreshold = getMetricsByThreshold
BinaryClassificationMetrics.confusionMatrix = confusionMatrix
BinaryClassificationMetrics.plot_roc_curve = plot_roc_curve
BinaryClassificationMetrics.plot_pr_curve = plot_pr_curve
BinaryClassificationMetrics.print_confusion_matrix = print_confusion_matrix
