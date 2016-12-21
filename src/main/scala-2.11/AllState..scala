import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression

import scala.collection.mutable.ListBuffer
//http://stackoverflow.com/questions/32357226/how-to-vectorize-dataframe-columns-for-ml-algorithms
//https://blog.knoldus.com/2016/02/09/a-sample-ml-pipeline-for-clustering-in-spark/
//http://stackoverflow.com/questions/34167105/using-spark-mls-onehotencoder-on-multiple-columns
//http://www.sparktutorials.net/Spark+MLLib+-+Predict+Store+Sales+with+ML+Pipelines
object AllState {
  def main(args: Array[String]) {
    /*val conf = new SparkConf()
      .setMaster(args(0))
      .setAppName("AllState")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")*/

    val spark = SparkSession
      .builder
        .master(args(0))
        .appName("AllState")
        .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    //code goes here
    import spark.implicits._
    val train = spark.read.option("header","true").csv(args(1))
    //val testDF = spark.read.csv(args(2))
    /*train.show()
    for(col <- train.columns)
      {
        train.groupBy(col).count().show()
      }*/
    val toDouble = udf[Double, String]( _.toDouble)
    val trainDF = train
      .withColumn("cont1",toDouble(train("cont1")))
      .withColumn("cont2",toDouble(train("cont2")))
      .withColumn("cont3",toDouble(train("cont3")))
      .withColumn("cont4",toDouble(train("cont4")))
      .withColumn("cont5",toDouble(train("cont5")))
      .withColumn("cont6",toDouble(train("cont6")))
      .withColumn("cont7",toDouble(train("cont7")))
      .withColumn("cont8",toDouble(train("cont8")))
      .withColumn("cont9",toDouble(train("cont9")))
      .withColumn("cont10",toDouble(train("cont10")))
      .withColumn("cont11",toDouble(train("cont11")))
      .withColumn("cont12",toDouble(train("cont12")))
      .withColumn("cont13",toDouble(train("cont13")))
      .withColumn("cont14",toDouble(train("cont14")))
      .withColumn("loss",toDouble(train("loss")))
      .drop("cat96","cat101","cat102","cat103","cat104","cat105","cat106",
        "cat107","cat108","cat109","cat110","cat111","cat112","cat113","cat114",
        "cat115","cat116","cont10")

    val indexer:Array[org.apache.spark.ml.PipelineStage]  = trainDF.columns.filter(_.startsWith("cat")).map {
      cname =>
        new StringIndexer()
          .setInputCol(cname)
          .setOutputCol(s"${cname}_index")
    }

    val indexStages: Array[org.apache.spark.ml.PipelineStage] = indexer
    val indexed = new Pipeline().setStages(indexStages).fit(trainDF).transform(trainDF)
    //val model = pipeline.fit(trainDF).transform(trainDF)
    //model.show()
    //indexer
    val encoder = indexed.columns.filter(_.contains("_index")).map {
      cname =>
        new OneHotEncoder()
            .setInputCol(cname)
            .setOutputCol(s"${cname}_vec")
    }
    val encoded = new Pipeline().setStages(encoder).fit(indexed).transform(indexed)
    //encoded.show()
    val discretizer:Array[org.apache.spark.ml.PipelineStage] = encoded.columns.filter(_.startsWith("cont")).map {
      cname =>
        new QuantileDiscretizer()
          .setInputCol(cname)
          .setOutputCol(s"${cname}_qd")
          .setNumBuckets(5)
    }
    val discretized = new Pipeline().setStages(discretizer).fit(encoded).transform(encoded)
    //discretized.show()
    //val featureCols = Array("")
    val assembler = new VectorAssembler()
      .setInputCols(discretized.columns.filter(x => x.contains("_vec") || x.contains("_qd")))
      .setOutputCol("features")

    val featureCols = assembler.transform(discretized).select("id","features","loss")
    //featureCols.show()
    val Array(trainingData, testData) = featureCols.randomSplit(Array(0.7, 0.3))
    val lr = new LinearRegression()
        .setLabelCol("loss")
      .setMaxIter(10)
      .setRegParam(0.3)
      //.setElasticNetParam(0.8)
    val model = lr.fit(trainingData)
    val predictions = model.transform(testData)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("loss")
        .setMetricName("mae")
    val mae = evaluator.evaluate(predictions)
    println("MAE " + mae)

    /*val stages: Array[org.apache.spark.ml.PipelineStage] = /*indexer ++*/ encoder ++ discretizer

    val pipeline = new Pipeline().setStages(stages)
    val model = pipeline.fit(trainDF).transform(trainDF)
    model.show()*/
    //val pipeline = new Pipeline()
      //.setStages(Array(indexer.foreach(_.), encoder, bucketiser))
    //pipeline.fit(trainDF).transform(trainDF).show()*/
    //val indexed = indexer.map(_.fit(trainDF).transform(trainDF))

    //indexed
    //val encoded = encoder.foreach(_.transform(indexed))
    //val featureCols = Array("balance", "duration", "history", "purpose", "amount",
    //  "savings", "employment", "instPercent", "sexMarried",  "guarantors",
    //  "residenceDuration", "assets",  "age", "concCredit", "apartment",
    //  "credits",  "occupation", "dependents",  "hasPhone", "foreign" )
    //set the input and output column names
    //val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    //val processedCategory = encoder.transform()
    //val splitSeed = 5043
    //val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)
    spark.stop()
    //sc.stop()
  }
}