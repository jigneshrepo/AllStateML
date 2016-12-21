name := "AllState"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.2" % "provided",
  "org.apache.spark" % "spark-sql_2.11" % "2.0.2",
  "org.apache.spark" % "spark-mllib_2.11" % "2.0.2"
)

    