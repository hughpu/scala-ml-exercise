/*
test if the performance of the model run well
*/
import breeze.linalg.{DenseMatrix, DenseVector, convert}
import github.hughpu.ml.classifier.LogisticRegressor
import github.hughpu.ml.utility.{Metric, helper}
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class LogisticRegressionTests extends FunSuite with Matchers {

  test("An empty list should be empty") {
    val dataset = helper.readCsv("../dataset/IRIS.csv")
    val xArr = dataset.map(_.slice(0, dataset(0).length - 1)).map(_.map(_.toString.toDouble))
    val yArr = dataset.map(_(dataset(0).length - 1)).map {
      case "Iris-setosa" => 1.0
      case _ => 0.0
    }
    val x = DenseMatrix(xArr:_*)
    val y = DenseVector(yArr:_*)
    val (trainX, trainY, testX, testY) = helper.spliter().split(x, y, 0.7f)

    val glrModel = new LogisticRegressor().fit(trainX, trainY)

    val pred = glrModel.predict_proba(testX)
    val performance = Metric.roc(testY, pred)

    println(s"Performance of the model IRIS dataset is auc: $performance")

    performance should be >= 0.6f
  }
}
