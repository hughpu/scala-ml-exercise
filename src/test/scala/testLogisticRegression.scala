/*
test if the performance of the model run well
*/
import breeze.linalg.{DenseMatrix, convert}
import github.hughpu.ml.classifier.LogisticRegressor
import github.hughpu.ml.utility.{Metric, helper}
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatestplus.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ListSuite extends FunSuite with Matchers {

  test("An empty list should be empty") {
    val dataset = helper.readCsv("../dataset/IRIS.csv")
    val x = convert(dataset(::, 0 to (dataset.cols - 2)), Double)
    val y = dataset(::, dataset.cols - 1).mapValues {
      case "Iris-setosa" => 1.0
      case _ => 0.0
    }
    val (trainX, trainY, testX, testY) = helper.Spliter().split(x, y, 0.7f)

    val glrModel = new LogisticRegressor().fit(trainX, trainY)

    val pred = glrModel.predict_proba(testX)
    val performance = Metric.roc(testY, pred)

    println(s"Performance of the model IRIS dataset is auc: $performance")

    performance >= 0.6 should be (true)
  }
}
