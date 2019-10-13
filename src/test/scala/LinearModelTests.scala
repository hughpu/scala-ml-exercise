/*
test if the performance of the model run well
*/
import breeze.linalg.{DenseMatrix, DenseVector}
import github.hughpu.ml.classifier.LogisticRegressor
import github.hughpu.ml.regressor.RidgeRegressor
import github.hughpu.ml.utility.{Metric, helper}
import org.scalatest._

class LinearModelTests extends FunSuite with Matchers {

    test("Logistic Regression Test") {
        val dataset = helper.readCsv("/dataset/IRIS.csv", deBug = true)
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

        performance should be >= 0.9f
    }

    test("Ridge Regression Test") {
        val dataset = helper.readCsv("/dataset/RegressionDataSet.csv", deBug = true)
        val xArr = dataset.map(_.slice(0, dataset(0).length - 1)).map(_.map(_.toString.toDouble))
        val yArr = dataset.map(_(dataset(0).length - 1)).map(_.toString.toDouble)
        val x = DenseMatrix(xArr:_*)
        val y = DenseVector(yArr:_*)
        val (trainX, trainY, testX, testY) = helper.spliter().split(x, y, 0.7f)

        val lrModel = new RidgeRegressor(0.5).fit(trainX, trainY)

        val pred = lrModel.predict(testX)
        val performance = Metric.rSquare(testY, pred)

        println(s"Performance of the model regression dataset is rSquare: $performance")

        performance should be >= 0.9
    }
}
