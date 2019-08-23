package github.hughpu.ml.classifier;

import breeze.linalg.{DenseMatrix, DenseVector}
import github.hughpu.ml.model.{Estimator, Model}
import github.hughpu.ml.utility.{Activation, Loss}

class LogisticRegressor(iters: Int = 1000, shrinkage: Double = 1e-2) extends Estimator[Double] {
    var params: DenseVector[Double] = null

    def fit(x: DenseMatrix[Double], y: DenseVector[Double]): LogisticRegressionModel = {

        require(x.rows == y.length, "[ERROR] The size of x and y should match!")

        val X = DenseMatrix.vertcat(x, DenseMatrix.fill(x.rows, 1){1.0})
        params = DenseVector.zeros(X.cols)

        for (i <- 0 to iters) {
            val grads = gradient(X, y)
            params -= grads *:* shrinkage
            if (i % 100 == 0) {
                val lossVal = Loss.crossEntropy(X dot params, y)
                println(s"[Info] Step: $i, CrossEntropy loss: $lossVal")
            }
        }
        new LogisticRegressionModel(params)
    }

    def gradient(X: DenseMatrix[Double], y: DenseVector[Double]): DenseVector[Double] = {
        (compute(X) - y) dot X /:/ X.rows
    }

    def compute(X: DenseMatrix[Double]): DenseVector[Double] = {
        val logit = X dot params
        Activation.sigmoid(logit)
    }

}

class LogisticRegressionModel(params: DenseVector[Double]) extends Model[Double] {
    def predict(x: DenseMatrix[Double]): DenseVector[Int] = {
        val pred = predict_proba(x)
        pred.mapValues {
            case x if x > 0.5 => 1
            case x if x <= 0.5 => 0
        }
    }

    def predict_proba(x: DenseMatrix[Double]): DenseVector[Double] = {
        val X = DenseMatrix.vertcat(x, DenseMatrix.fill(x.rows, 1){1.0})
        Activation.sigmoid(X dot params)
    }
}
