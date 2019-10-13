package github.hughpu.ml.regressor

import breeze.linalg.{DenseMatrix, DenseVector, inv}

class RidgeRegressor(alpha: Double) {
    var params: DenseVector[Double] = _

    def fit(x: DenseMatrix[Double], y: DenseVector[Double]): RidgeRegressionModel = {
        params = inv(x.t * x + (alpha * DenseMatrix.eye[Double](x.cols))) * x.t * y
        new RidgeRegressionModel(params)
    }
}

class RidgeRegressionModel(params: DenseVector[Double]) {
    def predict(x: DenseMatrix[Double]): DenseVector[Double] = {
        x * params
    }
}
