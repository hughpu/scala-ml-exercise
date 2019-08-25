package github.hughpu.ml.model

import breeze.linalg.{DenseMatrix, DenseVector}

trait Model[T] {
    def predict(x: DenseMatrix[T]): DenseVector[_]
    def predict_proba(x: DenseMatrix[T]): DenseVector[_]
}
