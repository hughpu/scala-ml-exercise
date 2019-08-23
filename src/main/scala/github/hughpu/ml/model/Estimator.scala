package github.hughpu.ml.model

import breeze.linalg._

trait Estimator[T] {
    def fit(x: DenseMatrix[T], y: DenseVector[T]): Model[T]
    def gradient(X: DenseMatrix[T], y: DenseVector[T]): DenseVector[T]
    def compute(X: DenseMatrix[T]): DenseVector[T]
    def loss(x: DenseVector[T]): T
}
