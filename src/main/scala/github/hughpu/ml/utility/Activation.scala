package github.hughpu.ml.utility

import breeze.numerics._
import breeze.linalg._

object Activation {
    def sigmoid(X: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 /:/ (exp(-X) + 1.0)
    }

    def softmax(x: DenseVector[Double]): DenseVector[Double] = {
        val maxVal = max(x)
        val expX = exp(x - maxVal)
        val sumExpX = sum(expX)
        return expX / sumExpX
    }
}
