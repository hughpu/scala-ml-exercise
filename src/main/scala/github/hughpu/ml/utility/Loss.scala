package github.hughpu.ml.utility

import breeze.linalg.DenseVector
import breeze.numerics.log

object Loss {
    def crossEntropy(logit: DenseVector[Double], lable: DenseVector[Double]) = {
        val pred = Activation.sigmoid(logit)
        - (lable *:* log(pred) + (1.0 - lable) *:* log(1 - pred))
    }
}
