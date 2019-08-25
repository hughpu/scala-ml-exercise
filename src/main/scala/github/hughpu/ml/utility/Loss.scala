package github.hughpu.ml.utility

import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.mean

object Loss {
    def crossEntropy(logit: DenseVector[Double], lable: DenseVector[Double]): Double = {
        val pred = Activation.sigmoid(logit)
        mean(- (lable *:* log(pred) + (1.0 - lable) *:* log(1 - pred)))
    }
}
