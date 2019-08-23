package github.hughpu.ml.utility

import breeze.linalg.DenseVector
import breeze.numerics.log

object Loss {
    def crossEntropy(logit: DenseVector[Double], lable: DenseVector[Double]) = {
        lable *:* log(Activation.sigmoid(logit))
    }
}
