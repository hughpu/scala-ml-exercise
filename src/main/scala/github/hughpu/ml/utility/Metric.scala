package github.hughpu.ml.utility

import breeze.linalg.DenseVector

object Metric {
    def roc(label: DenseVector[Double], pred: DenseVector[Double]): Float = {
        val labelArr = label.toArray
        val predArr = label.toArray

        // the vertical step of TPR axis, which equal to 1 / |P|, |P| is the length of true sample
        val vstep = 1.0 / labelArr.count(_ == 1)
        // the horizontal step of FPR axis, which should equal to 1 / |N|， |N| is the length of negative sample
        val hstep = 1.0 / labelArr.count(_ == 0)

        // match the label and pred values, sort them for computing roc
        val zippedArr = predArr.zip(labelArr)
        val sortedPairArr = zippedArr.sortWith(_._1 > _._1)

        var height = 0.0
        var roc = 0.0
        for(p <- sortedPairArr) {
            if(p._2 == 1) {
                height += vstep
            } else {
                roc += height * hstep
            }
        }
        roc.toFloat
    }

    //def f1Score(label: DenseVector[Double], pred: DenseVector[Double]): Float
    //
    //def accuracy(label: DenseVector[Double], pred: DenseVector[Double]): Float
    //
    //def rSquare(label: DenseVector[Double], pred: DenseVector[Double]): Float
    //
    //def rmse(label: DenseVector[Double], pred: DenseVector[Double]): Float

}
