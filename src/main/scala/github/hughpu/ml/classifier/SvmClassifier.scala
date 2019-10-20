package github.hughpu.ml.classifier

import breeze.linalg.{DenseMatrix, DenseVector, norm}
import breeze.numerics._
import scala.util.control.Breaks._
import scala.math.{max, min}
import scala.math
import scala.collection.mutable._
import github.hughpu.ml.model.Estimator

class SvmClassifier(kernelMethod: String, softMargin: Double){
    require(softMargin > 0, "The regular coefficient to soft margin should be larger than 0")

    var lambdas: DenseVector[Double] = _

    var bias: Double = 0.0

    var preds: DenseVector[Double] = _
    var errors: DenseVector[Double] = _



    def kernelWithX(x: Int, y: Int, X: DenseMatrix[Double], method: String): Double = {
        SvmClassifier.kernel(X(x, ::).t, X(y, ::).t, method)
    }

    def fit(X: DenseMatrix[Double], Y: DenseVector[Double]): SvmModel = {

        lambdas = DenseVector.zeros[Double](X.rows)
        preds = DenseVector.zeros[Double](X.rows)
        errors = preds - Y

        // SMO
        breakable {
            while(true) {
                // find sample 1 and sample 2
                val samples = findSamplePair(Y)
                if(samples._1 == -1) {
                    break
                }

                // get new lambda1 and lambda2
                val eta = kernelWithX(samples._1, samples._1, X, kernelMethod) + kernelWithX(samples._2, samples._2, X, kernelMethod) - 2 * kernelWithX(samples._1, samples._2, X, kernelMethod)
                var newLambda2 = lambdas(samples._2) + Y(samples._2) * (errors(samples._1) - errors(samples._2)) / eta
                newLambda2 = clip(newLambda2, samples._1, samples._2, Y)
                val kesigh = lambdas(samples._1) * Y(samples._1) + lambdas(samples._2) * Y(samples._2)
                val newLambda1 = kesigh - newLambda2


                // update bias and errors
                if(isSupportVector(newLambda1)) {
                    bias = -errors(samples._1) - Y(samples._1) * kernelWithX(samples._1, samples._1, X, kernelMethod) * (newLambda1 - lambdas(samples._1)) - Y(samples._2) * kernelWithX(samples._2, samples._1, X, kernelMethod) * (newLambda2 - lambdas(samples._2)) + bias
                } else if(isSupportVector(newLambda2)) {
                    bias = -errors(samples._2) - Y(samples._1) * kernelWithX(samples._1, samples._2, X, kernelMethod) * (newLambda1 - lambdas(samples._1)) - Y(samples._2) * kernelWithX(samples._2, samples._2, X, kernelMethod) * (newLambda2 - lambdas(samples._2)) + bias
                } else {
                    val bias1 = -errors(samples._1) - Y(samples._1) * kernelWithX(samples._1, samples._1, X, kernelMethod) * (newLambda1 - lambdas(samples._1)) - Y(samples._2) * kernelWithX(samples._2, samples._1, X, kernelMethod) * (newLambda2 - lambdas(samples._2)) + bias
                    val bias2 = -errors(samples._2) - Y(samples._1) * kernelWithX(samples._1, samples._2, X, kernelMethod) * (newLambda1 - lambdas(samples._1)) - Y(samples._2) * kernelWithX(samples._2, samples._2, X, kernelMethod) * (newLambda2 - lambdas(samples._2)) + bias
                    bias = (bias1 + bias2) / 2.0
                }

                lambdas(samples._1) = newLambda1
                lambdas(samples._2) = newLambda2

                updatePredsAndErrors(X, Y)
            }
        }

        // check if it satisfied condition sum(lambdas * y) == 0
        require(lambdas.t * Y <= 1e-7 && lambdas.t * Y >= -1e-7, "Failed to build the SVM model")
        // check if it satisfied other KKT conditions
        var satisfiedKKT = true
        breakable {
            lambdas.toArray.zipWithIndex.foreach(x => {
                if(checkKKT(x._2, Y) > 1e-7) {
                    satisfiedKKT = false
                    break
                }
            })
        }
        require(satisfiedKKT, "The result can not satisfied KKT conditions, failed to build the SVM model!")

        // generate the model with support vectors
        val supportVectorIndexes = lambdas.toArray.zipWithIndex.filter {
            x => isSupportVector(x._1)
        }.map(_._2)
        val supportVectors = ArrayBuffer[DenseVector[Double]]()
        val supportCoefficients = ArrayBuffer[Double]()
        supportVectorIndexes.foreach(idx => {
            supportVectors += X(idx, ::).t
            supportCoefficients += Y(idx) * lambdas(idx)
        })
        new SvmModel(supportVectors.toArray, supportCoefficients.toArray, bias, kernelMethod)
    }

    def updatePredsAndErrors(X: DenseMatrix[Double], Y: DenseVector[Double]): Unit = {
        val supportVectorIndexes: Array[Int] = lambdas.toArray.zipWithIndex.filter {
            x => isSupportVector(x._1)
        }.map(_._2)

        errors.toArray.zipWithIndex.foreach {
            x => {
                val y_lambda_k: Array[Double] = supportVectorIndexes.map(j => {
                    Y(j) * lambdas(j) * kernelWithX(j, x._2, X, kernelMethod)
                })
                preds(x._2) = y_lambda_k.sum + bias
                errors(x._2) = preds(x._2) - Y(x._2)
            }
        }
    }

    def isSupportVector(lambda: Double): Boolean = {
        if(lambda > 1e-7 && lambda < (softMargin - 1e7)) {
            true
        } else {
            false
        }
    }

    def findSamplePair(Y: DenseVector[Double]): (Int, Int) = {
        var didNotFindSample = true
        var ans: (Int, Int) = (-1, -1)

        // traverse support vectors to find the pair violate KKT most
        var maxError = 0.0
        lambdas.toArray.zipWithIndex.foreach {
            case x if isSupportVector(x._1) && checkKKT(x._2, Y) > 1e-7 =>
                val tmpPair = findSample2(x._2)
                val error = checkKKT(x._2, Y)
                if(error > maxError) {
                    maxError = error
                    ans = tmpPair
                }

            case _ => {}
        }
        if(ans._1 != -1) return ans

        // traverse vectors to find the pair violate KKT most
        maxError = 0.0
        lambdas.toArray.zipWithIndex.foreach {
            case x if !isSupportVector(x._1) && checkKKT(x._2, Y) > 1e-7 =>
                val tmpPair = findSample2(x._2)
                val error = checkKKT(x._2, Y)
                if(error > maxError) {
                    maxError = error
                    ans = tmpPair
                }

            case _ => {}
        }
        if(ans._1 != -1) return ans

        // traverse vectors
        var expectChange = 1e-3
        lambdas.toArray.zipWithIndex.foreach {
            case x if isSupportVector(x._1) & checkKKT(x._2, Y) <= 1e-7 =>
                val tmpPair = findSample2(x._2)
                val change = math.abs(errors(tmpPair._1) - errors(tmpPair._2))
                if(change > expectChange) {
                    ans = tmpPair
                    expectChange = change
                }
            case _ => {}
        }
        if(ans._1 != -1) return ans

        // traverse vectors
        expectChange = 1e-3
        lambdas.toArray.zipWithIndex.foreach {
            case x if !isSupportVector(x._1) & checkKKT(x._2, Y) <= 1e-7 =>
                val tmpPair = findSample2(x._2)
                val change = math.abs(errors(tmpPair._1) - errors(tmpPair._2))
                if(change > expectChange) {
                    ans = tmpPair
                    expectChange = change
                }
            case _ => {}
        }
        if(ans._1 != -1) return ans

        ans
    }

    def findSample2(i: Int): (Int, Int) = {
        val sortedErrorAndIndex = abs(errors - errors(i)).toArray.zipWithIndex.sortBy(-_._1).take(1)
        if(sortedErrorAndIndex(0)._1 > 1e-3) {
            (i, sortedErrorAndIndex(0)._2)
        } else {
            (-1, -1)
        }
    }

    def checkKKT(index: Int, Y:DenseVector[Double]): Double = {
        index match {
            case x if lambdas(x) <= 1e-7 => max(1.0 - preds(index) * Y(index), 0.0)
            case x if lambdas(x) <= softMargin - 1e-7 => math.abs(preds(index) * Y(index) - 1.0)
            case x => max(preds(x) * Y(x) - 1.0, 0.0)
        }
    }

    def clip(lambda: Double, sample1Idx: Int, sample2Idx: Int, Y: DenseVector[Double]): Double = {
        var L = 0.0
        var H = 0.0
        if(Y(sample1Idx) == Y(sample2Idx)) {
            L = max(lambdas(sample1Idx) + lambdas(sample2Idx) - softMargin, 0)
            H = min(lambdas(sample1Idx) + lambdas(sample2Idx), softMargin)
        } else {
            L = max(lambdas(sample2Idx) - lambdas(sample1Idx), 0)
            H = min(lambdas(sample2Idx) - lambdas(sample2Idx) + softMargin, softMargin)
        }

        min(max(L, lambda), H)
    }


}

object SvmClassifier {
    def rbf(x: DenseVector[Double], y: DenseVector[Double], gamma: Double): Double = {
        require(gamma > 0, "The bandwidth for rbf should be larger than 0")
        val diffVec = x - y
        val diffVal = diffVec.t * diffVec
        exp(-diffVal / gamma)
    }

    def kernel(x: DenseVector[Double], y: DenseVector[Double], method: String): Double = {
        method match {
            case "rbf" => rbf(x, y, 1.0)
            case "linear" => x.t * y
            case _ => throw new Exception(s"kernel method: $method is not support!")
        }
    }
}

class SvmModel(supportVectors: Array[DenseVector[Double]], supportVectorCoefficients: Array[Double], bias: Double, kernelMethod: String) {
    require(supportVectorCoefficients.length == supportVectors.length, "support vectors and support vector lambdas not match!")
    def predictDists(X: DenseMatrix[Double]): DenseVector[Double] = {
        val dists = ArrayBuffer[Double]()
        for(i <- 0 until X.rows) {
            var dist = 0.0
            for(j <- 0 until supportVectors.length) {
                dist += supportVectorCoefficients(j) * SvmClassifier.kernel(supportVectors(j), X(i, ::).t, kernelMethod)
            }
            dists += dist + bias
        }
        DenseVector(dists:_*)
    }

    def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
        val dists = predictDists(X)
        dists(dists >:> 0.0) = 1.0
        dists(dists <:< 0.0) = -1.0
        dists(dists :== 0.0) = -1.0
        dists
    }

}
