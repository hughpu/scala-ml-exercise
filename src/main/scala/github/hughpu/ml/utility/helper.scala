package github.hughpu.ml.utility

import java.io.InputStream

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.io.Source
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Random

object helper {
    def readCsv(path: String, delimiter: Char = ',', skipHead: Boolean = true, deBug: Boolean = false): Array[Array[Any]] = {
        val fileSource = {
            if(!deBug) {
                Source.fromFile(path)
            } else {
                val is: InputStream = getClass.getResourceAsStream(path)
                Source.fromInputStream(is)
            }
        }
        val lines = fileSource.getLines
        val arrBuf = ArrayBuffer.empty[Array[Any]]
        if(skipHead) {
            lines.next
        }
        for(l <- lines) {
            val line = ArrayBuffer.empty[Any]
            l.split(delimiter).foreach {
                line += _
            }
            arrBuf += line.toArray
        }
        fileSource.close
        arrBuf.toArray
    }

    def spliter() = new Spliter


}

class Spliter {
    def split(x: DenseMatrix[Double], ratio: Float): (DenseMatrix[Double], DenseMatrix[Double]) = {
        val firstBuf = ListBuffer.empty[DenseVector[Double]]
        val secondBuf = ListBuffer.empty[DenseVector[Double]]
        for(i <- 0 until x.rows) {
            if(Random.nextFloat() <= ratio) {
                firstBuf += x(i, ::).t
            } else {
                secondBuf += x(i, ::).t
            }
        }
        (DenseMatrix(firstBuf.toList:_*), DenseMatrix(secondBuf.toList:_*))
    }

    def split(x: DenseVector[Double], ratio: Float): (DenseVector[Double], DenseVector[Double]) = {
        val firstBuf = ListBuffer.empty[Double]
        val secondBuf = ListBuffer.empty[Double]
        for(i <- 0 until x.length) {
            if(Random.nextFloat() <= ratio) {
                firstBuf += x(i)
            } else {
                secondBuf += x(i)
            }
        }
        (DenseVector(firstBuf.toList:_*), DenseVector(secondBuf.toList:_*))
    }

    def split(x: DenseMatrix[Double], y: DenseVector[Double], ratio: Float): (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
        val firstBuf1 = ListBuffer.empty[DenseVector[Double]]
        val firstBuf2 = ListBuffer.empty[Double]
        val secondBuf1 = ListBuffer.empty[DenseVector[Double]]
        val secondBuf2 = ListBuffer.empty[Double]
        for(i <- 0 until y.length) {
            if(Random.nextFloat() <= ratio) {
                firstBuf1 += x(i, ::).t
                firstBuf2 += y(i)
            } else {
                secondBuf1 += x(i, ::).t
                secondBuf2 += y(i)
            }
        }
        (DenseMatrix(firstBuf1.toList:_*), DenseVector(firstBuf2.toList:_*), DenseMatrix(secondBuf1.toList:_*), DenseVector(secondBuf2.toList:_*))
    }
}
