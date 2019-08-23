package github.hughpu.ml.utility

import breeze.linalg.DenseMatrix

import scala.io.Source._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

class helper {
    def readCsv(path: String, delimiter: Char): DenseMatrix[Any] = {
        val lines = fromFile(path).getLines
        val listBuf = ListBuffer.empty[Array[Any]]
        for(l <- lines) {
            val line = ArrayBuffer.empty[Any]
            l.split(delimiter).foreach {
                line += _
            }
            listBuf += line.toArray
        }
        DenseMatrix(listBuf.toList:_*)
    }
}
