package com.joshmonson

import breeze.linalg.{DenseMatrix, DenseVector}


object Predictor {

  def predict(alpha: DenseMatrix[Double]) = {
    (0 until alpha.cols).map(i => {
      val col = alpha(::, i)
      col.toScalaVector().zipWithIndex.maxBy(_._1)._2
    })
  }

  def main(args: Array[String]) {

    val hmm = HiddenMarkovModel(HMMMeta(3, 3),
      DenseVector(0.9999, 0.0001, 0),
      DenseMatrix(
        (0.4405865489331041, 0.5564547764564678, 0.002958674610428116),
        (0.002593914919173562, 0.9973927900956003, 1.3294985226138694E-5),
        (0.0021272430518693808, 0.6144066464221304, 0.38346611052600027)
      ),
      DenseMatrix(
        (0.8764603582173833, 0.12353964178255142, 6.533308331730339E-14),
        (0.1936026321075297, 0.648417799200184, 0.15797956869228622),
        (0.3421816171010765, 0.657731341335911, 8.704156301250849E-5)
      ),
      List(0, 0, 0, 1, 1, 2)
    )

    val alpha = hmm.alpha()
    println("Forward algorithm:")
    println(alpha.toString(10, 200))

    val prediction = predict(alpha)
    println("\nMost likely states for sequence:")
    println(prediction)

    val next = hmm.transitions(prediction.last, ::).t.toScalaVector().zipWithIndex.maxBy(_._1)._2
    println("\nMost likely next state:")
    println(next)

  }
}
