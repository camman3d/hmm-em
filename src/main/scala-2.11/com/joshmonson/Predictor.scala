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

    val hmm = HiddenMarkovModel(HMMMeta(3,3),
      DenseVector(0.9999, 0.0001, 0),
      DenseMatrix(
        (0.5063339276825024, 0.4926208117960329, 0.00104526052146474),
        (0.0013076156737276998, 0.9986888372446822, 3.5470815901379086E-6),
        (8.963080476676163E-4, 0.5566707816592946, 0.44243291029303766)
      ),
      DenseMatrix(
        (0.8567586747102625, 0.1432413206892567, 4.600480793318188E-9),
        (0.12767913461534275, 0.6947352591397361, 0.17758560624492117),
        (0.25456708124705607, 0.7359238066458773, 0.00950911210706672)
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
