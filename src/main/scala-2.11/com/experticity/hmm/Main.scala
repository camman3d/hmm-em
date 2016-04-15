package com.experticity.hmm

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by josh.monson on 4/13/16.
  */
object Main {
  def main(args: Array[String]) {

    val numSequences = 100
    val sequenceSize = 50
    val numIterations = 2

    // Create the HMM we'll use to create data
    val initial = DenseVector(0.38, 0.44, 0.18)
    val transitions = DenseMatrix(
      (0.4, 0.5, 0.1),
      (0.02, 0.02, 0.96),
      (0.25, 0.1, 0.65)
    )
    val emissions = DenseMatrix(
      (0.9, 0.05, 0.05),
      (0.1, 0.8, 0.1),
      (0.2, 0.2, 0.6))
    val hmmMaster = HiddenMarkovModel(HMMMeta(3, 3), initial, transitions, emissions)
//    val data = (1 to numSequences).map(_ => hmmMaster.generateSequence(sequenceSize)).toList






    // Create the HMM we'll train
    var hmm = HiddenMarkovModel(2, 2).copy(
      initial = DenseVector(.85, .15),
      transitions = DenseMatrix((.3, .7), (.1, .9)),
      emissions = DenseMatrix((.4, .6), (.5, .5))
    )

    println(hmm)
//
    val data = List(List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0),
      List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1),
      List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1))

    (1 to numIterations).foreach(_ => {
      val e = HiddenMarkovModel.eStep(hmm, data)
      hmm   = HiddenMarkovModel.mStep(e)
      println(hmm)
    })

//    println(hmm)





//    data.map(_._2).foreach(d => {
//      println(hmm)
//      hmm = hmm.withSequence(d).emStep()
//    })

    //    (1 to 10).foreach(_ => {
    //      println(hmm)
    //      println("======================================================")
    //      hmm = hmm.emStep()
    //    })
    //    println(hmm)

  }
}
