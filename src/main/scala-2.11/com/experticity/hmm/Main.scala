package com.experticity.hmm

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by josh.monson on 4/13/16.
  */
object Main {
  def main(args: Array[String]) {

    val numSequences = 10
    val sequenceSize = 5

    // Create the HMM we'll use to create data
    val initial = DenseVector(0.38, 0.44, 0.18)
    val transitions = DenseMatrix(
      (0.4, 0.5, 0.1),
      (0.2, 0.2, 0.6),
      (0.25, 0.1, 0.65)
    )
    val emissions = DenseMatrix(
      (0.73, 0.11, 0.16),
      (0.32, 0.58, 0.1),
      (0.3, 0.3, 0.4))
    val hmmMaster = HiddenMarkovModel(HMMMeta(3, 3), initial, transitions, emissions)
    val data = (1 to numSequences).map(_ => hmmMaster.generateSequence(sequenceSize)).toList


    // Create the HMM we'll train
    var hmm = HiddenMarkovModel(2, 2).copy(
      initial = DenseVector(.85, .15),
      transitions = DenseMatrix((.3, .7), (.1, .9)),
      emissions = DenseMatrix((.4, .6), (.5, .5))
    )
    println(hmm)

    val obs = List(List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0), List(0,1,1,0),
      List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1),
      List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1), List(1,0,1))
    val foo = obs.map(seq => {
      val h = hmm.withSequence(seq)
      val a = h.alpha()
      val b = h.beta()
      val xi = h.xi(a, b)
      val gamma = h.gamma(a, xi)

      (xi, gamma)
    })
    val xis = foo.map(_._1)
    val gammas = foo.map(_._2)
    hmm = hmm.update(xis, gammas, obs)
    println(hmm)





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
