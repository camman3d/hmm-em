package com.experticity.hmm

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.mutable.ListBuffer
import scala.util.Random

case class HMMMeta(numStates: Int, numObservations: Int)

case class HiddenMarkovModel(meta: HMMMeta, initial: DenseVector[Double], transitions: DenseMatrix[Double], emissions: DenseMatrix[Double], sequence: List[Int] = List()) {
  // To make it easier to match equations
  val (a, b, pi, y) = (transitions, emissions, initial, sequence)

  def generateSequence(length: Int) = {
    def next(v: DenseVector[Double]) = {
      val r = Random.nextDouble()
      (0 until v.length).find(i => {
        val low = v(0 until i).sum
        val high = low + v(i)
        r >= low && r < high
      }).get
    }

    val states = new ListBuffer[Int]()
    states += next(initial)
    (1 until length).foreach(i => {
      states += next(transitions(states(i-1),::).t)
    })

    val observations = states.map(state => {
      next(emissions(state,::).t)
    })

    (observations.toList, states.toList)
  }

  def withSequence(sequence: List[Int]) = copy(sequence = sequence)

  def initialGuess() = {
    val stateP = 1.0 / meta.numStates.toDouble
    val obserP = 1.0 / meta.numObservations.toDouble

    val initial = DenseVector.fill(meta.numStates, stateP)
    val transitions = DenseMatrix.fill(meta.numStates, meta.numStates)(stateP)
    val emissions = DenseMatrix.fill(meta.numStates, meta.numObservations)(obserP)
    copy(initial = initial, transitions = transitions, emissions = emissions)
  }

  def forT(f: Int => Unit) = sequence.indices.foreach(f)
  def `forT-1`(f: Int => Unit) = (0 until sequence.size-1).foreach(f)
  def sumT(f: Int => Double) = (0 until sequence.size).map(f).sum
  def `sumT-1`(f: Int => Double) = (0 until sequence.size-1).map(f).sum
  def fromT(f: Int => Unit) = sequence.indices.reverse.foreach(f)
  def forN(f: Int => Unit) = (0 until meta.numStates).foreach(f)
  def sumN(f: Int => Double) = (0 until meta.numStates).map(f).sum
  def forM(f: Int => Unit) = (0 until meta.numObservations).foreach(f)

  def alpha() = {
    val alpha = DenseMatrix.zeros[Double](meta.numStates, sequence.size)

    // Forward algorithm
    forT(t => {
      forN(i => {
        if (t == 0) alpha(i, t) = pi(i) * b(i, y(t))
        else        alpha(i, t) = b(i, y(t)) * sumN(j => alpha(j, t-1) * a(j, i))
      })
    })

    alpha
  }

  def beta() = {
    val beta = DenseMatrix.zeros[Double](meta.numStates, sequence.size)

    // Backward algorithm
    fromT(t => {
      forN(i => {
        if (t == sequence.size - 1) beta(i, t) = 1
        else                        beta(i, t) = sumN(j => beta(j, t+1) * a(i,j) * b(j, y(t+1)))
      })
    })

    beta
  }

  def gamma(alpha: DenseMatrix[Double], beta: DenseMatrix[Double]) = {
    val gamma = DenseMatrix.zeros[Double](meta.numStates, sequence.size)

    forT(t => {
      forN(i => {
        gamma(i, t) = (alpha(i,t) * beta(i,t)) / sumN(j => alpha(j,t) * beta(j,t))
//        gamma(i, t) = (alpha(i,t) * beta(i,t)) / sumN(j => alpha(j,t) * beta(j,t))
      })
    })

    gamma
  }

  def xi(alpha: DenseMatrix[Double], beta: DenseMatrix[Double]) = {
    val xi = (0 until meta.numStates).map(_ => DenseMatrix.zeros[Double](meta.numStates, sequence.size - 1)).toList

    val sum = sumN(k => alpha(k, meta.numStates - 1))
    `forT-1`(t => {
      forN(i => {
        forN(j => {
          xi(i)(j,t) = (alpha(i,t) * a(i,j) * beta(j,t+1) * b(j, y(t+1))) / sum
        })
      })
    })

    xi
  }

  def normalize(a: DenseMatrix[Double]) = {
    val normalized = a.copy
    forN(i => {
      val sum = sumN(j => a(i,j))
      forN(j => normalized(i,j) = a(i,j) / sum)
    })
    normalized
  }

  def update(gamma: DenseMatrix[Double], xi: List[DenseMatrix[Double]]) = {
    val newPi = gamma(::, 0)

    val newA = DenseMatrix.zeros[Double](meta.numStates, meta.numStates)
    forN(i => {
      forN(j => {
        newA(i,j) = `sumT-1`(t => xi(i)(j,t)) / `sumT-1`(t => gamma(i,t))
      })
    })

    val newB = DenseMatrix.zeros[Double](meta.numStates, meta.numObservations)
    forN(i => {
      forM(v => {
        newB(i, v) = sumT(t => if (v == y(t)) gamma(i, t) else 0) / sumT(t => gamma(i, t))
      })
    })

    copy(initial = newPi, transitions = normalize(newA), emissions = newB)
  }

  def emStep() = {
    val _a = alpha()
    val _b = beta()
    val _g = gamma(_a, _b)
    val _x = xi(_a, _b)
    update(_g, _x)
  }

  override def toString: String =
    s"""
      |Initial:
      |${initial}
      |
      |Transitions:
      |${transitions}
      |
      |Emissions:
      |${emissions}
    """.stripMargin
}

object HiddenMarkovModel {

  def apply(numStates: Int, numObservations: Int): HiddenMarkovModel =
    HiddenMarkovModel(HMMMeta(numStates, numObservations), DenseVector(), DenseMatrix.zeros[Double](1,1), DenseMatrix.zeros[Double](1,1))

}