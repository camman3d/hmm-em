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

//    (observations.toList, states.toList)
    observations.toList
  }

  def withSequence(sequence: List[Int]) = copy(sequence = sequence)

  def randomGuess(data: List[List[Int]]) = {
    val initial = DenseVector((0 until meta.numStates).map(i => data.count(_.head == i)).map(_.toDouble).toVector: _*)
    initial /= data.size.toDouble

    val stateP = 1.0 / meta.numStates.toDouble
    val obserP = 1.0 / meta.numObservations.toDouble

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
  def forN2(f: Int => Unit) = (0 until meta.numStates*meta.numStates).foreach(f)
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

  def xi(alpha: DenseMatrix[Double], beta: DenseMatrix[Double]) = {
    val xi = DenseMatrix.zeros[Double](meta.numStates * meta.numStates, sequence.size - 1)

    val sum = sumN(k => alpha(k, sequence.size - 1))
    `forT-1`(t => {
      forN(i => {
        forN(j => {
          xi(i*2 + j, t) = (alpha(i,t) * a(i,j) * beta(j,t+1) * b(j, y(t+1))) / sum
        })
      })
    })

    xi
  }

  def gamma(alpha: DenseMatrix[Double], xi: DenseMatrix[Double]) = {
    val gamma = DenseMatrix.zeros[Double](meta.numStates, sequence.size)

    `forT-1`(t => {
      forN(i => {
        gamma(i, t) = sumN(j => xi(i*2 + j, t))
        //        gamma(i, t) = (alpha(i,t) * beta(i,t)) / sumN(j => alpha(j,t) * beta(j,t))
      })
    })
    val sum = sumN(k => alpha(k, sequence.size - 1))
    forN(i => {
      gamma(i, sequence.size - 1) = alpha(i, sequence.size - 1) / sum
    })

    gamma
  }



//  def emStep() = {
//    val _a = alpha()
//    val _b = beta()
//    val _g = gamma(_a, _b)
//    val _x = xi(_a, _b)
//    update(_g, _x)
//  }

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

  case class EStepOutput(hmm: HiddenMarkovModel, sequences: List[List[Int]], xis: List[DenseMatrix[Double]], gammas: List[DenseMatrix[Double]])

  def eStep(hmm: HiddenMarkovModel, sequences: List[List[Int]]) = {
    val data = sequences.map(seq => {
      val h = hmm.withSequence(seq)
      val a = h.alpha()
      val b = h.beta()
      val xi = h.xi(a, b)
      val gamma = h.gamma(a, xi)
      (xi, gamma)
    })
    EStepOutput(hmm, sequences, data.map(_._1), data.map(_._2))
  }

  def mStep(eout: EStepOutput) = {
    val (hmm, sequences, xis, gammas) = (eout.hmm, eout.sequences, eout.xis, eout.gammas)
    val (meta, forN, forN2, sumN, forM) = (hmm.meta, hmm.forN _, hmm.forN2 _, hmm.sumN _, hmm.forM _)

    // The new start is the average
    val newPi = {
      val pi = gammas.map(m => m(::, 0)).reduce((m1, m2) => m1 + m2)
      pi / pi.sum
    }

    val newA = {
      val v = xis.map(m => {
        val v = DenseVector.zeros[Double](meta.numStates * meta.numStates)
        forN2(i => v(i) = m(i,::).t.sum)
        v
      }).reduce((m1,m2) => m1 + m2)

      val a = DenseMatrix.zeros[Double](meta.numStates, meta.numStates)
      forN(i => {
        forN(j => {
          a(i,j) = v(2*i + j) / sumN(k => v(2*i + k))
        })
      })
      a
    }

    val newB = {
      val b = gammas.zip(sequences).map(g => {
        val (m, seq) = g
        val b = DenseMatrix.zeros[Double](meta.numStates, meta.numObservations)
        forN(i => {
          forM(j => {
            b(i,j) = m(i,::).t.toScalaVector() // gamma row
              .zip(seq)
              .filter(d => d._2 == j)
              .map(_._1).sum
          })
        })
        b
      }).reduce((m1,m2) => m1 + m2)

      forN(i => {
        val sum = b(i,::).t.sum
        forM(j => {
          b(i,j) /= sum
        })
      })
      b
    }

    hmm.copy(initial = newPi, transitions = newA, emissions = newB)
  }

}