package com.experticity.hmm

import breeze.linalg.{DenseMatrix, DenseVector}
import com.experticity.hmm.HiddenMarkovModel.Observation

case class HiddenMarkovModel(initial: DenseVector[Double], transitions: DenseMatrix[Double], emissions: DenseMatrix[Double], sequence: List[Observation]) {
  def withSequence(sequence: List[Observation]) = copy(sequence = sequence)

}

object HiddenMarkovModel {

  case class State(name: Symbol)
  case class Observation(name: Symbol)

  implicit def toState(name: Symbol): State = State(name)
  implicit def toObservation(name: Symbol): Observation = Observation(name)


  def initialGuess(states: List[State], observations: List[Observation]) = {
    val stateP = 1.0 / states.size.toDouble
    val obserP = 1.0 / observations.size.toDouble

    val initial = DenseVector.fill(states.size, stateP)
    val transitions = DenseMatrix.fill(states.size, states.size)(stateP)
    val emissions = DenseMatrix.fill(states.size, observations.size)(obserP)

//    val initial: InitialStateTable = states
//      .map(s => (s, 1.0 / states.size.toDouble))
//      .toMap
//    val transitions: TransitionTable = states
//      .flatMap(s1 => states.map(s2 => (s1, s2)))
//      .map(t => (t, 1.0 / states.size.toDouble))
//      .toMap
//    val emissions: EmissionTable = states
//      .flatMap(s => observations.map(o => (o, s)))
//      .map(e => (e, 1.0 / observations.size.toDouble))
//      .toMap
    HiddenMarkovModel(initial, transitions, emissions, List())
  }

}