package com.experticity.hmm

import com.experticity.hmm.HiddenMarkovModel.{Observation, State}

/**
  * Created by josh.monson on 4/13/16.
  */
object Main {
  def main(args: Array[String]) {

    val states = List[State](
      'WantToLearn,
      'WantToBuy
    )

    val observations = List[Observation](
      'Profile,
      'Training,
      'Store
    )

    val sequence = List[Observation]('Profile, 'Profile, 'Training, 'Profile, 'Store)

    val hmm = HiddenMarkovModel.initialGuess(states, observations).withSequence(sequence)
    println(hmm)

  }
}
