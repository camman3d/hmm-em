package com.joshmonson

object Main {
  def main(args: Array[String]) {

    val errThreshold = 0.1 // We stop when the log likelihood increases by less than this
    val data = DataGenerator.generate(Map(
      List(0,0,1)       -> 10,  //  TTS	    10
      List(0,0,1,1)     -> 15,  //  TTSS	  15
      List(0,0,1,0,1,1) -> 20,  //  TTSTSS  20
      List(0,1,1)       -> 15,  //  TSS	    15
      List(0,0,1,2)     -> 13,  //  TTSH	  13
      List(0,0,1,1,2)   -> 20,  //  TTSSH	  20
      List(0,0,1,1,1,2) -> 10,  //  TTSSSH	10
      List(0,1,1,0,2)   -> 5,   //  TSSTH	  5
      List(1,0,0,1)     -> 8,   //  STTS	  8
      List(1,0,0,1,2)   -> 8    //  STTSH	  8
    ))


    // Create the HMM we'll train
    var hmm = HiddenMarkovModel(3, 3).randomGuess(data)
    var ll = Double.NegativeInfinity
    var loop = true
    var i = 1

    while (loop) {
      val e = HiddenMarkovModel.eStep(hmm, data)
      val (newHmm, newLL) = HiddenMarkovModel.mStep(e)
      println(s"Iteration #$i, Log Likelihood: $newLL")

      val diff = newLL - ll
      if (diff < errThreshold) {
        loop = false
      }
      ll = newLL
      hmm = newHmm
      i += 1
    }

    println(hmm)
  }
}
