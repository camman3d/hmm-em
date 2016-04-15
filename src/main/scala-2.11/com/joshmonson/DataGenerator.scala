package com.joshmonson

/**
  * Created by josh.monson on 4/15/16.
  */
object DataGenerator {

  def generate(dataDesc: Map[List[Int], Int]) = {
    dataDesc.flatMap(entry => (0 until entry._2).map(_ => entry._1)).toList
  }

}
