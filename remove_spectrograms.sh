#!/bin/bash
echo Removing training spectrograms, cause you fucked up...
for d in ./Input_spectrogram/Training/*; do
  rm "$d";
  done

  echo Removing validation spectrograms, cause you fucked up...
  for d in ./Input_spectrogram/Validation/*; do
    rm "$d";
    done
