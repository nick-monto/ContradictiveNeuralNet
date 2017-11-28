#!/bin/bash
echo Removing training audio, cause you fucked up...
for d in ./Input_audio/Train/*; do
  rm "$d";
  done

echo Removing validation audio, cause you fucked up...
for d in ./Input_audio/Val/*; do
  rm "$d";
  done

echo Removing Test audio, cause you fucked up...
  for d in ./Input_audio/Test/*; do
    rm "$d";
    done
