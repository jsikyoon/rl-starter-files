#!/bin/bash

# remove unity_envs directory
rm -rf unity_envs

# make directory
mkdir unity_envs
cd unity_envs/

# GridWorld
wget https://www.dropbox.com/s/gh8z8f0z90f4nvq/GridWorld.zip
unzip GridWorld.zip
chmod 755 -R GridWorld
rm -rf __MACOSX

# OrderSeq4BallsDist4
#wget https://www.dropbox.com/s/c4u378cptced57m/AreaLSizeL4BallFixPosDist4FgNoResetPos.zip
#unzip AreaLSizeL4BallFixPosDist4FgNoResetPos.zip
#chmod 755 -R AreaLSizeL4BallFixPosDist4FgNoResetPos

# OrderSeq4BallsDist3
wget https://www.dropbox.com/s/c4u378cptced57m/AreaLSizeL4BallFixPosDist3FgNoResetPos.zip
unzip AreaLSizeL4BallFixPosDist3FgNoResetPos.zip
chmod 755 -R AreaLSizeL4BallFixPosDist3FgNoResetPos

# OrderSeq4BallsDist2
wget https://www.dropbox.com/s/sqi9ir7bnkle9ff/AreaLSizeL4BallFixPosDist2FgNoResetPos.zip
unzip AreaLSizeL4BallFixPosDist2FgNoResetPos.zip
chmod 755 -R AreaLSizeL4BallFixPosDist2FgNoResetPos

# OrderSeq5BallsDist2
wget https://www.dropbox.com/s/9wzo4eu4qmhucnm/AreaLSizeL5BallFixPosDist2FgNoResetPos.zip
unzip AreaLSizeL5BallFixPosDist2FgNoResetPos.zip
chmod 755 -R AreaLSizeL5BallFixPosDist2FgNoResetPos
