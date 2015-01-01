#!/bin/bash

# Define the input files
RGB_NET=matnet_rgb.partial
JOINT_NET=matnet_joints.partial
TRAIN_NET_HEAD=mattrain_head.partial
TRAIN_NET_FOOT=mattrain_foot.partial
TEST_NET_HEAD=matforward_head.partial
TEST_NET_FOOT=matforward_foot.partial
ROBOT_A_HEAD=controllera_head.partial
ROBOT_A_FOOT=controllera_foot.partial
ROBOT_B_HEAD=controllerb_head.partial
ROBOT_B_FOOT=controllerb_foot.partial

# Define the output files
TRAIN_NET=mattrain_val_rgb.prototxt
TEST_NET=matforward_rgb.prototxt
ROBOT_A=controller_forwarda.prototxt
ROBOT_B=controller_forwardb.prototxt

# Print the file names
echo "Assembling network definitions"
echo "The following files are read to get network defintions: $RGB_NET $JOINT_NET"
echo "The following files contain headers: $TRAIN_NET_HEAD $TEST_NET_HEAD $ROBOT_A_HEAD $ROBOT_B_HEAD"
echo "The following files contain footers: $TRAIN_NET_FOOT $TEST_NET_FOOT $ROBOT_A_FOOT $ROBOT_B_FOOT"
echo "The following files will be created: $TRAIN_NET $TEST_NET $ROBOT_A $ROBOT_B"

# Assemble the files
cat $TRAIN_NET_HEAD $RGB_NET $JOINT_NET $TRAIN_NET_FOOT > $TRAIN_NET
cat $TEST_NET_HEAD $RGB_NET $JOINT_NET $TEST_NET_FOOT > $TEST_NET
cat $ROBOT_A_HEAD $RGB_NET $ROBOT_A_FOOT > $ROBOT_A
cat $ROBOT_B_HEAD $JOINT_NET $ROBOT_B_FOOT > $ROBOT_B

# Copy the files to the robot
echo "Copying network to brett1"
scp $ROBOT_A $ROBOT_B svlevine@brett1:~/
cp $ROBOT_A ~/
cp $ROBOT_B ~/


