#!/bin/bash

num=$(find . -type f -name '*.config')
echo $num


config_file=$(find . -type f -name '*.config')

. $config_file

echo $Q_mean
#. Case9_m047_predeformed.config


