
#! /bin/bash
THISDIR=`pwd`
export DEEPTAU=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$DEEPTAU
cd /afs/cern.ch/user/j/jkiesele/work/DeepLearning/FCChh/DeepJetCore
if command -v nvidia-smi > /dev/null
then
        source gpu_env.sh
else
        source lxplus_env.sh
fi
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
#export PYTHONPATH=`pwd`/modules/datastructures:$PYTHONPATH
export LD_LIBRARY_PATH=~/.lib:$LD_LIBRARY_PATH
export PATH=$DEEPTAU/scripts:$PATH
