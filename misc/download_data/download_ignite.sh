#!/bin/bash

# Credit goes to https://github.com/Tsingularity/FRN for these scripts 🙌

. gdownload.sh
. conditional.sh

data_path='../..'
echo 'this is the data_path you are trying to download data into:'
echo $data_path

cd $data_path

echo "downloading IGNITE dataset..."
gdownload  13ShilmCtOdE8yKWlZAkGpn7t0k3o_l1B ignite_dataset.tar
conditional_targz ignite_dataset.tar ee95a6355f22e1fcd0bf362d55df2c8e
