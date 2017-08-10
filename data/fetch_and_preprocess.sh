#!/bin/bash
set -e
python download.py
python preprocess_data.py
python preprocess_data_treelstm.py
