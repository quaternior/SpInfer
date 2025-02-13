source run_all_main_res_nsys.sh
source run_cusparse_nsys.sh
source run_sparta_4090_nsys.sh
source run_sputnik_all_4090_nsys.sh

python handle_logcsv.py
pip install seaborn
python draw_fig10.py