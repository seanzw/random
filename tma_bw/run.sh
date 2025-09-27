make
./tma_bw.out > tma_bw_result.log
python process.py -i tma_bw_result.log -o tma_bw_result.png --peak-bw 5.6