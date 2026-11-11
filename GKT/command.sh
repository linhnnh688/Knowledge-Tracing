# Chạy từ THƯ MỤC GỐC repo: GKT/ là package (import tương đối), không chạy được bằng `python train.py`.
python -m GKT.train --data-dir=GKT/data --data-file=daily_2019_kp_all.csv --graph-type=VAE;
python -m GKT.train --data-dir=GKT/data --data-file=daily_2019_kp_all.csv --graph-type=MHA;
python -m GKT.train --data-dir=GKT/data --data-file=daily_2019_kp_all.csv --graph-type=Dense;
python -m GKT.train --data-dir=GKT/data --data-file=daily_2019_kp_all.csv --graph-type=Transition;
python -m GKT.train --data-dir=GKT/data --data-file=daily_2019_kp_all.csv --graph-type=PAM;