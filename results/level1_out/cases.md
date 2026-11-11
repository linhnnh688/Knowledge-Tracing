## 5 correct predictions

### correct #1  user=82163  t=63  skill=34:Addition Whole Numbers  y=1  p=0.967  pred=1  confidence=0.467
- history (8 steps): t55:40:Subtraction Whole Numbers=OK, t56:34:Addition Whole Numbers=OK, t57:40:Subtraction Whole Numbers=OK, t58:34:Addition Whole Numbers=OK, t59:40:Subtraction Whole Numbers=OK, t60:40:Subtraction Whole Numbers=OK, t61:34:Addition Whole Numbers=OK, t62:40:Subtraction Whole Numbers=OK
- same-skill history: 3 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 40:0.413, 70:0.220, 50:0.107, 30:0.063, 71:0.032
- counterfactual (flip one past attempt): t62 40:Subtraction Whole Numbers (OK -> NG): p 0.967 -> 0.902 (dp=-0.065)
  - next largest: t61 34:Addition Whole Numbers dp=-0.013; t60 40:Subtraction Whole Numbers dp=-0.010
- graph ablation for THIS prediction: A := 0 moves p 0.967 -> 0.845 (dp=-0.122)
- occlusion (remove one past attempt): t62 40:Subtraction Whole Numbers removed -> p 0.967 -> 3.180 (dp=+2.213)
  - next largest: t58 34:Addition Whole Numbers dp=-0.001; t57 40:Subtraction Whole Numbers dp=-0.001
- strongest edges at the target concept: 40 w=0.428 in (learner last touched t62, OK), 50 w=0.308 in (never touched by this learner), 70 w=0.220 out (never touched by this learner)
- knowledge state h_t at target concept: norm=4.502, rank 1/112 (global mean=1.454)
  - top-5 concepts by state norm: 34:Addition Whole Numbers (4.50), 50:Absolute Value (4.29), 40:Subtraction Whole Numbers (3.63), 70:Addition and Subtraction Integers (3.11), 53:Pattern Finding  (2.91)
  - graph-neighbor state norms: 40:Subtraction Whole Numbers (3.63), 70:Addition and Subtraction Integers (3.11), 50:Absolute Value (4.29), 30:Ordering Positive Decimals (1.76), 71:Addition and Subtraction Positive Decimals (1.86)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t62 (40:Subtraction Whole Numbers): lật kết quả làm p giảm 0.065; đồ thị làm p tăng 0.122 (p=0.967 vs 0.845 khi A := 0); bỏ hẳn lần t62 (40:Subtraction Whole Numbers) làm p thay đổi +2.213; lịch sử cùng kỹ năng: 1.00 đúng trên 3 lần.

### correct #2  user=96270  t=57  skill=6:Table  y=1  p=0.700  pred=1  confidence=0.200
- history (8 steps): t49:30:Ordering Positive Decimals=OK, t50:6:Table=NG, t51:6:Table=OK, t52:6:Table=NG, t53:6:Table=NG, t54:6:Table=NG, t55:6:Table=NG, t56:6:Table=OK
- same-skill history: 7 tries, accuracy=0.29
- graph neighbors (Transition train-only; case model=GKT-Transition): 13:0.138, 27:0.079, 7:0.076, 5:0.062, 11:0.056
- counterfactual (flip one past attempt): t56 6:Table (OK -> NG): p 0.700 -> 0.355 (dp=-0.345)
  - next largest: t55 6:Table dp=+0.109; t54 6:Table dp=+0.101
- graph ablation for THIS prediction: A := 0 moves p 0.700 -> 0.640 (dp=-0.060)
- occlusion (remove one past attempt): t56 6:Table removed -> p 0.700 -> -0.306 (dp=-1.006)
  - next largest: t51 6:Table dp=-0.043; t52 6:Table dp=+0.032
- strongest edges at the target concept: 13 w=0.264 in (never touched by this learner), 45 w=0.178 in (never touched by this learner), 61 w=0.173 in (never touched by this learner)
- knowledge state h_t at target concept: norm=2.602, rank 2/112 (global mean=1.239)
  - top-5 concepts by state norm: 13:Probability of a Single Event (2.78), 6:Table (2.60), 79:Area Rectangle (1.89), 81:Area Triangle (1.77), 72:Multiplication and Division Integers (1.72)
  - graph-neighbor state norms: 13:Probability of a Single Event (2.78), 27:Calculations with Similar Figures (1.29), 7:Venn Diagram (1.25), 5:Stem and Leaf Plot (0.91), 11:Counting Methods (0.92)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t56 (6:Table): lật kết quả làm p giảm 0.345; đồ thị làm p tăng 0.060 (p=0.700 vs 0.640 khi A := 0); bỏ hẳn lần t56 (6:Table) làm p thay đổi -1.006; lịch sử cùng kỹ năng: 0.29 đúng trên 7 lần.

### correct #3  user=88479  t=182  skill=72:Multiplication and Division Integers  y=0  p=0.100  pred=0  confidence=0.400
- history (8 steps): t174:72:Multiplication and Division Integers=NG, t175:72:Multiplication and Division Integers=NG, t176:72:Multiplication and Division Integers=NG, t177:72:Multiplication and Division Integers=NG, t178:72:Multiplication and Division Integers=NG, t179:72:Multiplication and Division Integers=NG, t180:72:Multiplication and Division Integers=NG, t181:72:Multiplication and Division Integers=NG
- same-skill history: 8 tries, accuracy=0.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 70:0.156, 73:0.099, 51:0.078, 71:0.077, 30:0.058
- counterfactual (flip one past attempt): t181 72:Multiplication and Division Integers (NG -> OK): p 0.100 -> 0.278 (dp=+0.178)
  - next largest: t180 72:Multiplication and Division Integers dp=+0.093; t179 72:Multiplication and Division Integers dp=+0.055
- graph ablation for THIS prediction: A := 0 moves p 0.100 -> 0.180 (dp=+0.079)
- occlusion (remove one past attempt): t181 72:Multiplication and Division Integers removed -> p 0.100 -> -2.195 (dp=-2.295)
  - next largest: t174 72:Multiplication and Division Integers dp=+0.000; t175 72:Multiplication and Division Integers dp=+0.000
- strongest edges at the target concept: 70 w=0.366 in (never touched by this learner), 53 w=0.264 in (never touched by this learner), 19 w=0.198 in (never touched by this learner)
- knowledge state h_t at target concept: norm=3.478, rank 3/112 (global mean=1.252)
  - top-5 concepts by state norm: 70:Addition and Subtraction Integers (4.32), 53:Pattern Finding  (3.93), 72:Multiplication and Division Integers (3.48), 71:Addition and Subtraction Positive Decimals (3.38), 19:Pythagorean Theorem (2.77)
  - graph-neighbor state norms: 70:Addition and Subtraction Integers (4.32), 73:Addition and Subtraction Fractions (2.38), 51:Exponents (1.17), 71:Addition and Subtraction Positive Decimals (3.38), 30:Ordering Positive Decimals (0.71)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t181 (72:Multiplication and Division Integers): lật kết quả làm p tăng 0.178; đồ thị làm p giảm 0.079 (p=0.100 vs 0.180 khi A := 0); bỏ hẳn lần t181 (72:Multiplication and Division Integers) làm p thay đổi -2.295; lịch sử cùng kỹ năng: 0.00 đúng trên 8 lần.

### correct #4  user=90831  t=70  skill=28:Conversion of Fraction Decimals Percents  y=0  p=0.400  pred=0  confidence=0.100
- history (8 steps): t62:72:Multiplication and Division Integers=NG, t63:72:Multiplication and Division Integers=NG, t64:72:Multiplication and Division Integers=NG, t65:72:Multiplication and Division Integers=NG, t66:72:Multiplication and Division Integers=NG, t67:53:Pattern Finding =NG, t68:72:Multiplication and Division Integers=NG, t69:72:Multiplication and Division Integers=NG
- same-skill history: none (cold skill for this student)
- graph neighbors (Transition train-only; case model=GKT-Transition): 43:0.093, 39:0.074, 89:0.049, 30:0.048, 37:0.047
- counterfactual (flip one past attempt): t66 72:Multiplication and Division Integers (NG -> OK): p 0.400 -> 0.415 (dp=+0.015)
  - next largest: t65 72:Multiplication and Division Integers dp=+0.014; t68 72:Multiplication and Division Integers dp=+0.013
- graph ablation for THIS prediction: A := 0 moves p 0.400 -> 0.674 (dp=+0.274)
- occlusion (remove one past attempt): t69 72:Multiplication and Division Integers removed -> p 0.400 -> -0.380 (dp=-0.780)
  - next largest: t67 53:Pattern Finding  dp=-0.010; t68 72:Multiplication and Division Integers dp=+0.006
- strongest edges at the target concept: 86 w=0.564 in (never touched by this learner), 67 w=0.424 in (never touched by this learner), 66 w=0.379 in (never touched by this learner)
- knowledge state h_t at target concept: norm=0.757, rank 101/112 (global mean=1.232)
  - top-5 concepts by state norm: 70:Addition and Subtraction Integers (3.87), 53:Pattern Finding  (3.70), 72:Multiplication and Division Integers (3.17), 71:Addition and Subtraction Positive Decimals (3.05), 19:Pythagorean Theorem (2.48)
  - graph-neighbor state norms: 43:Finding Percents (0.81), 39:Percent Of (0.70), 89:Equation Solving Two or Fewer Steps (0.70), 30:Ordering Positive Decimals (0.74), 37:Least Common Multiple (0.90)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t66 (72:Multiplication and Division Integers): lật kết quả làm p tăng 0.015; đồ thị làm p giảm 0.274 (p=0.400 vs 0.674 khi A := 0); bỏ hẳn lần t69 (72:Multiplication and Division Integers) làm p thay đổi -0.780.

### correct #5  user=78449  t=80  skill=54:nan  y=1  p=0.800  pred=1  confidence=0.300
- history (8 steps): t72:54:nan=OK, t73:54:nan=OK, t74:54:nan=OK, t75:54:nan=NG, t76:54:nan=OK, t77:54:nan=OK, t78:54:nan=OK, t79:54:nan=NG
- same-skill history: 8 tries, accuracy=0.75
- graph neighbors (Transition train-only; case model=GKT-Transition): 89:0.322, 109:0.231, 41:0.070, 40:0.049, 53:0.035
- counterfactual (flip one past attempt): t77 54:nan (OK -> NG): p 0.800 -> 0.744 (dp=-0.056)
  - next largest: t75 54:nan dp=+0.037; t78 54:nan dp=-0.028
- graph ablation for THIS prediction: A := 0 moves p 0.800 -> 0.800 (dp=-0.000)
- occlusion (remove one past attempt): t79 54:nan removed -> p 0.800 -> 1.541 (dp=+0.741)
  - next largest: t75 54:nan dp=+0.033; t76 54:nan dp=-0.008
- strongest edges at the target concept: 89 w=0.322 out (never touched by this learner), 109 w=0.231 out (never touched by this learner), 63 w=0.083 in (never touched by this learner)
- knowledge state h_t at target concept: norm=2.927, rank 1/112 (global mean=1.329)
  - top-5 concepts by state norm: 54:nan (2.93), 41:Square Root (2.33), 109:Solving for a variable (1.93), 89:Equation Solving Two or Fewer Steps (1.79), 53:Pattern Finding  (1.74)
  - graph-neighbor state norms: 89:Equation Solving Two or Fewer Steps (1.79), 109:Solving for a variable (1.93), 41:Square Root (2.33), 40:Subtraction Whole Numbers (1.52), 53:Pattern Finding  (1.74)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t77 (54:nan): lật kết quả làm p giảm 0.056; bỏ toàn bộ cạnh đồ thị không đổi p (|Δp| < 0.001); bỏ hẳn lần t79 (54:nan) làm p thay đổi +0.741; lịch sử cùng kỹ năng: 0.75 đúng trên 8 lần.
## 5 incorrect predictions

### incorrect #1  user=78863  t=107  skill=34:Addition Whole Numbers  y=0  p=0.967  pred=1  confidence=0.467
- history (8 steps): t99:40:Subtraction Whole Numbers=OK, t100:40:Subtraction Whole Numbers=OK, t101:40:Subtraction Whole Numbers=OK, t102:34:Addition Whole Numbers=OK, t103:40:Subtraction Whole Numbers=OK, t104:34:Addition Whole Numbers=OK, t105:34:Addition Whole Numbers=OK, t106:40:Subtraction Whole Numbers=OK
- same-skill history: 3 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 40:0.413, 70:0.220, 50:0.107, 30:0.063, 71:0.032
- counterfactual (flip one past attempt): t106 40:Subtraction Whole Numbers (OK -> NG): p 0.967 -> 0.887 (dp=-0.080)
  - next largest: t105 34:Addition Whole Numbers dp=-0.014; t104 34:Addition Whole Numbers dp=-0.007
- graph ablation for THIS prediction: A := 0 moves p 0.967 -> 0.875 (dp=-0.092)
- occlusion (remove one past attempt): t106 40:Subtraction Whole Numbers removed -> p 0.967 -> 3.072 (dp=+2.105)
  - next largest: t103 40:Subtraction Whole Numbers dp=-0.001; t104 34:Addition Whole Numbers dp=+0.001
- strongest edges at the target concept: 40 w=0.428 in (learner last touched t106, OK), 50 w=0.308 in (never touched by this learner), 70 w=0.220 out (never touched by this learner)
- knowledge state h_t at target concept: norm=4.441, rank 1/112 (global mean=1.471)
  - top-5 concepts by state norm: 34:Addition Whole Numbers (4.44), 50:Absolute Value (4.30), 40:Subtraction Whole Numbers (3.67), 70:Addition and Subtraction Integers (3.28), 53:Pattern Finding  (2.74)
  - graph-neighbor state norms: 40:Subtraction Whole Numbers (3.67), 70:Addition and Subtraction Integers (3.28), 50:Absolute Value (4.30), 30:Ordering Positive Decimals (1.82), 71:Addition and Subtraction Positive Decimals (2.06)
- why: Dự đoán sai: bước ảnh hưởng mạnh nhất là t106 (40:Subtraction Whole Numbers): lật kết quả làm p giảm 0.080; đồ thị làm p tăng 0.092 (p=0.967 vs 0.875 khi A := 0); bỏ hẳn lần t106 (40:Subtraction Whole Numbers) làm p thay đổi +2.105; lịch sử cùng kỹ năng: 1.00 đúng trên 3 lần.

### incorrect #2  user=78958  t=113  skill=43:Finding Percents  y=1  p=0.300  pred=0  confidence=0.200
- history (8 steps): t105:46:Unit Rate=NG, t106:46:Unit Rate=NG, t107:46:Unit Rate=NG, t108:46:Unit Rate=NG, t109:46:Unit Rate=NG, t110:43:Finding Percents=NG, t111:43:Finding Percents=NG, t112:43:Finding Percents=NG
- same-skill history: 3 tries, accuracy=0.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 39:0.266, 1:0.178, 28:0.086, 42:0.077, 44:0.054
- counterfactual (flip one past attempt): t112 43:Finding Percents (NG -> OK): p 0.300 -> 0.612 (dp=+0.312)
  - next largest: t111 43:Finding Percents dp=+0.246; t110 43:Finding Percents dp=+0.169
- graph ablation for THIS prediction: A := 0 moves p 0.300 -> 0.510 (dp=+0.210)
- occlusion (remove one past attempt): t112 43:Finding Percents removed -> p 0.300 -> -0.917 (dp=-1.217)
  - next largest: t110 43:Finding Percents dp=-0.014; t111 43:Finding Percents dp=-0.014
- strongest edges at the target concept: 42 w=0.278 in (never touched by this learner), 39 w=0.266 out (never touched by this learner), 1 w=0.225 in (never touched by this learner)
- knowledge state h_t at target concept: norm=3.135, rank 1/112 (global mean=1.239)
  - top-5 concepts by state norm: 43:Finding Percents (3.13), 39:Percent Of (2.86), 42:nan (2.79), 1:Circle Graph (2.32), 46:Unit Rate (2.25)
  - graph-neighbor state norms: 39:Percent Of (2.86), 1:Circle Graph (2.32), 28:Conversion of Fraction Decimals Percents (0.79), 42:nan (2.79), 44:Proportion (1.73)
- why: Dự đoán sai: bước ảnh hưởng mạnh nhất là t112 (43:Finding Percents): lật kết quả làm p tăng 0.312; đồ thị làm p giảm 0.210 (p=0.300 vs 0.510 khi A := 0); bỏ hẳn lần t112 (43:Finding Percents) làm p thay đổi -1.217; lịch sử cùng kỹ năng: 0.00 đúng trên 3 lần.

### incorrect #3  user=80662  t=27  skill=72:Multiplication and Division Integers  y=0  p=0.900  pred=1  confidence=0.400
- history (8 steps): t19:72:Multiplication and Division Integers=OK, t20:72:Multiplication and Division Integers=OK, t21:72:Multiplication and Division Integers=NG, t22:72:Multiplication and Division Integers=OK, t23:72:Multiplication and Division Integers=OK, t24:72:Multiplication and Division Integers=OK, t25:72:Multiplication and Division Integers=NG, t26:72:Multiplication and Division Integers=OK
- same-skill history: 8 tries, accuracy=0.75
- graph neighbors (Transition train-only; case model=GKT-Transition): 70:0.156, 73:0.099, 51:0.078, 71:0.077, 30:0.058
- counterfactual (flip one past attempt): t26 72:Multiplication and Division Integers (OK -> NG): p 0.900 -> 0.807 (dp=-0.093)
  - next largest: t24 72:Multiplication and Division Integers dp=-0.031; t25 72:Multiplication and Division Integers dp=+0.021
- graph ablation for THIS prediction: A := 0 moves p 0.900 -> 0.896 (dp=-0.004)
- occlusion (remove one past attempt): t26 72:Multiplication and Division Integers removed -> p 0.900 -> 1.892 (dp=+0.992)
  - next largest: t25 72:Multiplication and Division Integers dp=+0.021; t21 72:Multiplication and Division Integers dp=+0.005
- strongest edges at the target concept: 70 w=0.366 in (never touched by this learner), 53 w=0.264 in (never touched by this learner), 19 w=0.198 in (never touched by this learner)
- knowledge state h_t at target concept: norm=3.507, rank 1/112 (global mean=1.386)
  - top-5 concepts by state norm: 72:Multiplication and Division Integers (3.51), 16:Congruence (2.70), 53:Pattern Finding  (2.67), 81:Area Triangle (2.36), 27:Calculations with Similar Figures (2.32)
  - graph-neighbor state norms: 70:Addition and Subtraction Integers (1.96), 73:Addition and Subtraction Fractions (1.52), 51:Exponents (2.23), 71:Addition and Subtraction Positive Decimals (2.11), 30:Ordering Positive Decimals (1.65)
- why: Dự đoán sai: bước ảnh hưởng mạnh nhất là t26 (72:Multiplication and Division Integers): lật kết quả làm p giảm 0.093; đồ thị làm p tăng 0.004 (p=0.900 vs 0.896 khi A := 0); bỏ hẳn lần t26 (72:Multiplication and Division Integers) làm p thay đổi +0.992; lịch sử cùng kỹ năng: 0.75 đúng trên 8 lần.

### incorrect #4  user=77970  t=65  skill=28:Conversion of Fraction Decimals Percents  y=0  p=0.600  pred=1  confidence=0.100
- history (8 steps): t57:28:Conversion of Fraction Decimals Percents=NG, t58:28:Conversion of Fraction Decimals Percents=NG, t59:28:Conversion of Fraction Decimals Percents=NG, t60:28:Conversion of Fraction Decimals Percents=OK, t61:28:Conversion of Fraction Decimals Percents=OK, t62:28:Conversion of Fraction Decimals Percents=OK, t63:28:Conversion of Fraction Decimals Percents=OK, t64:28:Conversion of Fraction Decimals Percents=NG
- same-skill history: 8 tries, accuracy=0.50
- graph neighbors (Transition train-only; case model=GKT-Transition): 43:0.093, 39:0.074, 89:0.049, 30:0.048, 37:0.047
- counterfactual (flip one past attempt): t63 28:Conversion of Fraction Decimals Percents (OK -> NG): p 0.600 -> 0.452 (dp=-0.148)
  - next largest: t64 28:Conversion of Fraction Decimals Percents dp=+0.129; t62 28:Conversion of Fraction Decimals Percents dp=-0.113
- graph ablation for THIS prediction: A := 0 moves p 0.600 -> 0.600 (dp=+0.000)
- occlusion (remove one past attempt): t64 28:Conversion of Fraction Decimals Percents removed -> p 0.600 -> 0.887 (dp=+0.287)
  - next largest: t60 28:Conversion of Fraction Decimals Percents dp=-0.040; t61 28:Conversion of Fraction Decimals Percents dp=-0.040
- strongest edges at the target concept: 86 w=0.564 in (never touched by this learner), 67 w=0.424 in (never touched by this learner), 66 w=0.379 in (never touched by this learner)
- knowledge state h_t at target concept: norm=2.779, rank 3/112 (global mean=1.256)
  - top-5 concepts by state norm: 66:Percent Discount (3.12), 67:Percents (3.03), 28:Conversion of Fraction Decimals Percents (2.78), 86:Volume Sphere (2.63), 42:nan (2.49)
  - graph-neighbor state norms: 43:Finding Percents (2.29), 39:Percent Of (1.71), 89:Equation Solving Two or Fewer Steps (0.70), 30:Ordering Positive Decimals (1.26), 37:Least Common Multiple (1.40)
- why: Dự đoán sai: bước ảnh hưởng mạnh nhất là t63 (28:Conversion of Fraction Decimals Percents): lật kết quả làm p giảm 0.148; bỏ toàn bộ cạnh đồ thị không đổi p (|Δp| < 0.001); bỏ hẳn lần t64 (28:Conversion of Fraction Decimals Percents) làm p thay đổi +0.287; lịch sử cùng kỹ năng: 0.50 đúng trên 8 lần.

### incorrect #5  user=79054  t=126  skill=93:Write Linear Equation from Ordered Pairs  y=0  p=0.800  pred=1  confidence=0.300
- history (8 steps): t118:40:Subtraction Whole Numbers=OK, t119:40:Subtraction Whole Numbers=OK, t120:53:Pattern Finding =OK, t121:89:Equation Solving Two or Fewer Steps=OK, t122:96:Write Linear Equation from Graph=OK, t123:94:Write Linear Equation from Situation=OK, t124:94:Write Linear Equation from Situation=OK, t125:94:Write Linear Equation from Situation=OK
- same-skill history: none (cold skill for this student)
- graph neighbors (Transition train-only; case model=GKT-Transition): 19:0.329, 94:0.329, 89:0.037, 109:0.037, 46:0.024
- counterfactual (flip one past attempt): t123 94:Write Linear Equation from Situation (OK -> NG): p 0.800 -> 0.739 (dp=-0.061)
  - next largest: t124 94:Write Linear Equation from Situation dp=-0.022; t122 96:Write Linear Equation from Graph dp=-0.016
- graph ablation for THIS prediction: A := 0 moves p 0.800 -> 0.667 (dp=-0.133)
- occlusion (remove one past attempt): t125 94:Write Linear Equation from Situation removed -> p 0.800 -> 1.101 (dp=+0.301)
  - next largest: t123 94:Write Linear Equation from Situation dp=-0.049; t124 94:Write Linear Equation from Situation dp=-0.049
- strongest edges at the target concept: 65 w=0.333 in (never touched by this learner), 94 w=0.329 out (learner last touched t125, OK), 19 w=0.329 out (never touched by this learner)
- knowledge state h_t at target concept: norm=2.190, rank 3/112 (global mean=1.317)
  - top-5 concepts by state norm: 94:Write Linear Equation from Situation (3.51), 50:Absolute Value (2.23), 93:Write Linear Equation from Ordered Pairs (2.19), 84:Volume Cylinder (2.10), 34:Addition Whole Numbers (1.96)
  - graph-neighbor state norms: 19:Pythagorean Theorem (1.25), 94:Write Linear Equation from Situation (3.51), 89:Equation Solving Two or Fewer Steps (1.44), 109:Solving for a variable (1.69), 46:Unit Rate (1.05)
- why: Dự đoán sai: bước ảnh hưởng mạnh nhất là t123 (94:Write Linear Equation from Situation): lật kết quả làm p giảm 0.061; đồ thị làm p tăng 0.133 (p=0.800 vs 0.667 khi A := 0); bỏ hẳn lần t125 (94:Write Linear Equation from Situation) làm p thay đổi +0.301.