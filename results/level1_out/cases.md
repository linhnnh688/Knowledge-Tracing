## 5 correct predictions

### correct #1  user=75316  t=199  skill=90:Area Triangle  y=1  p=0.983  pred=1  confidence=0.483
- history (8 steps): t191:90:Area Triangle=OK, t192:90:Area Triangle=OK, t193:90:Area Triangle=OK, t194:90:Area Triangle=OK, t195:90:Area Triangle=OK, t196:90:Area Triangle=OK, t197:90:Area Triangle=OK, t198:90:Area Triangle=OK
- same-skill history: 8 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 93:0.500, 91:0.250, 94:0.250
- counterfactual: not available (re-run p=0.8962 != logged p=0.9826)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.

### correct #2  user=96270  t=173  skill=20:Pythagorean Theorem  y=1  p=0.700  pred=1  confidence=0.200
- history (8 steps): t165:14:Probability of a Single Event=OK, t166:14:Probability of a Single Event=OK, t167:15:Interior Angles Figures with More than 3 Sides=OK, t168:15:Interior Angles Figures with More than 3 Sides=OK, t169:16:Interior Angles Triangle=OK, t170:17:Congruence=OK, t171:18:Complementary and Supplementary Angles=OK, t172:18:Complementary and Supplementary Angles=OK
- same-skill history: none (cold skill for this student)
- graph neighbors (Transition train-only; case model=GKT-MHA): 21:0.526, 29:0.137, 30:0.074, 37:0.063, 27:0.053
- counterfactual (flip one past attempt): t168 15:Interior Angles Figures with More than 3 Sides (OK -> NG): p 0.700 -> 0.698 (dp=-0.001)
  - next largest: t167 15:Interior Angles Figures with More than 3 Sides dp=-0.001; t170 17:Congruence dp=-0.001
- occlusion (remove one past attempt): t172 18:Complementary and Supplementary Angles removed -> p 0.700 -> 0.847 (dp=+0.147)
  - next largest: t170 17:Congruence dp=-0.001; t167 15:Interior Angles Figures with More than 3 Sides dp=-0.001
- strongest edges at the target concept: 21 w=0.526 out (never touched by this learner), 18 w=0.324 in (learner last touched t172, OK), 19 w=0.300 in (never touched by this learner)
- knowledge state h_t at target concept: norm=1.178, rank 83/123 (global mean=1.217)
  - top-5 concepts by state norm: 18:Complementary and Supplementary Angles (2.04), 73:Linear Equations (1.42), 77:Slope (1.42), 36:Rounding (1.42), 37:Addition Whole Numbers (1.41)
  - graph-neighbor state norms: 21:Nets of 3D Figures (1.41), 29:Calculations with Similar Figures (1.11), 30:Conversion of Fraction Decimals Percents (1.18), 37:Addition Whole Numbers (1.41), 27:Perimeter of a Polygon (1.24)
- why: Dự đoán đúng: bước ảnh hưởng mạnh nhất là t168 (15:Interior Angles Figures with More than 3 Sides): lật kết quả làm p giảm 0.001; bỏ hẳn lần t172 (18:Complementary and Supplementary Angles) làm p thay đổi +0.147.

### correct #3  user=80641  t=33  skill=81:Multiplication and Division Integers  y=1  p=0.900  pred=1  confidence=0.400
- history (8 steps): t25:81:Multiplication and Division Integers=OK, t26:81:Multiplication and Division Integers=OK, t27:81:Multiplication and Division Integers=OK, t28:81:Multiplication and Division Integers=OK, t29:81:Multiplication and Division Integers=OK, t30:81:Multiplication and Division Integers=OK, t31:81:Multiplication and Division Integers=OK, t32:81:Multiplication and Division Integers=OK
- same-skill history: 8 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 82:0.784, 97:0.092, 96:0.063, 98:0.054, 92:0.003
- counterfactual: not available (re-run p=0.7189 != logged p=0.9000)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.

### correct #4  user=83954  t=57  skill=45:Subtraction Whole Numbers  y=1  p=0.600  pred=1  confidence=0.100
- history (8 steps): t49:32:Ordering Positive Decimals=OK, t50:32:Ordering Positive Decimals=OK, t51:37:Addition Whole Numbers=OK, t52:37:Addition Whole Numbers=OK, t53:37:Addition Whole Numbers=OK, t54:37:Addition Whole Numbers=OK, t55:37:Addition Whole Numbers=OK, t56:45:Subtraction Whole Numbers=NG
- same-skill history: 1 tries, accuracy=0.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 55:0.500, 46:0.268, 47:0.063, 48:0.054, 58:0.041
- counterfactual: not available (re-run p=0.6748 != logged p=0.6000)
- why: Dự đoán đúng: các bước gần đây thành công nên mô hình ngoại suy sang kỹ năng này.

### correct #5  user=88744  t=72  skill=55:Absolute Value  y=1  p=0.800  pred=1  confidence=0.300
- history (8 steps): t64:45:Subtraction Whole Numbers=NG, t65:45:Subtraction Whole Numbers=NG, t66:45:Subtraction Whole Numbers=OK, t67:45:Subtraction Whole Numbers=OK, t68:45:Subtraction Whole Numbers=NG, t69:45:Subtraction Whole Numbers=NG, t70:45:Subtraction Whole Numbers=NG, t71:55:Absolute Value=OK
- same-skill history: 1 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 79:0.335, 56:0.318, 58:0.106, 59:0.047, 78:0.042
- counterfactual: not available (re-run p=0.7061 != logged p=0.8000)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.
## 5 incorrect predictions

### incorrect #1  user=78591  t=104  skill=44:Percent Of  y=0  p=0.981  pred=1  confidence=0.481
- history (8 steps): t96:44:Percent Of=OK, t97:44:Percent Of=OK, t98:44:Percent Of=OK, t99:44:Percent Of=OK, t100:44:Percent Of=OK, t101:44:Percent Of=OK, t102:44:Percent Of=OK, t103:44:Percent Of=OK
- same-skill history: 8 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 48:0.242, 45:0.220, 51:0.130, 49:0.112, 47:0.099
- counterfactual: not available (re-run p=0.8854 != logged p=0.9811)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #2  user=78171  t=91  skill=98:Equation Solving Two or Fewer Steps  y=0  p=0.700  pred=1  confidence=0.200
- history (8 steps): t83:97:Order of Operations All=OK, t84:97:Order of Operations All=OK, t85:97:Order of Operations All=OK, t86:97:Order of Operations All=OK, t87:97:Order of Operations All=OK, t88:97:Order of Operations All=OK, t89:97:Order of Operations All=OK, t90:97:Order of Operations All=OK
- same-skill history: none (cold skill for this student)
- graph neighbors (Transition train-only; case model=GKT-MHA): 99:0.661, 106:0.182, 104:0.033, 119:0.025, 101:0.025
- counterfactual (flip one past attempt): t90 97:Order of Operations All (OK -> NG): p 0.700 -> 0.700 (dp=+0.000)
  - next largest: t86 97:Order of Operations All dp=-0.000; t85 97:Order of Operations All dp=-0.000
- occlusion (remove one past attempt): t90 97:Order of Operations All removed -> p 0.700 -> 0.844 (dp=+0.144)
  - next largest: t83 97:Order of Operations All dp=-0.001; t84 97:Order of Operations All dp=-0.001
- strongest edges at the target concept: 97 w=0.891 in (learner last touched t90, OK), 99 w=0.661 out (never touched by this learner), 85 w=0.571 in (never touched by this learner)
- knowledge state h_t at target concept: norm=1.180, rank 75/123 (global mean=1.217)
  - top-5 concepts by state norm: 97:Order of Operations All (2.52), 77:Slope (1.44), 73:Linear Equations (1.43), 21:Nets of 3D Figures (1.41), 36:Rounding (1.41)
  - graph-neighbor state norms: 99:Equation Solving More Than Two Steps (1.23), 106:Write Linear Equation from Graph (1.20), 104:Write Linear Equation from Situation (1.24), 119:Solving for a variable (1.18), 101:Greatest Common Factor (1.14)
- why: Dự đoán sai: bỏ hẳn lần t90 (97:Order of Operations All) làm p thay đổi +0.144.

### incorrect #3  user=78154  t=72  skill=56:Exponents  y=0  p=0.900  pred=1  confidence=0.400
- history (8 steps): t64:55:Absolute Value=OK, t65:55:Absolute Value=OK, t66:55:Absolute Value=OK, t67:56:Exponents=OK, t68:56:Exponents=OK, t69:56:Exponents=OK, t70:56:Exponents=OK, t71:56:Exponents=OK
- same-skill history: 5 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-MHA): 79:0.333, 59:0.124, 58:0.098, 78:0.076, 63:0.058
- counterfactual: not available (re-run p=0.6152 != logged p=0.8999)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #4  user=83322  t=73  skill=79:Addition and Subtraction Integers  y=0  p=0.600  pred=1  confidence=0.100
- history (8 steps): t65:79:Addition and Subtraction Integers=NG, t66:79:Addition and Subtraction Integers=OK, t67:79:Addition and Subtraction Integers=OK, t68:79:Addition and Subtraction Integers=OK, t69:79:Addition and Subtraction Integers=OK, t70:79:Addition and Subtraction Integers=OK, t71:79:Addition and Subtraction Integers=NG, t72:79:Addition and Subtraction Integers=NG
- same-skill history: 8 tries, accuracy=0.62
- graph neighbors (Transition train-only; case model=GKT-MHA): 80:0.395, 81:0.379, 82:0.179, 98:0.016, 97:0.014
- counterfactual: not available (re-run p=0.4593 != logged p=0.6000)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #5  user=88136  t=126  skill=80:Addition and Subtraction Positive Decimals  y=0  p=0.800  pred=1  confidence=0.300
- history (8 steps): t118:80:Addition and Subtraction Positive Decimals=NG, t119:80:Addition and Subtraction Positive Decimals=OK, t120:80:Addition and Subtraction Positive Decimals=NG, t121:80:Addition and Subtraction Positive Decimals=NG, t122:80:Addition and Subtraction Positive Decimals=NG, t123:80:Addition and Subtraction Positive Decimals=OK, t124:80:Addition and Subtraction Positive Decimals=NG, t125:80:Addition and Subtraction Positive Decimals=OK
- same-skill history: 8 tries, accuracy=0.38
- graph neighbors (Transition train-only; case model=GKT-MHA): 81:0.606, 82:0.386, 96:0.008
- counterfactual: not available (re-run p=0.4276 != logged p=0.7998)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.