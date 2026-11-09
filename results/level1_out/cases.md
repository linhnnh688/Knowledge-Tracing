## 5 correct predictions

### correct #1  user=78537  t=59  skill=7:Venn Diagram  y=1  p=0.953  pred=1  confidence=0.453
- history (8 steps): t51:7:Venn Diagram=OK, t52:7:Venn Diagram=OK, t53:7:Venn Diagram=OK, t54:7:Venn Diagram=OK, t55:7:Venn Diagram=OK, t56:7:Venn Diagram=OK, t57:7:Venn Diagram=OK, t58:7:Venn Diagram=OK
- same-skill history: 8 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 8:0.657, 44:0.136, 9:0.123, 30:0.021, 24:0.018
- counterfactual: not available (re-run p=0.5917 != logged p=0.9532)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.

### correct #2  user=88136  t=88  skill=78:Multiplication and Division Positive Decimals  y=1  p=0.700  pred=1  confidence=0.200
- history (8 steps): t80:78:Multiplication and Division Positive Decimals=NG, t81:78:Multiplication and Division Positive Decimals=NG, t82:78:Multiplication and Division Positive Decimals=NG, t83:78:Multiplication and Division Positive Decimals=OK, t84:78:Multiplication and Division Positive Decimals=NG, t85:78:Multiplication and Division Positive Decimals=NG, t86:78:Multiplication and Division Positive Decimals=NG, t87:78:Multiplication and Division Positive Decimals=OK
- same-skill history: 8 tries, accuracy=0.25
- graph neighbors (Transition train-only; case model=GKT-Transition): 79:0.613, 80:0.376, 81:0.011
- counterfactual: not available (re-run p=0.4501 != logged p=0.7000)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.

### correct #3  user=89443  t=10  skill=6:Table  y=1  p=0.900  pred=1  confidence=0.400
- history (8 steps): t2:6:Table=NG, t3:6:Table=OK, t4:6:Table=OK, t5:6:Table=OK, t6:6:Table=NG, t7:6:Table=OK, t8:6:Table=OK, t9:6:Table=OK
- same-skill history: 8 tries, accuracy=0.75
- graph neighbors (Transition train-only; case model=GKT-Transition): 7:0.446, 8:0.323, 12:0.083, 14:0.067, 9:0.037
- counterfactual: not available (re-run p=0.5243 != logged p=0.9000)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.

### correct #4  user=77949  t=14  skill=30:Conversion of Fraction Decimals Percents  y=0  p=0.400  pred=0  confidence=0.100
- history (8 steps): t6:1:Circle Graph=NG, t7:24:nan=NG, t8:24:nan=NG, t9:24:nan=NG, t10:24:nan=NG, t11:24:nan=OK, t12:24:nan=NG, t13:24:nan=NG
- same-skill history: none (cold skill for this student)
- graph neighbors (Transition train-only; case model=GKT-Transition): 32:0.398, 31:0.158, 33:0.123, 37:0.080, 38:0.078
- counterfactual: not available (re-run p=0.4479 != logged p=0.4000)
- why: Dự đoán đúng: lỗi trước đó ở kỹ năng cùng/liên quan giữ mastery thấp.

### correct #5  user=96283  t=184  skill=13:Probability of Two Distinct Events  y=1  p=0.800  pred=1  confidence=0.300
- history (8 steps): t176:13:Probability of Two Distinct Events=OK, t177:13:Probability of Two Distinct Events=OK, t178:13:Probability of Two Distinct Events=NG, t179:13:Probability of Two Distinct Events=OK, t180:13:Probability of Two Distinct Events=OK, t181:13:Probability of Two Distinct Events=OK, t182:13:Probability of Two Distinct Events=OK, t183:13:Probability of Two Distinct Events=OK
- same-skill history: 8 tries, accuracy=0.88
- graph neighbors (Transition train-only; case model=GKT-Transition): 14:0.913, 15:0.042, 29:0.013, 17:0.008, 55:0.004
- counterfactual: not available (re-run p=0.4887 != logged p=0.7999)
- why: Dự đoán đúng: lịch sử đã có câu đúng cùng kỹ năng nên mô hình nâng p.
## 5 incorrect predictions

### incorrect #1  user=75169  t=139  skill=7:Venn Diagram  y=0  p=0.953  pred=1  confidence=0.453
- history (8 steps): t131:7:Venn Diagram=NG, t132:7:Venn Diagram=OK, t133:7:Venn Diagram=OK, t134:7:Venn Diagram=OK, t135:7:Venn Diagram=OK, t136:7:Venn Diagram=OK, t137:7:Venn Diagram=OK, t138:7:Venn Diagram=OK
- same-skill history: 8 tries, accuracy=0.88
- graph neighbors (Transition train-only; case model=GKT-Transition): 8:0.657, 44:0.136, 9:0.123, 30:0.021, 24:0.018
- counterfactual: not available (re-run p=0.5848 != logged p=0.9531)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #2  user=88478  t=51  skill=30:Conversion of Fraction Decimals Percents  y=0  p=0.700  pred=1  confidence=0.200
- history (8 steps): t43:14:Probability of a Single Event=OK, t44:14:Probability of a Single Event=OK, t45:30:Conversion of Fraction Decimals Percents=OK, t46:30:Conversion of Fraction Decimals Percents=NG, t47:30:Conversion of Fraction Decimals Percents=OK, t48:30:Conversion of Fraction Decimals Percents=NG, t49:30:Conversion of Fraction Decimals Percents=NG, t50:30:Conversion of Fraction Decimals Percents=OK
- same-skill history: 6 tries, accuracy=0.50
- graph neighbors (Transition train-only; case model=GKT-Transition): 32:0.398, 31:0.158, 33:0.123, 37:0.080, 38:0.078
- counterfactual: not available (re-run p=0.4642 != logged p=0.7000)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #3  user=83206  t=116  skill=44:Percent Of  y=0  p=0.900  pred=1  confidence=0.400
- history (8 steps): t108:42:Multiplication Fractions=OK, t109:43:Multiplication Whole Numbers=NG, t110:43:Multiplication Whole Numbers=OK, t111:43:Multiplication Whole Numbers=OK, t112:44:Percent Of=OK, t113:44:Percent Of=OK, t114:44:Percent Of=OK, t115:44:Percent Of=OK
- same-skill history: 4 tries, accuracy=1.00
- graph neighbors (Transition train-only; case model=GKT-Transition): 48:0.242, 45:0.220, 51:0.130, 49:0.112, 47:0.099
- counterfactual: not available (re-run p=0.6242 != logged p=0.9000)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #4  user=84837  t=24  skill=79:Addition and Subtraction Integers  y=0  p=0.600  pred=1  confidence=0.100
- history (8 steps): t16:55:Absolute Value=NG, t17:55:Absolute Value=OK, t18:55:Absolute Value=OK, t19:55:Absolute Value=NG, t20:55:Absolute Value=NG, t21:79:Addition and Subtraction Integers=NG, t22:79:Addition and Subtraction Integers=NG, t23:79:Addition and Subtraction Integers=OK
- same-skill history: 3 tries, accuracy=0.33
- graph neighbors (Transition train-only; case model=GKT-Transition): 80:0.395, 81:0.379, 82:0.179, 98:0.016, 97:0.014
- counterfactual: not available (re-run p=0.5168 != logged p=0.6000)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.

### incorrect #5  user=73962  t=42  skill=9:Median  y=0  p=0.800  pred=1  confidence=0.300
- history (8 steps): t34:9:Median=NG, t35:9:Median=OK, t36:9:Median=OK, t37:9:Median=OK, t38:9:Median=OK, t39:9:Median=NG, t40:9:Median=NG, t41:9:Median=OK
- same-skill history: 8 tries, accuracy=0.62
- graph neighbors (Transition train-only; case model=GKT-Transition): 10:0.642, 11:0.275, 24:0.019, 14:0.017, 12:0.011
- counterfactual: not available (re-run p=0.5557 != logged p=0.8000)
- why: Dương tính giả: mô hình thấy các thành công gần đây và ngoại suy quá mức.