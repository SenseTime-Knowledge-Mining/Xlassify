#!/bin/sh 
# =============================================================
# 更换环境务必注意random_state是不一样的
# =============================================================

# logname="ge5_le50_dedup_f5_6m_r0"
# nohup python -u trainer6m.py --log $logname > log/$logname 2>&1 &

logname="ge5_le50_dedup_f5_7m_r0"
nohup python -u trainer7m.py --log $logname > log/$logname 2>&1 &

# logname="ge5_le50_dedup_7m_RF"
# nohup python -u trainer7m_RF.py > log/$logname 2>&1 &

# logname="ge5_le50_dedup_6m_RF"
# nohup python -u trainer6m_RF.py > log/$logname 2>&1 &

