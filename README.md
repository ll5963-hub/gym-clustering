# Gym member segments: who should get the monthly offer?

k-means clustering (scikit-learn) on a 70/30 split of `gym_members.csv`, with the scaler fit on the training members only. The script compares training and test elbow curves, profiles 4 segments, and computes the gym's monthly net from the $10 credit under the baseline (credit to everyone) vs v1 (credit to Drifting newcomers only).

```
pip install pandas scikit-learn matplotlib
python fit.py
```

`fit.py` prints every number on the slides (saved in `output.txt`) and writes `elbow.png`.
