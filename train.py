"""Train + calibrate + save. Usage: python train.py data/AirQualityUCI.csv"""
import sys, warnings; warnings.filterwarnings("ignore")
from pipeline import *
df = load_raw(sys.argv[1] if len(sys.argv) > 1 else "data/AirQualityUCI.csv")
F = make_features(df)
det = AirQualityDetector().fit(F[:"2004-07-01"], F["2004-07-01":"2004-07-29"])   # train Mar-Jun, calibrate July
det.save("models/detector.joblib")
df.loc[:"2005-04-04"].to_csv("data/history.csv")        # replay/history buffer used by the service
print("saved; train end", det.train_end)
