from __future__ import annotations

import argparse

import sghf


parser = argparse.ArgumentParser(description="Fit a pyhf workspace with SGHF")
parser.add_argument("workspace")
args = parser.parse_args()

model = sghf.load(args.workspace)
fit = model.fit()
limit = model.upper_limit()

print(f"POI: {model.poi_name}")
print(f"fit: {fit.value(model.poi_name):.6g} +/- {fit.error(model.poi_name):.6g}")
print(f"observed 95% CL upper limit: {limit.observed:.6g}")
