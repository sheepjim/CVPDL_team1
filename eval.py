import json
import sys

file_pathA = sys.argv[1]
file_pathB = sys.argv[2]
file_pathC = sys.argv[3]

with open(file_pathA, "r", encoding="utf-8-sig") as file:
    dataA = json.load(file)
    scoreA = [list(item.values())[0]["Score"] for item in dataA]

with open(file_pathB, "r", encoding="utf-8-sig") as file:
    dataB = json.load(file)
    scoreB = [item["Score"] for item in dataＢ]

with open(file_pathC, "r", encoding="utf-8-sig") as file:
    dataC = json.load(file)
    scoreC = [item["prediction_rate"] for item in dataC]



diff = 0
count = 0

for i in range(0, len(scoreB)):
    diff += (scoreA[i] - scoreB[i]) ** 2
    count += 1

print(f"Pink MSE : {(diff / count):.3f}")

diff = 0
count = 0

for i in range(0, len(scoreC)):
    diff += (scoreA[i] - scoreC[i]) ** 2
    count += 1

print(f"PixleLM MSE: {(diff / count):.3f}")