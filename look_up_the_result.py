import json
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM look up result")
    parser.add_argument("--type", default="seg", type=str)
    parser.add_argument("--range", default=None, type=str)
    parser.add_argument("--text_silence", action="store_true", default=False)
    parser.add_argument("--result_file", default="result_finish_v3.json", type=str)
    return parser.parse_args()


args = parse_args()
with open(args.result_file, 'r') as f:
    data = json.load(f)
if args.type == "all":
    p_rate = 0
    b_rate = 0
    t_rate = 0
    for key, values in data.items():
        try:
            p_rate +=  values["prediction_rate"]
            b_rate += values["box_iou"]
            t_rate += 1.
        except:
            continue
    print("Overall prediction rate: ", p_rate / t_rate)
    print("Overall box iou: ", b_rate / t_rate)

elif args.type == "seg":
    start, end = args.range.split(";")[0], args.range.split(";")[1]
    for key, values in data.items():
        try:
            if int(key.split("_")[0]) < int(start) or int(key.split("_")[0]) > int(end):
                continue
            print(f"Consider {key}:")
            if args.text_silence:
                print("text: ", values["text_output"])
            print("prediction_rate: ", values["prediction_rate"])
            print("box_iou: ", values["box_iou"])
            print("----------------------------------")
        except:
            print(f"No data in {key}.", file=sys.stderr)



