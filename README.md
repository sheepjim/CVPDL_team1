# CVPDL -- Created Vision-generation Preference D via LM

## Scenario Prompt Generation

- We utilize `ChatGPT` to generate the 10 prompts into the prompt pools.
- We utilize `llama3` via `unsolth` package to generate the scenario.

### Run
```
cd llama3_for_cv
python3 unsolth_llama3_cv.py
```

if encouter RUNTIME Error for unsolth

find
```sh
$ YOUR_PATH/python3.9/site-packages/unsloth/models/llama.py
```

comment out line 1645

```python
#raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
```

- It will generate 100 prompts and store into `llama3_for_cv/result.jsonl`.
```json
{"caption": "A serene beach scene at sunset with a wooden pier and calm sea.", "annos": [{"caption": "wooden pier"}, {"caption": "calm sea"}, {"caption": "setting sun"}]}
{"caption": "A person relaxing on a beach with a palm tree and a sailboat in the background.", "annos": [{"caption": "person"}, {"caption": "palm tree"}, {"caption": "sailboat"}]}
...
```

## Generate layout
```
python layout.py ${input} ${output}
```

## PixelLM BenchMark

### Preprocessing
```sh
$ git clone https://github.com/MaverickRen/PixelLM.git
$ mv ./inference_for_cv.sh ./PixelLM
$ mv ./chat_for_cv.py ./PixelLM
$ mv ./look_up_the_result.py/PixelLM
$ mkdir ./PixelLM/vis_output 
$ mkdir ./PixelLM/mask_result
```

- Download the `PixelLM-7B` checkpoint via the instruction of their README.


### Run
```sh
cd PixelLM
sh inference_for_cv.sh
```

### Look up the result

- Overall average
```sh
python3 look_up_the_result.py --type "all" --result_file "result.json"
```

- Specific range

```sh
python3 look_up_the_result.py --range "[start];[end]" --result_file "result.json"
```

## Calculate MSE
```
python eval.py ${human_score.json} ${PixelLM.json} ${PINK.json}  
```