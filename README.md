# CVPDL -- Created Vision-generation Preference D via LM

## 1. Scenario Prompt Generation

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
{"_COMMENT": "So on so forth"}
```

## 2. Generate layout
```
python layout.py ${input} ${output}
```
```
{
    "caption": "A person sitting on a beach with a sunset in the background.",
    "width": 512,
    "height": 512,
    "annos": [
        {
            "bbox": [
                0,
                200,
                143,
                187
            ],
            "mask": [],
            "category_name": "",
            "caption": "person"
        },
        {
            "bbox": [
                0,
                0,
                102,
                193
            ],
            "mask": [],
            "category_name": "",
            "caption": "beach"
        },
        {
            "bbox": [
                200,
                0,
                164,
                154
            ],
            "mask": [],
            "category_name": "",
            "caption": "sunset"
        }
    ]
}
```

## 3. Image Generation -- Instance Diffusion


## 4-1. PixelLM BenchMark

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

### File Input Structure

- The file input should be structured to
```
prompts_path
|
|---1.json
|---2.json
|---3.json
|.......
|
image_path
|
|---1.json
|     |-----gc7.5-seed0-alpha0.8
|     |           |-----0_xl_s0.4_n20.png
|                 |-----1_xl_s0.4_n20.png
|                 |-----......
|......
```


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

## 4-2 Pink Bench Mark

## 4-3 Human Bench Mark

## 5. Calculate MSE
```
python eval.py ${human_score.json} ${PixelLM.json} ${PINK.json}  
```
