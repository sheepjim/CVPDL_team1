## Install

```
conda env create -f environment.yml
```

## Run
```
python3 unsolth_llama3_cv.py
```

if encouter RUNTIME Error for unsolth

find
```sh
$ ~/miniconda3/envs/llama3_for_cv/lib/python3.9/site-packages/unsloth/models/llama.py
```

comment out line 1645

```python
#raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
```
