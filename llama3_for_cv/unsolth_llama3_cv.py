from unsloth import FastLanguageModel
import outlines
import torch
from transformers import TextStreamer
import os
from transformers import ConstrainedBeamSearchScorer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, DisjunctiveConstraint, PhrasalConstraint
from typing import List, Tuple, Dict
from tqdm import tqdm
import random


# Load model and tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_device('cuda:0')
torch.backends.cudnn.enabled = False

import time

few_shots = [
    '\{"caption": "A group of friends enjoying a picnic in a park on a sunny day.", "annos": [{"caption": "friends"}, {"caption": "picnic"}, {"caption": "park"}]\}',
    '\{"caption": "A scientist working in a laboratory with a microscope and test tubes.", "annos": [{"caption": "scientist"}, {"caption": "microscope"}, {"caption": "test tubes"}]\}',
    '{"caption": "A mountain landscape with a clear blue sky and a flowing river.", "annos": [{"caption": "mountain"}, {"caption": "blue sky"}, {"caption": "river"}]\}',
    '\{"caption": "A programmer coding on a laptop in a modern workspace.", "annos": [{"caption": "programmer"}, {"caption": "laptop"}, {"caption": "workspace"}]\}',
    '\{"caption": "A teacher explaining a math problem on a blackboard to students.", "annos": [{"caption": "teacher"}, {"caption": "blackboard"}, {"caption": "students"}]\}',
    '\{"caption": "A bustling city street at night with bright neon signs and moving traffic.", "annos": [{"caption": "city street"}, {"caption": "neon signs"}, {"caption": "traffic"}]\}',
    '\{"caption": "A street vendor selling fresh fruits and vegetables at a market.", "annos": [{"caption": "street vendor"}, {"caption": "fruits"}, {"caption": "vegetables"}]\}',
    '\{"caption": "A soccer player scoring a goal during an intense match.", "annos": [{"caption": "soccer player"}, {"caption": "goal"}, {"caption": "match"}]\}',
    '\{"caption": "An artist painting a colorful landscape on a large canvas.", "annos": [{"caption": "artist"}, {"caption": "landscape"}, {"caption": "canvas"}]\}',
    '\{"caption": "A spaceship flying through a galaxy filled with stars and planets.", "annos": [{"caption": "spaceship"}, {"caption": "stars"}, {"caption": "planets"}]\}'
]




prompt_template = """You are a excellent imaginer, very good at imagining a graph. In the following, please generate a figure with no more than 3 objects in. The following are the example:

{}<|eot_id|>

{}<|eot_id|>
"""

def generate_template():
    random_shots = random.sample(few_shots, 2)
    return prompt_template.format(random_shots[0], random_shots[1])
    


def generate_one_document(model, tokenizer):
    query = generate_template()
#    print(query)

    model_inputs = tokenizer(query, return_tensors="pt").to(model.device)

    FastLanguageModel.for_inference(model)
    outputs = model.generate(**model_inputs, do_sample=True, max_new_tokens = 1024, top_p=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



## Secondly, generate psuedo query...
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
)
for i in tqdm(range(100), total=100):
    out = generate_one_document(model, tokenizer).split("\{")[-1].split("\}")[0]
    few_shots.append("\{" + out + "\}")
    out = "{" + out + "}\n"
    print(out)
    with open("results.jsonl", 'a') as f:
        f.write(out)


