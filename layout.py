import json
import sys
import random
import os

def generate_non_overlapping_objects(object_count, min_size, max_size, captions):

    width, height = 512, 512
    grid_size = max_size 
    grid = [[True] * (width // grid_size) for _ in range(height // grid_size)]
    
    idx = 0
    annos = []
    
    for _ in range(object_count):
        while True:
            
            obj_width = random.randint(min_size, max_size)
            obj_height = random.randint(min_size, max_size)
            
            grid_width = (obj_width + grid_size - 1) // grid_size
            grid_height = (obj_height + grid_size - 1) // grid_size
            
            grid_x = random.randint(0, len(grid[0]) - grid_width)
            grid_y = random.randint(0, len(grid) - grid_height)
            
            if all(grid[y][x] for y in range(grid_y, grid_y + grid_height) for x in range(grid_x, grid_x + grid_width)):
                for y in range(grid_y, grid_y + grid_height):
                    for x in range(grid_x, grid_x + grid_width):
                        grid[y][x] = False

                x = grid_x * grid_size
                y = grid_y * grid_size
                annos .append({
                    "bbox": (x, y, obj_width, obj_height),
                    "mask": [],
                    "category_name": "",
                    "caption": captions[idx]["caption"]
                })
                idx += 1
                break
    
    return annos 


os.makedirs(sys.argv[2], exist_ok=True)
with open(sys.argv[1], 'r', encoding='utf-8') as file:
    idx = 1
    for line in file:
        record = json.loads(line)
        objectCount = len(record["annos"])
        annos  = generate_non_overlapping_objects(objectCount, 100, 200, record["annos"])
        output = {
            "caption": record["caption"],
            "width": 512,
            "height": 512,
            "annos": annos 
        }

        output_file = sys.argv[1] + f'/{idx}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        idx += 1

        
        