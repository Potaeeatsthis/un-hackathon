import json

with open("./results.txt/Teeramate-Resume/results.json") as f:
    data = json.load(f)

# Get the first file's results
for filename, pages in data.items():
    for page in pages:
        for line in page["text_lines"]:
            print(line["text"]) 
