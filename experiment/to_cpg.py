import re
import argparse
import ast

parser = argparse.ArgumentParser()

parser.add_argument(
    "file", type=str,
    help="file to process"
)

filepath = parser.parse_args().file

pattern = r'Performing Generation 199 \/ 200[\s\S]*?\(([\s\S]*?)\)'

with open(filepath) as file:

    cpgs = []
    for match in re.findall(pattern, file.read()):
        cpgs.append(
            ast.literal_eval(
                match.replace("\n", "").replace(" ", "")))
    print("Copy this array. Be careful with the index as it must be retained.")
    print(cpgs)