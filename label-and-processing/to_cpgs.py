import re
import argparse
import ast

parser = argparse.ArgumentParser()

parser.add_argument(
    "file", type=str,
    help="file to process"
)

parser.add_argument(
    "--g", required=True, type=int, help="The generation to select")

args = parser.parse_args()
filepath = args.file
gen = args.g


pattern = r'Performing Generation 1[\s\S]*?\(([\s\S]*?)\)'
pattern = re.compile(f"Performing Generation {gen}[\s\S]*?\(([\s\S]*?)\)")

with open(filepath) as file:

    cpgs = []
    for match in re.findall(pattern, file.read()):
        cpgs.append(
            ast.literal_eval(
                match.replace("\n", "").replace(" ", "")))
    print("Copy this array. Be careful with the index as it must be retained.")
    print(cpgs)