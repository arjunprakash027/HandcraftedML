import os
from pathlib import Path
from typing import List

def ignore_ruleset(item:Path,ignore_files:List[str]) -> bool:
    
    if item.name in ignore_files:
        return True
    
    if item.name.startswith('.') or item.name.startswith('__'):
        return True
    
    return False

def generate_markdown(directory:Path,ignore_files:List[str],level=0) -> List:
    
    markdown_lines = []

    for item in sorted(directory.iterdir()):
        
        if ignore_ruleset(item=item,ignore_files=ignore_files):
            continue
        
        indent = "  " * level

        if item.is_dir():
            markdown_lines.append(f"{indent} - **{item.name}**")
            markdown_lines.extend(generate_markdown(directory=item,ignore_files=ignore_files,level=level + 1))
        
        else:
            markdown_lines.append(f"{indent} - {item.name}")
    return markdown_lines

def traverser():

    current_file = Path(__file__).resolve()
    root_dir = current_file.parent
    ignore_files = ['volume']
    markdown_content = generate_markdown(directory=root_dir,ignore_files=ignore_files)

    print(markdown_content)
    with open("README.md", "w") as md_file:
        md_file.write("# Sandbox\n\n")
        md_file.write('''Sandbox is collection of completely random codes that almost all the time does not inter-relate with each other and are standalone random scripts that I use to learn an concept\n\n''')
        md_file.write("\n".join(markdown_content))



if __name__ == '__main__':
    traverser()