# Problem Set #1
## Setup for Written Parts

- Use an editor with built-in typesetting such as  TeXMaker or TeXShop
## Setup for Coding Parts

1. Intsall Miniconda
2. cd into src, run <code> conda env create -f environment.yml </code>
- This creates a Conda environment called <code> cs229 </code>
3. Run <code> source activate cs229 </code>
- This activates the <code> cs229 </code> environment
- Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
- Open the <code> src </code> directory in PyCharm
- Go to PyCharm > Preferences > Project > Project interpreter
- Click the gear in the top-right corner, then Add
- Select Conda environment > Existing environment > Button on the right with …
- Select /Users/YOUR_USERNAME/miniconda3/envs/cs229/bin/python
- Select OK then Apply
