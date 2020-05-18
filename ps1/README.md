# Problem Set #1
## Setup for Written Parts
- Use an editor with built-in typesetting such as  TeXMaker or TeXShop
## Setup for Coding Parts
1. Install [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation)
2. cd into src, run `conda env create -f environment.yml`
  - This creates a Conda environment called `cs229`
3. Run `source activate cs229`
  - This activates the `cs229` environment
  - Do this each time you want to write/test your code
4. If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/cs229/bin/python`
  - Select `OK` then `Apply`
 
