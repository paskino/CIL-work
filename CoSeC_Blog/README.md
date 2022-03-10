# Code to reproduce the images in the blog post

## download the data
The chocolate egg can be downloaded from from https://zenodo.org/record/4822516#.YioQWOjP29Y (egg2)
## install `cil`

You can install `cil` via conda using the following command

`conda create --name cosec_blog_env -c conda-forge -c intel -c ccpi cil tigre`

## run the code

After you've downloaded the data and installed `cil`, modify the path to the data file and you can run the code as

```bash

python cosec_blog.py --recon
```

