### Setup with Conda
We suggest to create a new conda environment. 

```bash
# create environment
conda create --name CIM python=3.6
conda activate CIM
```

Then, install the following packages:

- [PyTorch](https://pytorch.org/): `pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html`
- [mmcv](https://openmmlab.com/): `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html`

Finally, install other relevant dependencies.
```bash
pip install -r requirements.txt
```

### ⚠️Warning
If you want to use other versions of python, you may encounter an error ```No module named 'utils.cython_bbox'```. This is because this repository is not compiled in other versions of python.
The compilation steps are as follows.

```
cd ./lib
python setup.py build_ext --inplace
```

Please refer to [issue2](https://github.com/ZechengLi19/CIM/issues/2) for more details.