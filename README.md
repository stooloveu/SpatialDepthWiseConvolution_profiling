# SpatialDepthWiseConvolution_profiling
Profiling of SpatialDepthWiseConvolution for Torch (CUDA &amp; CPU)
## Installing from source
### Installing 'nn' with SpatialDepthWiseConvolution
```bash
git clone https://github.com/stooloveu/nn
cd nn
luarocks make rocks/nn-scm-1.rockspec
```
### Installing 'cunn' with SpatialDepthWiseConvolution
```bash
git clone https://github.com/stooloveu/cunn
cd cunn
luarocks make rocks/cunn-scm-1.rockspec
```
### Trouble-shooting
If any errors occur when installing, try updating [Torch](https://github.com/torch/).

## Results
Tested with the following settings:

| Constant variable      	| Value 	|
|------------------------	|-------	|
| stride                 	|     1 	|
| input height and width 	|   299 	|
| kernel size            	|     3 	|
| padding                	|     1 	|
| nInputPlane            	|     3 	|

Different multiplier:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 16         	|      1 	| 3          	| 1.213               	| 1.945                	|
| 32         	|      1 	| 3          	| 1.242               	| 1.766                	|
| 64         	|      1 	| 3          	| 1.264               	| 1.775                	|
| 96         	|      1 	| 3          	| 1.486                	| 1.770                	|
| 128        	|      1 	| 3          	| 1.589                	| 1.774                	|

Different batch size:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 1          	| 1.471                	| 1.711                	|
| 128        	|      1 	| 2          	| 1.524                	| 1.683                	|
| 128        	|      1 	| 3          	| 1.591                	| 1.773               	|
| 128        	|      1 	| 5          	| 1.562                	| 1.735                	|
| 128        	|      1 	| 9          	| 1.611                	| 1.733                	|

Different stride:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 3          	| 1.584                	| 1.773                	|
| 128        	|      2 	| 3          	| 1.524                	| 1.777                	|
| 128        	|      3 	| 3          	| 1.531                	| 1.787                	|
| 128        	|      4 	| 3          	| 1.527                	| 1.910                	|
