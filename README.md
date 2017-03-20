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
| 16         	|      1 	| 3          	| 1.289               	| 1.945                	|
| 32         	|      1 	| 3          	| 1.301               	| 1.766                	|
| 64         	|      1 	| 3          	| 1.321               	| 1.775                	|
| 96         	|      1 	| 3          	| 1.599                	| 1.770                	|
| 128        	|      1 	| 3          	| 1.601                	| 1.774                	|

Different batch size:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 1          	| 1.501                	| 1.711                	|
| 128        	|      1 	| 2          	| 1.544                	| 1.683                	|
| 128        	|      1 	| 3          	| 1.601                	| 1.773               	|
| 128        	|      1 	| 5          	| 1.607                	| 1.735                	|
| 128        	|      1 	| 9          	| 1.621                	| 1.733                	|

Different stride:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 3          	| 1.611                	| 1.773                	|
| 128        	|      2 	| 3          	| 1.589                	| 1.777                	|
| 128        	|      3 	| 3          	| 1.624                	| 1.787                	|
| 128        	|      4 	| 3          	| 1.602                	| 1.910                	|
