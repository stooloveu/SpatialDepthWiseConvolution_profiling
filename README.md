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
| 16         	|      1 	| 3          	| 1.2134619411         	| 1.9449484079         	|
| 32         	|      1 	| 3          	| 1.2480890052         	| 1.7657131765         	|
| 64         	|      1 	| 3          	| 1.2613883163         	| 1.7746661121         	|
| 96         	|      1 	| 3          	| 1.4767474365         	| 1.7701507928         	|
| 128        	|      1 	| 3          	| 1.5892915779         	| 1.7736582884         	|

Different batch size:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 1          	| 1.4754545049         	| 1.7114487926         	|
| 128        	|      1 	| 2          	| 1.5200951403         	| 1.6836418586         	|
| 128        	|      1 	| 3          	| 1.5892915779         	| 1.7736582884         	|
| 128        	|      1 	| 5          	| 1.5726599499         	| 1.7356732435         	|
| 128        	|      1 	| 9          	| 1.6288659791         	| 1.7333621289         	|

Different stride:

| multiplier 	| stride 	| batch size 	| speed-up ratio (GPU) 	| speed-up ratio (CPU) 	|
|------------	|--------	|------------	|----------------------	|----------------------	|
| 128        	|      1 	| 3          	| 1.5892915779         	| 1.7736582884         	|
| 128        	|      2 	| 3          	| 1.5036589081         	| 1.7777762794         	|
| 128        	|      3 	| 3          	| 1.5286214587         	| 1.7874905326         	|
| 128        	|      4 	| 3          	| 1.5308350952         	| 1.9108370174         	|
