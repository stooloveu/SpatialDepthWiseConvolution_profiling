local cuda = true
local test_utility = true
local epsilon = 0.00001

require 'nn'
local SC = nn.SpatialConvolution
local SDWC = nn.SpatialDepthWiseConvolution

local function nnSpatialDepthWiseConv(
      nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
   )
   local conv = SDWC(nInputPlane, multiplier, kernel, kernel, stride, stride, padding, padding)
   conv.weight = weight
   conv.bias = bias
   return conv
end

-- Utility spatialDepthWiseConv() function -------------------------------------
local function spatialDepthWiseConv(
      nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
   )

   local conv = nn.Sequential()
   conv:add(nn.Contiguous())
   conv:add(nn.View(-1, 1, inputSize, inputSize))
   conv:add(SC(1, multiplier, kernel, kernel, stride, stride, padding, padding))

   local depthWiseConv = nn.Parallel(2, 2)
   for channel = 1, nInputPlane do
      local tempConv = conv:clone()
      tempConv:get(3).weight = weight:narrow(2, channel, 1):clone()
      tempConv:get(3).bias = bias:clone():select(2, channel):clone()
	  depthWiseConv:add(tempConv) 
   end
   depthWiseConv:add(nn.Contiguous()) 
   return depthWiseConv
end


-- torch.manualSeed(1234)
local model = nn.Sequential()
local n = 3 -- nInputPlane
local s = 299 -- input height and width
local b = 1 -- batch size
local m = 64 -- multiplier
local k = 3 -- kernel size
local p = 1 -- padding
local st = 2 -- stride

local X = torch.rand(b, n, s, s) -- 1x3x299x299 images
local weight = torch.rand(m, n, k, k) -- weight
local bias = torch.rand(m, n) -- bias

model:add(nnSpatialDepthWiseConv(n, m, k, st, p, s, weight, bias))

if test_utility then
    model_util = spatialDepthWiseConv(n, m, k, st, p, s, weight, bias)
    for epoch = 1, 10 do model_util:forward(X) end  -- warm up

    local timer = torch.Timer()  -- start a new timer object
    for epoch = 1, 1e3 do
        Y_util = model_util:forward(X)
    end
    time = timer:time().real  -- get the final time

    print('Util time:', time)
end

if cuda then
   -- Running on GPU
   require 'cutorch'
   require 'cunn'
   model:cuda()
   X = X:cuda()
end

for epoch = 1, 10 do model:forward(X) end  -- warm up

if cuda then cutorch.synchronize() end  -- wait for the GPU to finish
local timer = torch.Timer()  -- start a new timer object
for epoch = 1, 1e3 do
    Y = model:forward(X)
end

if cuda then cutorch.synchronize() end  -- wait for the GPU to finish
time = timer:time().real  -- get the final time

if cuda then print('GPU time:', time)
        else print('CPU time:', time) end

-- print(weight)
-- print(Y)
-- print(Y_util)
-- print(model_util:get(3):get(3).weight)
-- print(model_util:get(3):get(3):forward(X:narrow(2, 1, 1)))

if test_utility  then
    if cuda then Y_util = Y_util:cuda() end
    local tmp = Y_util:clone()
    tmp:csub(Y):abs()
    print('Correctness:', torch.all(tmp:lt(epsilon)))
end

net = model
