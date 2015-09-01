require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'inputs/starry_night.jpg', 'Style target image')
cmd:option('-content_image', 'inputs/lake.jpg', 'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 0.1)
cmd:option('-style_weight', 1.0)
cmd:option('-learning_rate', 5.0)
cmd:option('-momentum', 0.9)
cmd:option('-num_iterations', 1000)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')


local function main(params)
  if params.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
  end
  
  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):float()
  if params.gpu >= 0 then
    cnn:cuda()
  end
  
  local content_image = grayscale_to_rgb(image.load(params.content_image))
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_image = grayscale_to_rgb(image.load(params.style_image))
  style_image = image.scale(style_image, params.image_size, 'bilinear')
  local style_image_caffe = preprocess(style_image):float()
  
  if params.gpu >= 0 then
    content_image_caffe = content_image_caffe:cuda()
    style_image_caffe = style_image_caffe:cuda()
  end
  
  -- Hardcode these for now
  local content_layers = {21}
  local style_layers = {2, 7, 12, 15, 21}
  local style_layer_weights = {1e3, 1e2, 1e2, 1e1, 1e1}

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      net:add(cnn:get(i))
      if i == content_layers[next_content_idx] then
        local target = net:forward(content_image_caffe):clone()
        local loss_module = nn.ContentLoss(params.content_weight, target):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end
      if i == style_layers[next_style_idx] then
        local gram = GramMatrix():float()
        if params.gpu >= 0 then
          gram = gram:cuda()
        end
        local target_features = net:forward(style_image_caffe):clone()
        local target = gram:forward(target_features)
        target:div(target_features:nElement())
        local weight = params.style_weight * style_layer_weights[next_style_idx]
        local loss_module = nn.StyleLoss(weight, target):float()
        if params.gpu >= 0 then
          loss_module:cuda()
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
  end
  
  -- Initialize the image
  -- local img = torch.randn(content_image:size()):float()
  local img = content_image_caffe:clone():float()
  if params.gpu >= 0 then
    img = img:cuda()
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()
  
  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  local function feval(x)
    net:forward(x)
    local grad = net:backward(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    return loss, grad
  end
  
  -- Run optimization.
  local optim_state = {
    learningRate = params.learning_rate,
    momentum = params.momentum,
  }
  for t = 1, params.num_iterations do
    -- Take a step
    local x, losses = optim.sgd(feval, img, optim_state)
    assert(x == img)

    -- Maybe print the iteration number and losses
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
    end

    -- Maybe save the intermediate output
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end
      image.save(filename, disp)
    end
  end
end
  

function build_filename(output_image, iteration)
  local idx = string.find(output_image, '%.')
  local basename = string.sub(output_image, 1, idx - 1)
  local ext = string.sub(output_image, idx)
  return string.format('%s_%d%s', basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


function grayscale_to_rgb(img)
  local c, h, w = img:size(1), img:size(2), img:size(3)
  if c == 1 then
    return img:expand(3, h, w):contiguous()
  else
    return img
  end
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local params = cmd:parse(arg)
main(params)
