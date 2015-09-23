local ffi = require 'ffi'
require 'loadcaffe'
local C = loadcaffe.C

--[[
  Most of this function is copied from
  https://github.com/szagoruyko/loadcaffe/blob/master/loadcaffe.lua
  with some horrible horrible hacks added by Justin Johnson to
  make it possible to load VGG-19 without any CUDA dependency.
--]]
local function loadcaffe_load(prototxt_name, binary_name, backend)
  local backend = backend or 'nn'
  local handle = ffi.new('void*[1]')

  -- loads caffe model in memory and keeps handle to it in ffi
  local old_val = handle[1]
  C.loadBinary(handle, prototxt_name, binary_name)
  if old_val == handle[1] then return end

  -- transforms caffe prototxt to torch lua file model description and 
  -- writes to a script file
  local lua_name = prototxt_name..'.lua'

  -- C.loadBinary creates a .lua source file that builds up a table
  -- containing the layers of the network. As a horrible dirty hack,
  -- we'll modify this file when backend "nn-cpu" is requested by
  -- doing the following:
  --
  -- (1) Delete the lines that import cunn and inn, which are always
  --     at lines 2 and 4
  local model = nil
  if backend == 'nn-cpu' then
    C.convertProtoToLua(handle, lua_name, 'nn')
    local lua_name_cpu = prototxt_name..'.cpu.lua'
    local fin = assert(io.open(lua_name), 'r')
    local fout = assert(io.open(lua_name_cpu, 'w'))
    local line_num = 1
    while true do
      local line = fin:read('*line')
      if line == nil then break end
      if line_num ~= 2  and line_num ~= 4 then
        fout:write(line, '\n')
      end
      line_num = line_num + 1
    end
    fin:close()
    fout:close()
    model = dofile(lua_name_cpu)
  else
    if backend == "clnn" then
      C.convertProtoToLua(handle, lua_name, 'nn')
      local lua_name_opencl = prototxt_name..'.opencl.lua'
      local fin = assert(io.open(lua_name), 'r')
      local fout = assert(io.open(lua_name_opencl, 'w'))
      local line_num = 1
      while true do
        local line = fin:read('*line')
        if line == nil then break end
        -- Fix for using nin_imagenet_conv.caffemodel
        if line:find("inn") then
          line = line:gsub("inn", "nn")
        end
        --[[
        -- Hack to replace CUDA libraries with openCL libs. 
        -- My machine only has an ATI Firepro V3900 so can't run CUDA libs.
        --]]--
        if line_num > 2 and line_num ~=4 then
          fout:write(line, '\n')
        elseif line_num == 1 then
          fout:write("require 'nn'", '\n')
          fout:write("require 'clnn'", '\n')
        end
        line_num = line_num + 1
      end
      fin:close()
      fout:close()
      model = dofile(lua_name_opencl)
    else
      C.convertProtoToLua(handle, lua_name, backend)
      model = dofile(lua_name)
    end
  end

  -- goes over the list, copying weights from caffe blobs to torch tensor
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
    item[2].name = item[1]
    if item[2].weight then
      local w = torch.FloatTensor()
      local bias = torch.FloatTensor()
      C.loadModule(handle, item[1], w:cdata(), bias:cdata())
      if backend == 'ccn2' then
        w = w:permute(2,3,4,1)
      end
      item[2].weight:copy(w)
      item[2].bias:copy(bias)
    end
    net:add(item[2])
  end
  C.destroyBinary(handle)

  if backend == 'cudnn' or backend == 'ccn2' then
    net:cuda()
  elseif backend == 'clnn' then
    net:cl()
  end

  return net
end

return {
  load = loadcaffe_load
}
