local ffi = require 'ffi'
require 'loadcaffe'
local C = loadcaffe.C


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
      if line_num ~= 2 then
        fout:write(line, '\n')
      end
      line_num = line_num + 1
    end
    fin:close()
    fout:close()
    model = dofile(lua_name_cpu)
  else
    C.convertProtoToLua(handle, lua_name, backend)
    model = dofile(lua_name)
  end

  -- goes over the list, copying weights from caffe blobs to torch tensor
  local net = nn.Sequential()
  local list_modules = model
  for i,item in ipairs(list_modules) do
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
  end

  return net
end

return {
  load = loadcaffe_load
}
