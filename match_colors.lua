-- Based on Leon Gatys's work "Controlling Perceptual Factors in Neural Style Transfer":
-- https://arxiv.org/abs/1611.07865
-- his code:
-- https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ColourControl.ipynb
-- and on ProGamerGov's issue and code:
-- https://github.com/jcjohnson/neural-style/issues/376

require 'torch'
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Coloring style image to content palette.')
cmd:text()
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content source image')
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-color_function', 'pca',
           'Color matching function: pca, chol, sym')
cmd:option('-output_image', 'out.png')
cmd:text()


local function main(params)
  local content_img = image.load(params.content_image, 3)
  local style_img = image.load(params.style_image, 3)

  local output_img = match_color(style_img, content_img, params.color_function)
  image.save(params.output_image, output_img)
end


function match_color(target_img, source_img, mode, eps)
  -- Matches the colour distribution of the target image to that of the source image
  -- using a linear transform.
  -- Images are expected to be of form (c,w,h) and float in [0,1].
  -- Modes are chol, pca or sym for different choices of basis.

  mode = mode or 'pca'
  eps = eps or 1e-5
  local eyem = torch.eye(source_img:size(1)):mul(eps)

  local mu_s = torch.mean(source_img, 3):mean(2)
  local s = source_img - mu_s:expandAs(source_img)
  s = s:view(s:size(1), s[1]:nElement())
  local Cs = s * s:t() / s:size(2) + eyem

  local mu_t = torch.mean(target_img, 3):mean(2)
  local t = target_img - mu_t:expandAs(target_img)
  t = t:view(t:size(1), t[1]:nElement())
  local Ct = t * t:t() / t:size(2) + eyem

  local ts
  if mode == 'chol' then
    local chol_s = torch.potrf(Cs):t()
    local chol_t = torch.potrf(Ct):t()
    ts = chol_s * torch.inverse(chol_t) * t
  elseif mode == 'pca' then
    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local eva_s, eve_s = torch.symeig(Cs, 'V', 'L')
    local Qs = eve_s * torch.diag(eva_s):sqrt() * eve_s:t()
    ts = Qs * torch.inverse(Qt) * t
  elseif mode == 'sym' then
    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local Qt_Cs_Qt = Qt * Cs * Qt
    local eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, 'V', 'L')
    local QtCsQt = eve_QtCsQt * torch.diag(eva_QtCsQt):sqrt() * eve_QtCsQt:t()
    ts = torch.inverse(Qt) * QtCsQt * torch.inverse(Qt) * t
  else
    assert((mode == 'chol' or
            mode == 'pca' or
            mode == 'sym'),
            'Unknown color matching mode. Stop.')
  end

  local matched_img = ts:viewAs(target_img) + mu_s:expandAs(target_img)
  -- matched_img = image.minmax{tensor=matched_img, min=0, max=1}
  return matched_img:clamp(0, 1)
end


local params = cmd:parse(arg)
main(params)
