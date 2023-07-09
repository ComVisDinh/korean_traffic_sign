#! /usr/bin/env luajit

require 'torch'

io.stdout:setvbuf('no')
for i = 1,#arg do
   io.write(arg[i] .. ' ')
end
io.write('\n')

dataset = 'kitti2015'
arch = 'slow'

cmd = torch.CmdLine()
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-debug', false)
cmd:option('-d', 'kitti | mb')
cmd:option('-a', 'train_all', 'train_all')
cmd:option('-net_fname', '')
cmd:option('-make_cache', false)
cmd:option('-use_cache', false)
cmd:option('-print_args', false)
cmd:option('-sm_terminate', '', 'terminate the stereo method after this step')
cmd:option('-sm_skip', '', 'which part of the stereo method to skip')
cmd:option('-tiny', false)
cmd:option('-subset', 1)

cmd:option('-left', '')
cmd:option('-right', '')
cmd:option('-disp_max', '')


cmd:option('-hflip', 0)
cmd:option('-vflip', 0)
cmd:option('-rotate', 28)
cmd:option('-hscale', 0.8)
cmd:option('-scale', 0.8)
cmd:option('-trans', 1)
cmd:option('-hshear', 0.1)
cmd:option('-brightness', 1.3)
cmd:option('-contrast', 1.1)
cmd:option('-d_vtrans', 2)
cmd:option('-d_rotate', 4)
cmd:option('-d_hscale', 0.9)
cmd:option('-d_hshear', 0.3)
cmd:option('-d_brightness', 0.7)
cmd:option('-d_contrast', 1.1)

cmd:option('-rect', 'imperfect')
cmd:option('-color', 'gray')

if arch == 'slow' then	
	cmd:option('-ds', 2001)
	cmd:option('-d_exp', 0.2)
	cmd:option('-d_light', 0.2)
	
	cmd:option('-l1', 4)
    cmd:option('-fm', 112)
    cmd:option('-ks', 3)
    cmd:option('-l2', 3)
    cmd:option('-nh2', 384)
    cmd:option('-lr', 0.003)
    cmd:option('-bs', 128)
    cmd:option('-mom', 0.9)
    cmd:option('-true1', 1)
    cmd:option('-false1', 4)
    cmd:option('-false2', 10)

	cmd:option('-L1', 14)
	cmd:option('-tau1', 0.02)
	cmd:option('-cbca_i1', 2)
	cmd:option('-cbca_i2', 16)
	cmd:option('-pi1', 1.3)
	cmd:option('-pi2', 13.9)
	cmd:option('-sgm_i', 1)
	cmd:option('-sgm_q1', 4.5)
	cmd:option('-sgm_q2', 2)
	cmd:option('-alpha1', 2.75)
	cmd:option('-tau_so', 0.13)
	cmd:option('-blur_sigma', 1.67)
	cmd:option('-blur_t', 2)
end

opt = cmd:parse(arg)

if opt.print_args then   
   print(opt.false1, 'dataset_neg_low')
   print(opt.false2, 'dataset_neg_high')
   print(opt.true1, 'dataset_pos_low')
   print(opt.tau1, 'cbca_intensity')
   print(opt.L1, 'cbca_distance')
   print(opt.cbca_i1, 'cbca_num_iterations_1')
   print(opt.cbca_i2, 'cbca_num_iterations_2')
   print(opt.pi1, 'sgm_P1')
   print(opt.pi1 * opt.pi2, 'sgm_P2')
   print(opt.sgm_q1, 'sgm_Q1')
   print(opt.sgm_q1 * opt.sgm_q2, 'sgm_Q2')
   print(opt.alpha1, 'sgm_V')
   print(opt.tau_so, 'sgm_intensity')
   print(opt.blur_sigma, 'blur_sigma')
   print(opt.blur_t, 'blur_threshold')
   os.exit()
end


require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'
cudnn.benchmark = true

include('Margin2.lua')
include('Normalize2.lua')
include('BCECriterion2.lua')
--include('DiceCriterion.lua')
include('StereoJoin.lua')
include('StereoJoin1.lua')
include('SpatialConvolution1_fw.lua')


torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

cmd_str = dataset .. '_' .. arch
for i = 1,#arg do
   cmd_str = cmd_str .. '_' .. arg[i]
end

function isnan(n)
   return tostring(n) == tostring(0/0)
end

function fromfile(fname)
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   if #dim == 1 and dim[1] == 0 then
      return torch.Tensor()
   end

   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      local s=1
      for i = 1,#dim do	 
	  	s = s * dim[i]
      end
	
      x = torch.FloatTensor(torch.FloatStorage(s))
      torch.DiskFile(fname,'r'):binary():readFloat(x:storage())
   elseif type == 'int32' then
      local s=1
      for i = 1,#dim do	 
	 	s = s * dim[i]
      end
	
      x = torch.IntTensor(torch.IntStorage(s))
      torch.DiskFile(fname,'r'):binary():readInt(x:storage())
   elseif type == 'int64' then
      local s=1
      for i = 1,#dim do	 
	 	s = s * dim[i]
      end
	
      x = torch.LongTensor(torch.LongStorage(s))
      torch.DiskFile(fname,'r'):binary():readLong(x:storage())
   else 
      assert(false)
   end

   x = x:reshape(torch.LongStorage(dim))
   return x
end

function get_window_size(net)
   local ws = 64
   return ws
end

function savePNG(fname, x, isvol)
   local pred
   local pred_jet = torch.Tensor(1, 3, x:size(3), x:size(4))

   if isvol == true then
      pred = torch.CudaTensor(1, 1, x:size(3), x:size(4))
      adcensus.spatial_argmin(x, pred)
   else
      pred = x:double():add(1)
   end
   adcensus.grey2jet(pred[{1,1}]:div(disp_max):double(), pred_jet)
   image.savePNG(fname, pred_jet[1])
end

function saveOutlier(fname, x0, outlier)
   local img = torch.Tensor(1,3,height,img_width)
   img[{1,1}]:copy(x0)
   img[{1,2}]:copy(x0)
   img[{1,3}]:copy(x0)
   for i=1,height do
      for j=1,img_width do
         if outlier[{1,1,i,j}] == 1 then
            img[{1,1,i,j}] = 0
            img[{1,2,i,j}] = 1
            img[{1,3,i,j}] = 0
         elseif outlier[{1,1,i,j}] == 2 then
            img[{1,1,i,j}] = 1
            img[{1,2,i,j}] = 0
            img[{1,3,i,j}] = 0
         end
      end
   end
   image.savePNG(fname, img[1])
end

function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end

function print_net(net)
   local s
   local t = torch.typename(net) 
   if t == 'cudnn.SpatialConvolution' then
      print(('conv(in=%d, out=%d, k=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW))
   elseif t == 'nn.SpatialConvolutionMM_dsparse' then
      print(('conv_dsparse(in=%d, out=%d, k=%d, s=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW, net.sW))
   elseif t == 'cudnn.SpatialMaxPooling' then
      print(('max_pool(k=%d, d=%d)'):format(net.kW, net.dW))
   elseif t == 'nn.StereoJoin' then
      print(('StereoJoin(%d)'):format(net.disp_max))
   elseif t == 'nn.Margin2' then
      print(('Margin2(margin=%f, pow=%d)'):format(opt.m, opt.pow))
   elseif t == 'nn.GHCriterion' then
      print(('GHCriterion(m_pos=%f, m_neg=%f, pow=%d)'):format(opt.m_pos, opt.m_neg, opt.pow))
   elseif t == 'nn.Sequential' then
      for i = 1,#net.modules do
         print_net(net.modules[i])
      end
   else
      print(net)
   end
end

function clean_net(net)
   net.output 		= torch.CudaTensor()
   net.gradInput 	= nil
   net.weight_v 	= nil
   net.bias_v 		= nil
   net.gradWeight 	= nil
   net.gradBias 	= nil
   net.iDesc 		= nil
   net.oDesc 		= nil

   net.finput 		= torch.CudaTensor()
   net.fgradInput 	= torch.CudaTensor()
   net.tmp_in 		= torch.CudaTensor()
   net.tmp_out 		= torch.CudaTensor()

   if net.modules then
      for _, module in ipairs(net.modules) do
         clean_net(module)
      end
   end

   return net
end

function save_net(epoch)
   if arch == 'slow' then
      obj = {clean_net(net1_te), clean_net(net2_te), clean_net(net3_te), clean_net(net4_te), clean_net(net_c_te), clean_net(net_te_remain), opt}   
   end

   if epoch == 0 then
      fname = ('net/sign32_%s.t7'):format(cmd_str)
   else
      fname = ('net/sign32_%s_%d.t7'):format(cmd_str, epoch)
   end

   torch.save(fname, obj, 'ascii')

   return fname
end

function mul32(a,b)
	return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
end

function make_patch(src, dst, dim3, dim4, scale, phi, trans, hshear, brightness, contrast)
	local m = {1, 0, -dim4, 0, 1, -dim3}
	m = mul32({1, 0, trans[1], 0, 1, trans[2]}, m) -- translate
	m = mul32({scale[1], 0, 0, 0, scale[2], 0}, m) -- scale
	
	local c = math.cos(phi)
	local s = math.sin(phi)
	m = mul32({c, s, 0, -s, c, 0}, m)     -- rotate
	m = mul32({1, hshear, 0, 0, 1, 0}, m) -- shear
	m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
	m = torch.FloatTensor(m)

	cv.warp_affine(src, dst, m)
	dst:mul(contrast):add(brightness)
end



----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------


------------------------
-- 1. load training data
------------------------
n_input_plane = 1

--if dataset == 'kitti2015' then

kt_height   = 350
kt_width    = 1242
kt_disp_max = 228
kt_n_te 	= 200


kt_X0       = fromfile('data.kitti2015/x0.bin')
kt_X1       = fromfile('data.kitti2015/x1.bin')
kt_dispnoc  = fromfile('data.kitti2015/dispnoc.bin')
kt_metadata = fromfile('data.kitti2015/metadata.bin')

kt_tr 		= fromfile('data.kitti2015/tr.bin')
kt_te 		= fromfile('data.kitti2015/te.bin')
kt_nnz_tr 	= fromfile('data.kitti2015/nnz_tr.bin')
kt_nnz_te 	= fromfile('data.kitti2015/nnz_te.bin')

kt_nnz  = torch.cat(kt_nnz_tr, kt_nnz_te, 1)
kt_perm = torch.randperm(kt_nnz:size(1))
print("KT data size: " .. kt_nnz:size(1))



--elseif dataset == 'mb' then
	
mb_data_dir 	= ('data.mb.%s_%s'):format(opt.rect, opt.color)
mb_te 			= fromfile(('%s/te.bin'):format(mb_data_dir))
mb_metadata 	= fromfile(('%s/meta.bin'):format(mb_data_dir))
mb_nnz_tr 		= fromfile(('%s/nnz_tr.bin'):format(mb_data_dir))
mb_nnz_te 		= fromfile(('%s/nnz_te.bin'):format(mb_data_dir))

mb_fname_submit = {}
for mb_line in io.open(('%s/fname_submit.txt'):format(mb_data_dir), 'r'):lines() do
	table.insert(mb_fname_submit, mb_line)
end

mb_X 	   = {}
mb_dispnoc = {}
mb_height  = 1500
mb_width   = 1000

for mb_n = 1, mb_metadata:size(1) do
	mb_XX    = {}
	mb_light = 1

	while true do
		mb_fname = ('%s/x_%d_%d.bin'):format(mb_data_dir, mb_n, mb_light)
		if not paths.filep(mb_fname) then
			break
		end

		table.insert(mb_XX, fromfile(mb_fname))
		mb_light = mb_light + 1			
	end
	table.insert(mb_X, mb_XX)

	mb_fname = ('%s/dispnoc%d.bin'):format(mb_data_dir, mb_n)
	if paths.filep(mb_fname) then
		table.insert(mb_dispnoc, fromfile(mb_fname))
	end
end

mb_nnz  = torch.cat(mb_nnz_tr, mb_nnz_te, 1)
mb_perm = torch.randperm(mb_nnz:size(1))

print("MB data size: " .. mb_nnz:size(1))

--end

fm = torch.totable(torch.linspace(opt.fm, opt.fm, opt.l1):int())



-----------------------------------------------------
-------------- 2. network architecture --------------
-----------------------------------------------------


-- network for training
-- net_patch1
net1_tr = nn.Sequential()
for i = 1,#fm do
	net1_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
	net1_tr:add(cudnn.ReLU(true))
	if i ==1 then
	    net1_tr:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net1_tr:add(nn.Reshape(opt.bs*2, fm[#fm]))

-- net_patch2
net2_tr = nn.Sequential()
for i = 1,#fm do
	net2_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
	net2_tr:add(cudnn.ReLU(true))
	if i ==1 then
	    net2_tr:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net2_tr:add(nn.Reshape(opt.bs*2, fm[#fm]))

-- net_patch3
net3_tr = nn.Sequential()
for i = 1,#fm do
	net3_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
	net3_tr:add(cudnn.ReLU(true))
	if i ==1 then
	    net3_tr:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net3_tr:add(nn.Reshape(opt.bs*2, fm[#fm]))

-- net_patch4
net4_tr = nn.Sequential()
for i = 1,#fm do
	net4_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
	net4_tr:add(cudnn.ReLU(true))
	if i ==1 then
	    net4_tr:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net4_tr:add(nn.Reshape(opt.bs*2, fm[#fm]))

-- net_patch_c
net_c_tr = nn.Sequential()
for i = 1,#fm do
	net_c_tr:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks))
	net_c_tr:add(cudnn.ReLU(true))
	if i ==1 then
	    net_c_tr:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net_c_tr:add(nn.Reshape(opt.bs*2, fm[#fm]))



-- parallel
parallel_training = nn.ParallelTable()
parallel_training:add(net1_tr)
parallel_training:add(net2_tr)
parallel_training:add(net3_tr)
parallel_training:add(net4_tr)
parallel_training:add(net_c_tr)

-- main model
main_training = nn.Sequential()
main_training:add(parallel_training)
main_training:add(nn.JoinTable(1,1))
main_training:add(nn.Reshape(opt.bs, 10*fm[#fm]))
for i = 1,opt.l2 do
	main_training:add(nn.Linear(i == 1 and 10*fm[#fm] or opt.nh2, opt.nh2))
	main_training:add(cudnn.ReLU(true))
end
main_training:add(nn.Linear(opt.nh2, 1))
main_training:add(cudnn.Sigmoid(false))
main_training:cuda()  
criterion = nn.BCECriterion2():cuda()



--print(main_training)



-- load pretrained and set to train_net 
load_net      = torch.load('net/sign32_kitti2015_slow_14.t7', 'ascii')
load_net1_te  = load_net[1]
load_net2_te  = load_net[2]
load_net3_te  = load_net[3]
load_net4_te  = load_net[4]
load_netc_te  = load_net[5]
load_net_te_remain = load_net[6]


load_net_te_all = {}
for i, v in ipairs(load_net1_te.modules)  do table.insert(load_net_te_all, v) end
for i, v in ipairs(load_net2_te.modules)  do table.insert(load_net_te_all, v) end
for i, v in ipairs(load_net3_te.modules)  do table.insert(load_net_te_all, v) end
for i, v in ipairs(load_net4_te.modules)  do table.insert(load_net_te_all, v) end
for i, v in ipairs(load_netc_te.modules)  do table.insert(load_net_te_all, v) end
for i, v in ipairs(load_net_te_remain.modules) do table.insert(load_net_te_all, v) end

local load_i_tr = 1
local load_i_te = 1

-- for net1_tr
while load_i_tr <= net1_tr:size()-1 do
	local load_module_tr = net1_tr:get(load_i_tr)
	local load_module_te = load_net_te_all[load_i_te]

	if load_module_tr.weight then
		assert(load_module_te.weight:nElement() == load_module_tr.weight:nElement())
		assert(load_module_te.bias:nElement() == load_module_tr.bias:nElement())

		load_module_tr.weight = torch.CudaTensor(load_module_te.weight:storage(), 1, load_module_te.weight:size())
		load_module_te.bias = torch.CudaTensor(load_module_tr.bias:storage(), 1, load_module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end 

-- net2_tr
load_i_tr = 1
while load_i_tr <= net2_tr:size()-1 do
	local load_module_tr = net2_tr:get(load_i_tr)
	local load_module_te = load_net_te_all[load_i_te]
	
	if load_module_tr.weight then
		assert(load_module_te.weight:nElement() == load_module_tr.weight:nElement())
		assert(load_module_te.bias:nElement() == load_module_tr.bias:nElement())

		load_module_tr.weight = torch.CudaTensor(load_module_te.weight:storage(), 1, load_module_te.weight:size())
		load_module_te.bias = torch.CudaTensor(load_module_tr.bias:storage(), 1, load_module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end 

-- net3_tr
load_i_tr = 1
while load_i_tr <= net3_tr:size()-1 do
	local load_module_tr = net3_tr:get(load_i_tr)
	local load_module_te = load_net_te_all[load_i_te]
	
	if load_module_tr.weight then
		assert(load_module_te.weight:nElement() == load_module_tr.weight:nElement())
		assert(load_module_te.bias:nElement() == load_module_tr.bias:nElement())

		load_module_tr.weight = torch.CudaTensor(load_module_te.weight:storage(), 1, load_module_te.weight:size())
		load_module_te.bias = torch.CudaTensor(load_module_tr.bias:storage(), 1, load_module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end 

-- net4_tr
load_i_tr = 1
while load_i_tr <= net4_tr:size()-1 do
	local load_module_tr = net4_tr:get(load_i_tr)
	local load_module_te = load_net_te_all[load_i_te]
	
	if load_module_tr.weight then
		assert(load_module_te.weight:nElement() == load_module_tr.weight:nElement())
		assert(load_module_te.bias:nElement() == load_module_tr.bias:nElement())

		load_module_tr.weight = torch.CudaTensor(load_module_te.weight:storage(), 1, load_module_te.weight:size())
		load_module_te.bias = torch.CudaTensor(load_module_tr.bias:storage(), 1, load_module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end 

-- net_c_tr
load_i_tr = 1
while load_i_tr <= net_c_tr:size()-1 do
	local load_module_tr = net_c_tr:get(load_i_tr)
	local load_module_te = load_net_te_all[load_i_te]
	
	if load_module_tr.weight then
		assert(load_module_te.weight:nElement() == load_module_tr.weight:nElement())
		assert(load_module_te.bias:nElement() == load_module_tr.bias:nElement())

		load_module_tr.weight = torch.CudaTensor(load_module_te.weight:storage(), 1, load_module_te.weight:size())
		load_module_te.bias = torch.CudaTensor(load_module_tr.bias:storage(), 1, load_module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end 

-- for remaining
load_i_tr = 4
while load_i_tr <= 10 do
	local module_tr = main_training:get(load_i_tr)
	local module_te = load_net_te_all[load_i_te]

	if module_tr.weight then
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias   = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	load_i_tr = load_i_tr + 1
	load_i_te = load_i_te + 1
end

-----------[[]]





-- network for testing (make sure it's synched with net_tr)
local pad = 0 --(opt.ks - 1) / 2

net1_te = nn.Sequential()
for i = 1,#fm do
	net1_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
	net1_te:add(cudnn.ReLU(true))
	if i ==1 then
	    net1_te:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net1_te:cuda()

net2_te = nn.Sequential()
for i = 1,#fm do
	net2_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
	net2_te:add(cudnn.ReLU(true))
	if i ==1 then
	    net2_te:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net2_te:cuda()

net3_te = nn.Sequential()
for i = 1,#fm do
	net3_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
	net3_te:add(cudnn.ReLU(true))
	if i ==1 then
	    net3_te:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net3_te:cuda()

net4_te = nn.Sequential()
for i = 1,#fm do
	net4_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
	net4_te:add(cudnn.ReLU(true))
	if i ==1 then
	    net4_te:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net4_te:cuda()

net_c_te = nn.Sequential()
for i = 1,#fm do
	net_c_te:add(cudnn.SpatialConvolution(i == 1 and n_input_plane or fm[i - 1], fm[i], opt.ks, opt.ks, 1, 1, pad, pad))
	net_c_te:add(cudnn.ReLU(true))
	if i ==1 then
	    net_c_te:add(nn.SpatialMaxPooling(2, 2)) 
	end
end
net_c_te:cuda()


net_te_remain = nn.Sequential()
for i = 1,opt.l2 do
	net_te_remain:add(nn.SpatialConvolution1_fw(i == 1 and 10*fm[#fm] or opt.nh2, opt.nh2))
	net_te_remain:add(cudnn.ReLU(true))
end
net_te_remain:add(nn.SpatialConvolution1_fw(opt.nh2, 1))
net_te_remain:add(cudnn.Sigmoid(true))
net_te_remain:cuda()




-- tie weights
net_te_all = {}
for i, v in ipairs(net1_te.modules) do table.insert(net_te_all, v) end
for i, v in ipairs(net2_te.modules) do table.insert(net_te_all, v) end
for i, v in ipairs(net3_te.modules) do table.insert(net_te_all, v) end
for i, v in ipairs(net4_te.modules) do table.insert(net_te_all, v) end
for i, v in ipairs(net_c_te.modules) do table.insert(net_te_all, v) end
for i, v in ipairs(net_te_remain.modules) do table.insert(net_te_all, v) end

local finput = torch.CudaTensor()
local i_tr = 1
local i_te = 1


-- for net1_tr
while i_tr <= net1_tr:size()-1 do
	local module_tr = net1_tr:get(i_tr)
	local module_te = net_te_all[i_te]
	
	print('net1_tr: ', module_tr)
	print('net1_tr: ', module_te)

	if module_tr.weight then
	    print('--------------------- net1_tr')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias   = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end 

-- for net2_tr
i_tr = 1
while i_tr <= net2_tr:size()-1 do
	local module_tr = net2_tr:get(i_tr)
	local module_te = net_te_all[i_te]

	print('net2_tr: ', module_tr)
	print('net2_tr: ', module_te)

	if module_tr.weight then
	    print('--------------------- net2_tr')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end 

-- for net3_tr
i_tr = 1
while i_tr <= net3_tr:size()-1 do
	local module_tr = net3_tr:get(i_tr)
	local module_te = net_te_all[i_te]

	print('net3_tr: ', module_tr)
	print('net3_tr: ', module_te)

	if module_tr.weight then
	    print('--------------------- net3_tr')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end 

-- for net4_tr
i_tr = 1
while i_tr <= net4_tr:size()-1 do
	local module_tr = net4_tr:get(i_tr)
	local module_te = net_te_all[i_te]

	print('net4_tr: ', module_tr)
	print('net4_tr: ', module_te)

	if module_tr.weight then
	    print('--------------------- net4_tr')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end 

-- for net_c_tr
i_tr = 1
while i_tr <= net_c_tr:size()-1 do
	local module_tr = net_c_tr:get(i_tr)
	local module_te = net_te_all[i_te]

	print('net_c_tr: ', module_tr)
	print('net_c_tr: ', module_te)

	if module_tr.weight then
	    print('--------------------- net_c_tr')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end

-- for remaining
i_tr = 4
while i_tr <= 10 do
	local module_tr = main_training:get(i_tr)
	local module_te = net_te_all[i_te]

	print('remaining: ', module_tr)
	print('remaining: ', module_te)

	if module_tr.weight then
	    print('--------------------- remaining')
		assert(module_te.weight:nElement() == module_tr.weight:nElement())
		assert(module_te.bias:nElement() == module_tr.bias:nElement())

		module_te.weight = torch.CudaTensor(module_tr.weight:storage(), 1, module_te.weight:size())
		module_te.bias   = torch.CudaTensor(module_tr.bias:storage(), 1, module_te.bias:size())
	end

	i_tr = i_tr + 1
	i_te = i_te + 1
end







-- link weights

local params 	= {}
local grads 	= {}
local momentums = {}

-- for net1_tr
for i = 1,net1_tr:size() do
	local m = net1_tr:get(i)
	if m.weight then
	    print('--------------------- net1_tr')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end

-- for net2_tr
for i = 1,net2_tr:size() do
	local m = net2_tr:get(i)
	if m.weight then
	    print('--------------------- net2_tr')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end

-- for net3_tr
for i = 1,net3_tr:size() do
	local m = net3_tr:get(i)
	if m.weight then
	    print('--------------------- net3_tr')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end

-- for net4_tr
for i = 1,net4_tr:size() do
	local m = net4_tr:get(i)
	if m.weight then
	    print('--------------------- net4_tr')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end

-- for net_c_tr
for i = 1,net_c_tr:size() do
	local m = net_c_tr:get(i)
	if m.weight then
	    print('--------------------- net_c_tr')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end

-- for remaining
for i = 4,10 do
	local m = main_training:get(i)
	if m.weight then
	    print('--------------------- remaining')
		m.weight_v = torch.CudaTensor(m.weight:size()):zero()
		table.insert(params, m.weight)
		table.insert(grads, m.gradWeight)
		table.insert(momentums, m.weight_v)
	end

	if m.bias then
		m.bias_v = torch.CudaTensor(m.bias:size()):zero()
		table.insert(params, m.bias)
		table.insert(grads, m.gradBias)
		table.insert(momentums, m.bias_v)
	end
end 




ws         = 32 --get_window_size(net_tr)
patch_size = 16

print('window size: ', ws)

x_batch_tr        = torch.CudaTensor(opt.bs*2, n_input_plane, ws, ws)
y_batch_tr        = torch.CudaTensor(opt.bs)
x_net1_batch_tr   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net2_batch_tr   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net3_batch_tr   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net4_batch_tr   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net_c_batch_tr  = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)

x_batch_tr_        = torch.FloatTensor(x_batch_tr:size())
y_batch_tr_        = torch.FloatTensor(y_batch_tr:size())
x_net1_batch_tr_   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net2_batch_tr_   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net3_batch_tr_   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net4_batch_tr_   = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)
x_net_c_batch_tr_  = torch.CudaTensor(opt.bs*2, n_input_plane, patch_size, patch_size)





----------------------------------
---------- 3. training -----------
----------------------------------

print('Starting training ..... ')
local time = sys.clock()

for epoch = 15,21 do 

	if epoch == 19 then
		opt.lr = opt.lr / 10
	end

	local kt_err_tr 	= 0
	local kt_err_tr_cnt = 0

	local mb_err_tr     = 0
	local mb_err_tr_cnt = 0
	

	for mb_t = 1, mb_nnz:size(1) - opt.bs/2, opt.bs/2 do

		------------------------
		-- for middlebury ------
		for mb_i = 1,opt.bs/2 do

			mb_d_pos = torch.uniform(-opt.true1, opt.true1)
			mb_d_neg = torch.uniform(opt.false1, opt.false2)

			if torch.uniform() < 0.5 then
				mb_d_neg = -mb_d_neg
			end

		    assert(opt.hscale <= 1 and opt.scale <= 1)
		    local mb_s 		= torch.uniform(opt.scale, 1)
		    local mb_scale 	= {mb_s * torch.uniform(opt.hscale, 1), mb_s}

		    if opt.hflip == 1 and torch.uniform() < 0.5 then
		       	mb_scale[1] = -mb_scale[1]
		    end

		    if opt.vflip == 1 and torch.uniform() < 0.5 then
		       	mb_scale[2] = -mb_scale[2]
		    end

		    local mb_hshear = torch.uniform(-opt.hshear, opt.hshear)
		    local mb_trans  = {torch.uniform(-opt.trans, opt.trans), torch.uniform(-opt.trans, opt.trans)}
		    local mb_phi    = torch.uniform(-opt.rotate * math.pi / 180, opt.rotate * math.pi / 180)
		    local mb_brightness = torch.uniform(-opt.brightness, opt.brightness)

		    assert(opt.contrast >= 1 and opt.d_contrast >= 1)
		    local mb_contrast = torch.uniform(1 / opt.contrast, opt.contrast)

		    local mb_scale_ 	= {mb_scale[1] * torch.uniform(opt.d_hscale, 1), mb_scale[2]}
		    local mb_hshear_ 	= mb_hshear + torch.uniform(-opt.d_hshear, opt.d_hshear)
		    local mb_trans_ 	= {mb_trans[1], mb_trans[2] + torch.uniform(-opt.d_vtrans, opt.d_vtrans)}
		    local mb_phi_ 		= mb_phi + torch.uniform(-opt.d_rotate * math.pi / 180, opt.d_rotate * math.pi / 180)
		    local mb_brightness_ 	= mb_brightness + torch.uniform(-opt.d_brightness, opt.d_brightness)
		    local mb_contrast_ 		= mb_contrast * torch.uniform(1 / opt.d_contrast, opt.d_contrast)

		    local mb_ind = mb_perm[mb_t + mb_i - 1]
		    mb_img  = mb_nnz[{mb_ind, 1}]
		    mb_dim3 = mb_nnz[{mb_ind, 2}]
		    mb_dim4 = mb_nnz[{mb_ind, 3}]
		    mb_d    = mb_nnz[{mb_ind, 4}]

	    	
	       	mb_light 	= (torch.random() % (#mb_X[mb_img] - 1)) + 2
	       	mb_exp 		= (torch.random() % mb_X[mb_img][mb_light]:size(1)) + 1
	       	mb_light_ 	= mb_light
	       	mb_exp_ 	= mb_exp

		   	if torch.uniform() < opt.d_exp then
			  	mb_exp_ = (torch.random() % mb_X[mb_img][mb_light]:size(1)) + 1
		   	end
		   	if torch.uniform() < opt.d_light then
			  	mb_light_ = math.max(2, mb_light - 1)
		   	end

		   	mb_x0 = mb_X[mb_img][mb_light][{mb_exp,1}]
		   	mb_x1 = mb_X[mb_img][mb_light_][{mb_exp_,2}]
			
		    make_patch(mb_x0, x_batch_tr_[mb_i * 4 - 3], mb_dim3, mb_dim4, 						mb_scale, 	mb_phi, 	mb_trans, 	mb_hshear, 	mb_brightness, 	mb_contrast)
		    make_patch(mb_x1, x_batch_tr_[mb_i * 4 - 2], mb_dim3, mb_dim4 - mb_d + mb_d_pos, 	mb_scale_, 	mb_phi_, 	mb_trans_, 	mb_hshear_, mb_brightness_, mb_contrast_)
		    make_patch(mb_x0, x_batch_tr_[mb_i * 4 - 1], mb_dim3, mb_dim4, 						mb_scale, 	mb_phi, 	mb_trans, 	mb_hshear, 	mb_brightness, 	mb_contrast)
		    make_patch(mb_x1, x_batch_tr_[mb_i * 4 - 0], mb_dim3, mb_dim4 - mb_d + mb_d_neg, 	mb_scale_, 	mb_phi_, 	mb_trans_, 	mb_hshear_, mb_brightness_, mb_contrast_)

		    y_batch_tr_[mb_i * 2 - 1] 	= 0
		    y_batch_tr_[mb_i * 2] 		= 1
		end




		-----------------------------------------------------------
		--================== have data and train ==================
		-- prepare patch data
        for data_ii = 1,x_batch_tr_:size(1) do
            x_net1_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 1,  patch_size):narrow(3, 1,  patch_size):clone()
            x_net2_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 1,  patch_size):narrow(3, 17, patch_size):clone()
            x_net3_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 1,  patch_size):clone()
            x_net4_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 17, patch_size):clone()
            x_net_c_batch_tr_[data_ii] = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 17, patch_size):clone()
        end
        

     	--x_batch_tr:copy(x_batch_tr_)
     	y_batch_tr:copy(y_batch_tr_)
     	x_net1_batch_tr:copy(x_net1_batch_tr_)
     	x_net2_batch_tr:copy(x_net2_batch_tr_)
     	x_net3_batch_tr:copy(x_net3_batch_tr_)
     	x_net4_batch_tr:copy(x_net4_batch_tr_)
     	x_net_c_batch_tr:copy(x_net_c_batch_tr_)
     	
     	
     	
     	
     	
     	
		for mb_i = 1,#params do
			grads[mb_i]:zero()
		end

		main_training:forward({x_net1_batch_tr, x_net2_batch_tr, x_net3_batch_tr, x_net4_batch_tr, x_net_c_batch_tr})
		local mb_err = criterion:forward(main_training.output, y_batch_tr)
		if mb_err >= 0 and mb_err < 100 then
			mb_err_tr 		= mb_err_tr + mb_err
			mb_err_tr_cnt 	= mb_err_tr_cnt + 1
		else
			print(('WARNING! err=%f'):format(mb_err))
		end

		criterion:backward(main_training.output, y_batch_tr)
     	main_training:backward({x_net1_batch_tr, x_net2_batch_tr, x_net3_batch_tr, x_net4_batch_tr, x_net_c_batch_tr}, criterion.gradInput)

		for mb_i = 1,#params do
			momentums[mb_i]:mul(opt.mom):add(-opt.lr, grads[mb_i])
			params[mb_i]:add(momentums[mb_i])
		end
		-- end middlebury




		------------------------
		-- for kitti2015 -------
		if mb_t < (kt_nnz:size(1) - opt.bs/2) then
		
			for kt_i = 1,opt.bs/2 do

				kt_d_pos = torch.uniform(-opt.true1, opt.true1)
				kt_d_neg = torch.uniform(opt.false1, opt.false2)

				if torch.uniform() < 0.5 then
					kt_d_neg = -kt_d_neg
				end

				assert(opt.hscale <= 1 and opt.scale <= 1)
				local kt_s     = torch.uniform(opt.scale, 1)
				local kt_scale = {kt_s * torch.uniform(opt.hscale, 1), kt_s}

				if opt.hflip == 1 and torch.uniform() < 0.5 then
				   	kt_scale[1] = -kt_scale[1]
				end

				if opt.vflip == 1 and torch.uniform() < 0.5 then
				   	kt_scale[2] = -kt_scale[2]
				end

				local kt_hshear = torch.uniform(-opt.hshear, opt.hshear)
				local kt_trans  = {torch.uniform(-opt.trans, opt.trans), torch.uniform(-opt.trans, opt.trans)}
				local kt_phi    = torch.uniform(-opt.rotate * math.pi / 180, opt.rotate * math.pi / 180)
				local kt_brightness = torch.uniform(-opt.brightness, opt.brightness)

				assert(opt.contrast >= 1 and opt.d_contrast >= 1)
				local kt_contrast = torch.uniform(1 / opt.contrast, opt.contrast)

				local kt_scale_ 	= {kt_scale[1] * torch.uniform(opt.d_hscale, 1), kt_scale[2]}
				local kt_hshear_ 	= kt_hshear + torch.uniform(-opt.d_hshear, opt.d_hshear)
				local kt_trans_ 	= {kt_trans[1], kt_trans[2] + torch.uniform(-opt.d_vtrans, opt.d_vtrans)}
				local kt_phi_ 		= kt_phi + torch.uniform(-opt.d_rotate * math.pi / 180, opt.d_rotate * math.pi / 180)
				local kt_brightness_ 	= kt_brightness + torch.uniform(-opt.d_brightness, opt.d_brightness)
				local kt_contrast_ 		= kt_contrast * torch.uniform(1 / opt.d_contrast, opt.d_contrast)

				local kt_ind = kt_perm[mb_t + kt_i - 1]
				kt_img  = kt_nnz[{kt_ind, 1}]
				kt_dim3 = kt_nnz[{kt_ind, 2}]
				kt_dim4 = kt_nnz[{kt_ind, 3}]
				kt_d    = kt_nnz[{kt_ind, 4}]
				
			   	kt_x0 = kt_X0[kt_img]
			   	kt_x1 = kt_X1[kt_img]		    	

				make_patch(kt_x0, x_batch_tr_[kt_i * 4 - 3], kt_dim3, kt_dim4, 						kt_scale, 	kt_phi, 	kt_trans, 	kt_hshear, 	kt_brightness, 	kt_contrast)
				make_patch(kt_x1, x_batch_tr_[kt_i * 4 - 2], kt_dim3, kt_dim4 - kt_d + kt_d_pos, 	kt_scale_, 	kt_phi_, 	kt_trans_, 	kt_hshear_, kt_brightness_, kt_contrast_)
				make_patch(kt_x0, x_batch_tr_[kt_i * 4 - 1], kt_dim3, kt_dim4, 						kt_scale, 	kt_phi, 	kt_trans, 	kt_hshear, 	kt_brightness, 	kt_contrast)
				make_patch(kt_x1, x_batch_tr_[kt_i * 4 - 0], kt_dim3, kt_dim4 - kt_d + kt_d_neg, 	kt_scale_, 	kt_phi_, 	kt_trans_, 	kt_hshear_, kt_brightness_, kt_contrast_)

				y_batch_tr_[kt_i * 2 - 1] = 0
				y_batch_tr_[kt_i * 2] 	  = 1
			end




			-----------------------------------------------------------
			--================== have data and train ==================
			-- prepare patch data
            for data_ii = 1,x_batch_tr_:size(1) do
                x_net1_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 1,  patch_size):narrow(3, 1,  patch_size):clone()
                x_net2_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 1,  patch_size):narrow(3, 17, patch_size):clone()
                x_net3_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 1,  patch_size):clone()
                x_net4_batch_tr_[data_ii]  = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 17, patch_size):clone()
                x_net_c_batch_tr_[data_ii] = x_batch_tr_[data_ii]:narrow(2, 17, patch_size):narrow(3, 17, patch_size):clone()
            end
            

         	--x_batch_tr:copy(x_batch_tr_)
         	y_batch_tr:copy(y_batch_tr_)
         	x_net1_batch_tr:copy(x_net1_batch_tr_)
         	x_net2_batch_tr:copy(x_net2_batch_tr_)
         	x_net3_batch_tr:copy(x_net3_batch_tr_)
         	x_net4_batch_tr:copy(x_net4_batch_tr_)
         	x_net_c_batch_tr:copy(x_net_c_batch_tr_)
     	
     	
     	

			for kt_i = 1,#params do
				grads[kt_i]:zero()
			end

			main_training:forward({x_net1_batch_tr, x_net2_batch_tr, x_net3_batch_tr, x_net4_batch_tr, x_net_c_batch_tr})
			local kt_err = criterion:forward(main_training.output, y_batch_tr)
			if kt_err >= 0 and kt_err < 100 then
				kt_err_tr 	  = kt_err_tr + kt_err
				kt_err_tr_cnt = kt_err_tr_cnt + 1
			else
				print(('WARNING! err=%f'):format(kt_err))
			end

			criterion:backward(main_training.output, y_batch_tr)
     	    main_training:backward({x_net1_batch_tr, x_net2_batch_tr, x_net3_batch_tr, x_net4_batch_tr, x_net_c_batch_tr}, criterion.gradInput)

			for kt_i = 1, #params do
				momentums[kt_i]:mul(opt.mom):add(-opt.lr, grads[kt_i])
				params[kt_i]:add(momentums[kt_i])
			end
		end

	end

	print(epoch, mb_err_tr / mb_err_tr_cnt, opt.lr, sys.clock() - time)
	print(epoch, kt_err_tr / kt_err_tr_cnt, opt.lr, sys.clock() - time)
	collectgarbage()
	
	save_net(epoch)
end

opt.net_fname = save_net(0)   
collectgarbage()






