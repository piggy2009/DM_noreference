from PIL import Image
import numpy as np
import torchvision
import torch
from model.ddpm_trans_modules.unet import UNet
# from model.ddpm_trans_modules import unet_backup
from model.clip_loss2 import generate_src_target_txt, FrozenCLIPEmbedder
from model.utils import load_part_of_model2
import nethook
import os
from matplotlib import pyplot as plt
import core.metrics as Metrics
from tqdm import tqdm
import cv2
from label_clip import compute_semantic_dis

src_model_path = '...' # pre-trained model
image_root = 'dataset/water_val_16_128' # image root
image_name = '00014.png'
device = 1
label = 1

betas = np.linspace(1e-6, 1e-2, 1000, dtype=np.float64)
# betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod))
eta = 0
denoise_fn = None
clip_embedding = FrozenCLIPEmbedder(device=device).to(device)
# units = [8] # conv1
set_units = [0]  # range(0, 1)

model = UNet(inner_channel=48, norm_groups=24, in_channel=6).to(device=device)
# model = unet_backup.DiT(depth=12, in_channels=6, hidden_size=384, patch_size=4, num_heads=6, input_size=128).to(device=device)
model = load_part_of_model2(model, src_model_path)
print(model)
totensor = torchvision.transforms.ToTensor()

if not isinstance(model, nethook.InstrumentedModel):
    model = nethook.InstrumentedModel(model)

def zero_out_tree_units(data, model):
    data[:, set_units, :, :] = 0.0
    return data

###### modify the values of feature maps #####
def turn_off_tree_units(data, model):
    data[:, set_units, :, :] = -5.0
    return data

def p_sample_ddim2(x, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=None, context=None):
    b, *_, device = *x.shape, x.device
    bt = extract(betas, t, x.shape)
    at = extract((1.0 - betas).cumprod(), t, x.shape)
    # print('at=', at)
    if condition_x is not None:
        et = denoise_fn(torch.cat([condition_x, x], dim=1), t,
                             style, context, None)
    else:
        et = denoise_fn(x, t)

    x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
    # x0_air_t = (x_air - et_air * (1 - at).sqrt()) / at.sqrt()
    if t_next == None:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - betas).cumprod(), t_next, x.shape)
    if eta == 0:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # xt_air_next = at_next.sqrt() * x0_air_t + (1 - at_next).sqrt() * et_air
    elif at > (at_next):
        print('Inversion process is only possible with eta = 0')
        raise ValueError
    else:
        c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
        # xt_air_next = at_next.sqrt() * x0_air_t + c2 * et_air + c1 * torch.randn_like(x0_t)

    # noise = noise_like(x.shape, device, repeat_noise)
    # no noise when t == 0

    return xt_next

def p_sample_loop2(sr, style, label, continous=False):
    g_gpu = torch.Generator(device=device).manual_seed(44444)
    sample_inter = 10
    x = sr
    # condition_x = torch.mean(x, dim=1, keepdim=True)
    condition_x = x
    shape = x.shape
    b = shape[0]
    img = torch.randn(shape, device=device, generator=g_gpu)
    start = 1000
    src_txt_inputs, target_txt_inputs = generate_src_target_txt(torch.unsqueeze(torch.tensor(label), 0))
    context = clip_embedding(target_txt_inputs)
    # t = torch.full((b,), start, device=device, dtype=torch.long)

    # noise = torch.randn(shape, device=device)
    # img = self.q_sample(x_start=x, t=t, noise=noise)
    # img_air = q_sample(x_start=x_air, t=t, noise=noise)

    ret_img = img
    # num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
    c = 100
    num_timesteps_ddim = np.asarray(list(range(0, start, c)))
    time_steps = np.flip(num_timesteps_ddim[:-1])
    for j, i in enumerate(time_steps):
        # print('i = ', i)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        if j == len(time_steps) - 1:
            t_next = None
        else:
            t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
        img = p_sample_ddim2(img, t, t_next, condition_x=condition_x, style=style, context=context)
        # print('i=', i)
        # acts = model.retained_layer(check_layer)
        # # print(acts.shape)
        # print(acts[0, 0, 0])
        if i % sample_inter == 0:
            ret_img = torch.cat([ret_img, img], dim=0)

    if continous:
        return ret_img
    else:
        return ret_img[-1]

def conver2image(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def q_sample(x_start, t, noise=None):
    # noise = default(noise, lambda: torch.randn_like(x_start))
    if noise is None:
        noise = torch.randn_like(x_start)
    # fix gama
    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(sqrt_one_minus_alphas_cumprod,
                t, x_start.shape) * noise
    )

def q_sample_recover(x_noisy, t, predict_noise=None):
    # noise = default(noise, lambda: torch.randn_like(x_start))
    return (x_noisy - extract(sqrt_one_minus_alphas_cumprod,
                t, x_noisy.shape) * predict_noise) / extract(sqrt_alphas_cumprod, t, x_noisy.shape)

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

def generate_basic_value(sr, style, label):
    denoise_fn = model
    r = p_sample_loop2(sr, style, label, continous=False)
    img = conver2image(r)
    # cv2.imwrite('./a.jpg', cv2.cvtColor(conver2image(r), cv2.COLOR_RGB2BGR))
    dis = compute_semantic_dis(Image.fromarray(img), 'red style')

def voted_units(input, unit_num):
    # arr = np.array(input)
    vote_units = []
    for i in range(0, unit_num):
        unit_count = 0
        for j in range(0, len(input)):
            arr = np.array(input[j])
            # print('test:', arr)
            if len(arr) == 0:
                continue
            if i in arr[:, 0]:
                unit_count = unit_count + 1
        if unit_count > len(input) / 3:
            # print('unit', str(i), 'is useful')
            vote_units.append(i)
            # for j in range(0, len(input)):
            #     arr = np.array(input[j])
            #     for z in range(0, arr.shape[0]):
            #         if i == arr[z, 0]:
            #             select_units.append([arr[z, 0], arr[z, 1]])
    return vote_units

def generate_voted_dict(vote_units, input):
    vote_units_value = {}
    for unit in vote_units:
        count = 0
        value = 0
        for i in range(0, len(input)):
            one = input[i]
            for j in one:
                if unit == j[0]:
                    # print(j)
                    count += 1
                    value += j[1]
        vote_units_value[unit] = value / count

    return sorted(vote_units_value.items(), key=lambda x:x[1])




if __name__ == '__main__':
    # label_file = open()
    units_list_pos = []
    units_list_neg = []
    unit_num = 48 * 2 ** 1
    check_layer = 'encoder_water.block2_control'
    semantic_thres = 0.008
    close_to = 'red style'
    for i, line in enumerate(open(image_root + '/label.txt')):
        if i == 30:
            break
        full_image_name, label_ = line.split(' ')
        # print(full_image_name.split('/')[-1])

        sr = Image.open(os.path.join(image_root, 'sr_16_128', full_image_name.split('/')[-1])).convert("RGB")
        style = Image.open(os.path.join(image_root, 'style_128', full_image_name.split('/')[-1])).convert("RGB")
        hr = Image.open(os.path.join(image_root, 'hr_128', full_image_name.split('/')[-1])).convert("RGB")
        sr = totensor(sr) * 2 - 1
        hr = totensor(hr) * 2 - 1
        sr = torch.unsqueeze(sr, 0)
        hr = torch.unsqueeze(hr, 0)

        style = totensor(style) * 2 - 1
        style = torch.unsqueeze(style, 0)

        # t = torch.full((1,), 100, dtype=torch.long)
        # sr = q_sample(sr, t)
        # hr = q_sample(hr, t)

        sr = sr.to(device=device)
        hr = hr.to(device=device)
        style = style.to(device=device)

        chose_units_set_pos = []
        chose_units_set_neg = []
        denoise_fn = model
        r = p_sample_loop2(sr, style, int(label_.strip()), continous=False)
        img = conver2image(r)
        base_semantic_value = compute_semantic_dis(Image.fromarray(img), close_to)
        print('base value:', base_semantic_value)
        ####### check every unit with loop ###########
        for i in range(0, unit_num):
            print('testing unit num:', i)
            set_units = [i]
            model.edit_layer(check_layer, rule=turn_off_tree_units)
            # model.retain_layer(check_layer)
            # model.retain_layer('blocks2.0')
            # img = model(sr, hr, t)
            # denoise_fn = model
            r = p_sample_loop2(sr, style, int(label_.strip()), continous=False)
            img = conver2image(r)
            # cv2.imwrite('./a.jpg', cv2.cvtColor(conver2image(r), cv2.COLOR_RGB2BGR))
            dis = compute_semantic_dis(Image.fromarray(img), 'red style')
            print('dis=', dis - base_semantic_value, 'curr=', dis, 'base=', base_semantic_value)
            if (dis - base_semantic_value) > semantic_thres:
                chose_units_set_pos.append([i, float(dis - base_semantic_value)])
            # elif (dis - base_semantic_value) < -semantic_thres:
            #     chose_units_set_neg.append(i)
        # print(chose_units_set_pos)
        # chose_units_set_pos.sort(key=lambda t:t[1])
        units_list_pos.append(chose_units_set_pos)
        # units_list_neg.append(chose_units_set_neg)
    print('positive units:', units_list_pos)
    # print('negative units:', units_list_neg)
    units = voted_units(units_list_pos, unit_num)
    r = generate_voted_dict(units, units_list_pos)
    print(units)
    print(r)
    rr = []
    for temp in r:
        rr.append(temp[0])
    print(rr)
    # block2 units = [10, 14, 33, 45, 48, 49, 56, 58, 62, 63, 64, 73, 74, 78, 79, 83, 84, 85, 90, 91, 93, 94, 95]
    # block1 units = [0, 6, 11, 15, 17, 24, 25, 44, 47]
    # [(11, 0.01865234375), (15, 0.019073486328125), (17, 0.0548583984375)]


'''
if __name__ == '__main__':
    sr = Image.open(os.path.join(image_root, 'sr_16_128', image_name)).convert("RGB")
    style = Image.open(os.path.join(image_root, 'style_128', image_name)).convert("RGB")
    hr = Image.open(os.path.join(image_root, 'hr_128', image_name)).convert("RGB")
    sr = totensor(sr) * 2 - 1
    hr = totensor(hr) * 2 - 1
    sr = torch.unsqueeze(sr, 0)
    hr = torch.unsqueeze(hr, 0)

    style = totensor(style) * 2 - 1
    style = torch.unsqueeze(style, 0)

    sr = sr.to(device=device)
    hr = hr.to(device=device)
    style = style.to(device=device)

    ####### visualize ###########
    test = [0, 6, 15, 11, 17, 1, 3]
    check_layer = 'encoder_water.block1_control'
    for unit in test:
        set_units.append(unit)
        model.edit_layer(check_layer, rule=turn_off_tree_units)
        denoise_fn = model
        r = p_sample_loop2(sr, style, label, continous=False)
        img = conver2image(r)
        cv2.imwrite(os.path.join('units3', image_name + '_' + str(unit) + '.jpg'), cv2.cvtColor(conver2image(r), cv2.COLOR_RGB2BGR))
        set_units.clear()

    # model.edit_layer(check_layer, rule=turn_off_tree_units)
    denoise_fn = model
    r = p_sample_loop2(sr, style, label, continous=False)
    img = conver2image(r)
    cv2.imwrite(os.path.join('units3', image_name + '_normal' + '.jpg'),
                cv2.cvtColor(conver2image(r), cv2.COLOR_RGB2BGR))
    # compute_semantic_dis(Image.fromarray(img), 'red style')
    # plt.subplot(3, 1, 1)
    # plt.imshow(conver2image(sr))


    # plt.subplot(2, 1, 1)
    # plt.imshow(conver2image(hr))
    #
    # plt.subplot(2, 1, 2)
    # plt.imshow(conver2image(r))
    # plt.show()
'''