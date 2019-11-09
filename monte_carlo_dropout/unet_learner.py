from typing import List, Callable, Optional, Tuple, Union, Any

import numpy as np

from fastai.torch_core import *
from fastai.basic_train import *
from fastai.basic_data import *
from fastai.layers import *
from fastai.callbacks.hooks import *

from monte_carlo_dropout.mc_dropout import MCDropout
from monte_carlo_dropout.model_meta import model_meta, _default_meta


def _get_sfs_idxs(sizes: Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class UnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c: int, x_in_c: int, hook: Hook, final_div: bool = True, blur: bool = False,
                 leaky: float = None, self_attention: bool = False, dropout_rate: float = .0,
                 **kwargs):
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.dropout = MCDropout(dropout_rate)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        out = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        out = self.conv2(self.conv1(out))
        out = self.dropout(out)
        return out


class DynamicUnet(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(self, encoder: nn.Module, n_classes: int, block_drate: float, final_drate: float,
                 img_size: Tuple[int,int] = (256, 256), blur: bool = False, blur_final = True,
                 self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True,
                 bottle: bool = False, **kwargs):

        imsize = img_size
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_idxs)-3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur, self_attention=sa,
                                   dropout_rate=block_drate,
                                   **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        x = PixelShuffle_ICNR(ni)(x)
        if imsize != x.shape[-2:]: layers.append(Lambda(lambda x: F.interpolate(x, imsize, mode='nearest')))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        layers += [MCDropout(final_drate)]
        layers += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

    @classmethod
    def _set_force_dropout(cls, module: nn.Module, mode):
        for submodule in module.children():
            if isinstance(submodule, MCDropout):
                submodule.force_dropout = mode
            cls._set_force_dropout(submodule, mode)

    def force_dropout(self, mode: bool = True):
        self._set_force_dropout(self, mode)


def get_mc_dropout_preds(learner, n_iter=10, dtype=torch.float32, **pred_args):

    preds = learner.get_preds()[0].to(dtype)
    for i in range(1, n_iter):
        print(i)
        preds += learner.get_preds()[0].to(dtype)
    return preds / n_iter




def has_pool_type(m):
    if is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False


def create_body(arch:Callable, pretrained:bool=True, cut:Optional[Union[int, Callable]]=None):
    """
    Cut off the body of a typically pretrained `model` at `cut` (int) or cut the model as specified by `cut(model)`
    (function).
    """
    model = arch(pretrained)
    cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif isinstance(cut, Callable):
        return cut(model)
    else:
        raise TypeError("cut must be either integer or a function")


def cnn_config(arch):
    "Get the metadata associated with `arch`."
    torch.backends.cudnn.benchmark = True
    return model_meta.get(arch, _default_meta)


def unet_learner(data: DataBunch, arch: Callable,  block_drate: float, final_drate: float,
                 pretrained: bool = True, blur_final: bool = True,
                 norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None,
                 blur: bool = False, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False, cut: Union[int, Callable]=None,
                 **learn_kwargs: Any) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    try:    size = data.train_ds[0][0].size
    except: size = next(iter(data.train_dl))[0].shape[-2:]
    model = to_device(DynamicUnet(body, n_classes=data.c, img_size=size, blur=blur, blur_final=blur_final,
                                  self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
                                  bottle=bottle, block_drate=block_drate, final_drate=final_drate), data.device)
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn