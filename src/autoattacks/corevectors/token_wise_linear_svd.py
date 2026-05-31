from pathlib import Path

import torch

from peepholelib.coreVectors.dimReduction.dim_reduction_base import DimReductionBase as DRB


class TokenWiseLinearSVD(DRB):
    def __init__(self, **kwargs):
        DRB.__init__(self, **kwargs)
        path = Path(kwargs['path'])
        layer = kwargs['layer']
        model = kwargs['model']
        q = kwargs.get('rank', 300)
        self.cv_dim = kwargs.get('cv_dim', None)
        verbose = kwargs.get('verbose', False)

        path.mkdir(parents=True, exist_ok=True)
        file_path = path / layer

        _layer = model._target_modules[layer]
        device = model.device

        if file_path.exists():
            if verbose:
                print(f'File {file_path} exists. Loading from disk.')
            self._svd = torch.load(file_path)
        else:
            W = _layer.weight
            use_bias = _layer.bias is not None
            if use_bias:
                W = torch.hstack((W, _layer.bias.reshape(-1, 1)))
            W = W.to(device)
            U, s, Vh = torch.svd_lowrank(W, q)
            U, s, Vh = U.detach().cpu(), s.detach().cpu(), Vh.detach().cpu()
            self._svd = {
                'U': U,
                's': s,
                'Vh': Vh.T,
                'use_bias': use_bias,
            }

            if verbose:
                print(f'saving {file_path}')
            torch.save(self._svd, file_path)

        self.reduct_m = self._svd['Vh'].detach().to(device)
        in_features = _layer.weight.shape[1]
        in_dim = self.reduct_m.shape[1]
        if in_dim == in_features + 1:
            self.use_bias = True
        elif in_dim == in_features:
            self.use_bias = False
        else:
            raise RuntimeError(
                f'Loaded SVD input dimension ({in_dim}) does not match layer '
                f'input dimension ({in_features}) for layer {layer}.'
            )

    def __call__(self, **kwargs):
        """
        Project every token embedding independently.

        Accepted activation shapes:
        - [ns, c]: projected as standard per-sample linear activations -> [ns, q]
        - [ns, nt, c]: projected per token -> [ns, nt, q]
        - [ns, h, w, c]: flattened to tokens, then projected -> [ns, h*w, q]
        """
        act_data = kwargs['act_data']
        squeeze_token_dim = False

        if act_data.ndim == 2:
            act_data = act_data.unsqueeze(1)
            squeeze_token_dim = True
        elif act_data.ndim == 4:
            act_data = act_data.flatten(start_dim=1, end_dim=2)
        elif act_data.ndim != 3:
            raise RuntimeError(
                f'Expected 2D/3D/4D activations, got shape {tuple(act_data.shape)}.'
            )

        if self.use_bias:
            ones = torch.ones(
                *act_data.shape[:-1],
                1,
                dtype=act_data.dtype,
                device=act_data.device,
            )
            act_data = torch.cat((act_data, ones), dim=-1)

        cvs = torch.matmul(act_data, self.reduct_m.T)
        if squeeze_token_dim:
            cvs = cvs.squeeze(1)
        return cvs

    def parser(self, **kwargs):
        cvs = kwargs['cvs']
        dss = kwargs.get('dss', None)
        label_key = kwargs.get('label_key', 'label')

        tcvs = cvs[..., 0:self.cv_dim]
        ret = tcvs if dss is None else (tcvs, dss[label_key])
        return ret
