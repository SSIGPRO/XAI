import sys
import random
from pathlib import Path
from datetime import datetime
from time import time

from tqdm import tqdm

sys.path.insert(0, (Path.home() / 'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home() / 'repos/XAI/src/autoattacks').as_posix())

import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from cuda_selector import auto_cuda

# Register Debenedetti2022 XCiT architectures with timm
import robustbench.model_zoo.architectures.xcit
import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.training.trainingBase import Trainer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Standard training of XCiT-L12 on CIFAR-100')
parser.add_argument('--epochs',         type=int,   default=300,  help='Number of training epochs')
parser.add_argument('--bs',             type=int,   default=2**10, help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-3, help='Initial learning rate for AdamW')
parser.add_argument('--min-lr',         type=float, default=1e-6, help='Minimum learning rate for cosine annealing')
parser.add_argument('--wd',             type=float, default=5e-2, help='Weight decay for AdamW')
parser.add_argument('--drop-path-rate', type=float, default=0.1,  help='Stochastic depth drop path rate')
parser.add_argument('--warmup-epochs',  type=int,   default=20,   help='Linear warmup epochs before cosine annealing')
parser.add_argument('--warmup-lr-factor', type=float, default=1e-3, help='Warmup starts at lr * this factor')
parser.add_argument('--clip-grad',      type=float, default=1.0,  help='Max grad norm; use <=0 to disable clipping')
parser.add_argument('--best-metric', choices=['acc', 'loss'], default='loss', help='Validation metric used for best checkpoint selection')
parser.add_argument('--patience',       type=int,   default=75,   help='Early stopping patience when --early-stopping is enabled')
parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping on the selected validation metric')
parser.add_argument('--save-every',     type=int,   default=10,   help='Save checkpoint every N epochs')
parser.add_argument('--n-threads',      type=int,   default=4,    help='DataLoader worker threads')
parser.add_argument('--train-ratio',    type=float, default=0.8,  help='Fraction of training data used for train (rest -> val)')
parser.add_argument('--seed',           type=int,   default=29,   help='Random seed for train/val split')
parser.add_argument('--mixup-alpha',    type=float, default=0.8,  help='MixUp alpha; use 0 to disable MixUp')
parser.add_argument('--cutmix-alpha',   type=float, default=1.0,  help='CutMix alpha; use 0 to disable CutMix')
parser.add_argument('--mixup-prob',     type=float, default=1.0,  help='Probability of applying MixUp/CutMix')
parser.add_argument('--switch-prob',    type=float, default=0.5,  help='Probability of switching from MixUp to CutMix')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing used by MixUp soft targets')
parser.add_argument('--random-erasing', type=float, default=0.25, help='Random erasing probability after ToTensor; use 0 to disable')
parser.add_argument(
    '--data-path',
    default='/srv/newpenny/dataset/CIFAR100',
    help='Root directory for CIFAR-100 download/cache',
)
parser.add_argument(
    '--ckpt-dir',
    default=Path.cwd() / 'checkpoints' / 'xcit_l12_standard',
    help='Directory for intermediate training checkpoints and loss plots',
)
parser.add_argument(
    '--out-path',
    default=Path.cwd() / 'models' / 'xcit_l12_cifar100_standard.pth',
    help='Final output path for the best model state dict (used by configs/models/xcit_l12.py)',
)
args = parser.parse_args()

# MixUp+CutMix fn and its loss — created in __main__, used in the train loop.
mixup_fn  = None
soft_loss = None


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adamw_param_groups(module, weight_decay):
    decay = []
    no_decay = []
    no_decay_names = ('bias', 'cls_token', 'dist_token', 'pos_embed')

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(no_decay_names):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def append_train_transform(dataset, transform):
    train_ds = dataset.__dataset__['CIFAR100-train']
    base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds

    if hasattr(base_ds.transform, 'transforms'):
        base_ds.transform.transforms.append(transform)
    else:
        base_ds.transform = transforms.Compose([base_ds.transform, transform])


def build_scheduler(optimizer):
    warmup_epochs = min(max(args.warmup_epochs, 0), max(args.epochs - 1, 0))
    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)

    if warmup_epochs == 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=args.warmup_lr_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def AccuracyAwareValidationLoop(**kwargs):
    trainer = kwargs['trainer']
    epoch = kwargs['epoch']
    t0 = kwargs.get('t0', None)

    trainer.model._model.eval()
    with torch.no_grad():
        loss_acc = 0.0
        acc_acc = 0.0
        samples_acc = 0

        pbar = tqdm(zip(range(trainer.iter_val), trainer.val_dl),
                    total=trainer.iter_val, desc=f'Validation {epoch}', leave=False)

        for _, _data in pbar:
            data = trainer.in_parser(_data)
            images = data['image'].contiguous().to(trainer.device, non_blocking=True)
            labels = data['label'].contiguous().to(trainer.device, non_blocking=True)
            n_samples = len(images)
            samples_acc += n_samples

            pred = trainer.out_parser(trainer.model(images))

            loss_acc += trainer.loss_fn(pred, labels)
            acc_acc += trainer.acc_fn(pred, labels)

            pbar.set_postfix(loss=f'{(loss_acc / samples_acc).item():.4f}',
                             acc=f'{(acc_acc / samples_acc).item() * 100:.2f}%')

        loss_mean = loss_acc / samples_acc
        acc_mean = acc_acc / samples_acc

        loss_cpu = loss_mean.detach().cpu()
        acc_cpu = acc_mean.detach().cpu()
        trainer.val_losses[epoch] = loss_cpu
        trainer.val_acc[epoch] = acc_cpu

    if trainer.scheduler is not None:
        if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            old_lr = trainer.optim.param_groups[0]['lr']
            trainer.scheduler.step(trainer.val_losses[epoch])
            current_lr = trainer.optim.param_groups[0]['lr']

            if old_lr != current_lr and trainer.verbose:
                print(f'New LR: {current_lr:.6f}')
        else:
            trainer.scheduler.step()

    if not hasattr(trainer, 'best_val_acc'):
        if epoch > 0:
            trainer.best_val_acc = trainer.val_acc[:epoch].max()
        else:
            trainer.best_val_acc = torch.tensor(float('-inf'))

    improved = (
        acc_cpu > trainer.best_val_acc
        if args.best_metric == 'acc'
        else loss_cpu < trainer.best_val_loss
    )

    if improved:
        trainer.best_val_loss = loss_cpu
        trainer.best_val_acc = acc_cpu
        trainer.best_epoch = epoch
        if trainer.num_bad_epochs is not None:
            trainer.num_bad_epochs = 0

        best_model_path = getattr(trainer, 'best_model_path', trainer.path / 'best_model')
        best_model_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'train_losses': trainer.train_losses[: epoch + 1],
            'train_accuracy': trainer.train_acc[: epoch + 1],
            'val_losses': trainer.val_losses[: epoch + 1],
            'val_accuracy': trainer.val_acc[: epoch + 1],
            'state_dict': trainer.model._model.state_dict(),
            'optimizer': trainer.optim.state_dict(),
            'scheduler': trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
            'best_epoch': trainer.best_epoch,
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'best_metric': args.best_metric,
            'num_bad_epochs': trainer.num_bad_epochs,
        }
        torch.save(checkpoint, best_model_path / 'best_model_config.pt')

        if trainer.verbose:
            if args.best_metric == 'acc':
                print(f'  -> New best validation accuracy: {trainer.best_val_acc * 100:.2f}')
            else:
                print(f'  -> New best validation loss: {trainer.best_val_loss:.6f}')
    else:
        if trainer.num_bad_epochs is not None:
            trainer.num_bad_epochs += 1

    if trainer.early_stopping:
        if trainer.num_bad_epochs > trainer.early_stopping_patience:
            if trainer.verbose:
                print(
                    f'Early stopping: no improvement for {trainer.num_bad_epochs} epochs '
                    f'(patience={trainer.early_stopping_patience}).'
                )
            return True

    if trainer.verbose and t0 is not None:
        print(
            f'epoch {epoch} - train loss: {trainer.train_losses[epoch]:.4f} - '
            f'val loss: {trainer.val_losses[epoch]:.4f} - '
            f'train acc: {trainer.train_acc[epoch] * 100:.2f} - '
            f'val acc: {trainer.val_acc[epoch] * 100:.2f} - '
            f'time: {time() - t0:.2f}'
        )

    return False


# ---------------------------------------------------------------------------
# Mini-batch train loop (replaces the default epoch-level accumulation loop)
# The default loop accumulates all mini-batch losses before a single backward
# pass, which is memory-prohibitive for a large transformer. This version
# does the standard per-batch forward → loss → backward → step cycle.
# When mixup_fn is set, images and labels are mixed before the forward pass.
# The optimizer step uses a mean loss. Loss accumulation remains sum-scaled so
# train/val/test logs stay comparable with peepholelib's default loops.
# ---------------------------------------------------------------------------

def MinibatchTrainLoop(**kwargs):
    trainer = kwargs['trainer']
    epoch   = kwargs['epoch']

    loss_acc    = 0.0
    acc_acc     = 0.0
    samples_acc = 0

    trainer.model._model.train()

    current_lr = trainer.optim.param_groups[0]['lr']
    print(f'  lr: {current_lr:.2e}', flush=True)

    pbar = tqdm(zip(range(trainer.iter_train), trainer.train_dl),
                total=trainer.iter_train, desc=f'Epoch {epoch}', leave=False)

    for _, _data in pbar:
        data   = trainer.in_parser(_data)
        images = data['image'].contiguous().to(trainer.device, non_blocking=True)
        labels = data['label'].contiguous().to(trainer.device, non_blocking=True)
        n      = len(images)
        samples_acc += n

        if mixup_fn is not None:
            images, soft_labels = mixup_fn(images, labels)

        trainer.optim.zero_grad(set_to_none=True)

        pred = trainer.out_parser(trainer.model(images))

        if mixup_fn is not None:
            loss = soft_loss(pred, soft_labels)
            loss_acc += loss.detach() * n
            acc  = (pred.argmax(1) == soft_labels.argmax(1)).float().sum()
        else:
            loss_sum = trainer.loss_fn(pred, labels)
            loss = loss_sum / n
            loss_acc += loss_sum.detach()
            acc  = trainer.acc_fn(pred, labels)

        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(trainer.model._model.parameters(), max_norm=args.clip_grad)
        trainer.optim.step()

        acc_acc  += acc.detach()

        pbar.set_postfix(loss=f'{(loss_acc / samples_acc).item():.4f}',
                         acc=f'{(acc_acc / samples_acc).item() * 100:.2f}%')

    trainer.train_losses[epoch] = (loss_acc / samples_acc).cpu()
    trainer.train_acc[epoch]    = (acc_acc  / samples_acc).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # use_cuda = torch.cuda.is_available()
    # device   = torch.device(auto_cuda('memory')) if use_cuda else torch.device('cpu')
    # print(f'Using {device}')
    gpu_id = 3  # physical GPU index
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    print(f"Using {device} device")

    n_classes = 100
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir  = Path(args.ckpt_dir) / timestamp
    out_path  = Path(args.out_path).with_stem(Path(args.out_path).stem + f'_{timestamp}')
    print(f"Run timestamp: {timestamp}")

    # -----------------------------------------------------------------------
    # Dataset
    # XCiT has normalization baked in (ImageNormalizer), so no normalization
    # transform needed here — only pixel-level augmentation for training.
    # -----------------------------------------------------------------------

    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    ])

    dataset = Cifar100(
        path          = args.data_path,
        aug_transform = aug,
        train_ratio   = args.train_ratio,
        seed          = args.seed,
    )
    dataset.__load_data__()

    if args.random_erasing > 0:
        append_train_transform(
            dataset,
            transforms.RandomErasing(p=args.random_erasing, value='random'),
        )

    # -----------------------------------------------------------------------
    # MixUp + CutMix
    # Applied per-batch inside MinibatchTrainLoop. Label smoothing is folded
    # into the soft targets produced by Mixup, so no separate LS loss.
    # -----------------------------------------------------------------------

    if args.mixup_prob > 0 and (args.mixup_alpha > 0 or args.cutmix_alpha > 0):
        mixup_fn = Mixup(
            mixup_alpha     = args.mixup_alpha,
            cutmix_alpha    = args.cutmix_alpha,
            prob            = args.mixup_prob,
            switch_prob     = args.switch_prob,
            label_smoothing = args.label_smoothing,
            num_classes     = n_classes,
        )
        soft_loss = SoftTargetCrossEntropy()

    # -----------------------------------------------------------------------
    # Model
    # timm creates Sequential(normalize=ImageNormalizer, model=Xcit).
    # ModelWrap.__init__ sets requires_grad=False; get_trainable_parameters
    # re-enables gradients for all parameters before passing them to AdamW.
    # drop_path_rate enables stochastic depth: each residual block is dropped
    # with linearly increasing probability up to drop_path_rate at the last layer.
    # -----------------------------------------------------------------------

    arch = timm.create_model(
        'debenedetti2022light_xcit_l12_cifar100_linf',
        pretrained      = False,
        num_classes     = n_classes,
        drop_path_rate  = args.drop_path_rate,
    )

    model = ModelWrap(model=arch, device=device)

    model.get_trainable_parameters(
        layers_to_train = None,
        verbose         = True,
    )

    # -----------------------------------------------------------------------
    # Optimizer and scheduler
    # AdamW + warmup cosine is the standard recipe for ViT/XCiT models.
    # -----------------------------------------------------------------------

    param_groups = adamw_param_groups(model._model, weight_decay=args.wd)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=0.0)
    scheduler = build_scheduler(optimizer)

    # -----------------------------------------------------------------------
    # Dataloader kwargs
    # -----------------------------------------------------------------------

    dl_kwargs = dict(
        collate_fn          = default_collate,
        num_workers         = args.n_threads,
        pin_memory          = device.type == 'cuda',
        persistent_workers  = args.n_threads > 0,
    )
    if args.n_threads > 0:
        dl_kwargs['prefetch_factor'] = 2

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------

    trainer = Trainer(
        model                   = model,
        path                    = ckpt_dir,
        name                    = 'xcit_l12_standard',
        dataset                 = dataset,
        train_key               = 'CIFAR100-train',
        val_key                 = 'CIFAR100-val',
        test_key                = 'CIFAR100-test',
        batch_size              = args.bs,
        dataloader_kwargs       = dl_kwargs,
        iterations              = 'full',
        optimizer               = optimizer,
        scheduler               = scheduler,
        max_epochs              = args.epochs,
        save_every              = args.save_every,
        early_stopping          = args.early_stopping,
        early_stopping_patience = args.patience,
        verbose                 = True,
        train_loop              = MinibatchTrainLoop,
        validation_loop         = AccuracyAwareValidationLoop,
    )

    trainer.fit()

    # -----------------------------------------------------------------------
    # Test on the held-out test set using the best checkpoint
    # -----------------------------------------------------------------------

    test_loss, test_acc = trainer.test()
    print(f'\nTest accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}')

    # -----------------------------------------------------------------------
    # Export: load the best state dict and save it in the format expected by
    # configs/models/xcit_l12.py (plain state dict, no wrapper keys).
    # -----------------------------------------------------------------------

    best_ckpt = torch.load(
        ckpt_dir / 'best_model' / 'best_model_config.pt',
        weights_only=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_ckpt['state_dict'], out_path)
    print(f'Best model state dict saved to {out_path}')
