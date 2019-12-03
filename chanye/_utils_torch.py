from chanye._visualizer import denormalize, show, show_batch, reshape_batch


def torch2numpy(tensor, is_img=True):
    try:
        tensor.cpu().numpy()
    except:
        tensor = tensor.detach()

    if is_img:
        if len(tensor.size()) == 4:
            return tensor.cpu().numpy().transpose([0, 2, 3, 1])
        elif len(tensor.size()) == 3:
            return tensor.cpu().numpy().transpose([1, 2, 0])
    return tensor.cpu().numpy()


def reset_gradients(optimizers):
    assert type(optimizers) == list
    for optimizer in optimizers:
        optimizer.zero_grad()


def sample_from_loader(loader, device):
    for n, data in enumerate(loader):
        if type(data) == list:
            return [i.to(device) if i.__class__.__name__ == 'Tensor' else i for i in data]
        return data.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def denormalize_torch(tensor, mode='tanh'):
    img = torch2numpy(tensor, is_img=True)
    return denormalize(img, mode)


def reshape_batch_torch(tensor, *a, **k):
    array = torch2numpy(tensor, is_img=True)
    return reshape_batch(array, *a, **k)


def show_torch(tensor, *a, **k):
    array = torch2numpy(tensor, is_img=True)
    return show(array, *a, **k)


def show_batch_torch(tensor, *a, **k):
    arrays = torch2numpy(tensor, is_img=True)
    return show_batch(arrays, *a, **k)
