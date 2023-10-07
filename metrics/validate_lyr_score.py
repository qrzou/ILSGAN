import torch

def validate_lyr_score(G, D, device, mode='mean', n=1000):
    assert mode in ['mean', 'sign']
    # Generate images.
    G = G.to(device)
    D = D.to(device)
    batch_gpu = 16 * 4
    truncation_psi = 1
    noise_mode = 'const'
    fg_score_list = []
    bg_score_list = []
    img_score_list = []

    print('Validating lyr score...')

    for base_i in range(0, n, batch_gpu):
        this_batch = min(batch_gpu, n - base_i)
        z = torch.randn(this_batch, G.z_dim).to(device)
        c = None

        lyr = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode, return_layers=True)
        bg = lyr['bg']
        fg = lyr['fg']
        img = lyr['img']

        fg_score = D(fg, c, get_feature=False)
        bg_score = D(bg, c, get_feature=False)
        img_score = D(img, c, get_feature=False)

        fg_score_list.append(fg_score)
        bg_score_list.append(bg_score)
        img_score_list.append(img_score)

    if mode == 'mean':
        fg = torch.cat(fg_score_list).mean()
        bg = torch.cat(bg_score_list).mean()
        img = torch.cat(img_score_list).mean()
    elif mode == 'sign':
        fg = torch.cat(fg_score_list).sign().mean()
        bg = torch.cat(bg_score_list).sign().mean()
        img = torch.cat(img_score_list).sign().mean()

    print('fg:', fg, 'bg:', bg, 'img:', img)

    return fg, bg, img

