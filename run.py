from models import Poisson2D, RBF, Cauchy, ResNNGP, DenseNNGP, NNet, make_mlp, Kernel, Kernel2, NNetKernel, make_rbf
import jax
import jax.numpy as jnp
import jax.random as jr
import cheb
import optax
import neural_tangents as nt
import tqdm
import pickle
import yaml
import os

def retrieve_samples(config, key):
    samples_cfg = config['train']['samples']
    N = samples_cfg['N']
    dim = config['dim']
    dist = samples_cfg['dist']

    if dist == 'uniform':
        samples = 2 * (jr.uniform(shape=(N, dim), key=key) - 0.5)
        return samples
    else:
        raise NotImplementedError('This distribution is not implemented')


def retrieve_optimizer(config):
    id = config['train']['opt']['id']
    if id == "adam":
        optimizer = optax.adam(config['train']['opt']['lr'])
    else:
        raise ValueError("invalid optimizer")

    return optimizer



def retrieve_op(config):
    id = config['train']['truth']['id'].lower()
    if id == "poisson-exa":
        return Poisson2D()
    else:
        raise NotImplementedError("This operator is not implemented")


def retrieve_grid(config):
    N_grid = config['gridpts']
    assert config['dim'] == 2, "dimension must be two in current implementation"
    ax_grid, weights = cheb.gridpts(N_grid, with_weights=True)
    dim_grid = jnp.stack(jnp.meshgrid(ax_grid, ax_grid), axis=-1)
    dim_weights = jnp.kron(weights, weights)
    return ax_grid, weights, dim_grid, dim_weights


def retrieve_model(config, key):
    if config["model"]["id"].lower() == "mlp":
        return make_mlp(dims=config["model"]["dims"], key=key)
    elif config["model"]["id"].lower() == "rbf":
        return make_rbf(bandwidth=config["model"]["bandwidth"], operator=retrieve_op(config))
    else:
        raise ValueError("IMPLEMENT OTHER KERNELS NEXT")


def save_diagnostics(diagnostics, config):
    d_name = f"diagnostics-p={config['train']['reg']['PINN']:.4f}-g={config['train']['reg']['DATA']:.4f}"
    if not os.path.exists(f"expts/{config['name']}/{d_name}"):
        os.makedirs(f"expts/{config['name']}/{d_name}")

        with open(os.path.join(f"expts/{config['name']}/{d_name}/config.yml"), "w+") as f_out:
            yaml.dump(config, f_out)

    with open(f"expts/{config['name']}/{d_name}/diagnostics.pkl", "wb+") as f_out:
        f_out.write(pickle.dumps(diagnostics))

def generalization_error(quad_grid, quad_weights, R_grid, Y_grid, model, model_diff_grid, params):
    data_residual = model.apply_fn(params, quad_grid) - Y_grid
    diff_residual = model_diff_grid(params) - R_grid
    model_gen_error = jnp.linalg.norm(jnp.sqrt(quad_weights) * data_residual.flatten()) ** 2
    model_phys_error = jnp.linalg.norm(jnp.sqrt(quad_weights) * diff_residual.flatten()) ** 2
    return (model_gen_error, model_phys_error)

def pile(quad_grid, quad_weights, X_sample, Y_sample_noisy, R_grid, model, model_diff_grid, params, operator, config):
    kernel = NNetKernel(model, operator, params)

    data_reg = config['train']['reg']['DATA']
    pinn_reg = config['train']['reg']['PINN']

    N = len(X_sample)
    M = len(quad_grid)

    gma = data_reg
    rho = pinn_reg

    # fitting process:
    #  generate K, H, G, W
    Kxx = kernel.K(X_sample, X_sample)
    G = kernel.G(quad_grid, quad_grid)
    Hxz = kernel.H(X_sample, quad_grid)


    cov = jnp.block([[Kxx, Hxz], [Hxz.T, G]])
    noise = jnp.concatenate((gma * (1/N) * jnp.ones((N,)), rho * quad_weights))

    Fhat = model.apply_fn(params, X_sample)
    Ghat = model_diff_grid(params)
    joint = jnp.concatenate((Fhat, Ghat), axis=0)

    #  compute PILE
    # sanity check: L should be training loss
    L = gma * (1/N) * jnp.linalg.norm(Y_sample_noisy - Fhat) ** 2 + \
        rho * jnp.linalg.norm(jnp.sqrt(quad_weights) * (R_grid - Ghat)) ** 2
    RKHS = jnp.sum(joint.flatten() * jnp.linalg.lstsq(cov, joint.flatten())[0].flatten())

    noise_cst = N * 0.5 * jnp.log(2*jnp.pi*(N/gma)) + 0.5 * (M * jnp.log(2 * jnp.pi / rho) - jnp.sum(jnp.log(quad_weights)))
    _, logdet = jnp.linalg.slogdet(jnp.eye(N + M) + noise.reshape((-1, 1))*cov)
    PILE = L + RKHS + 0.5 * logdet + noise_cst
    return PILE, L, RKHS, logdet, noise_cst

def linear_diagnostics(quad_grid, quad_weights, X_sample, Y_sample_noisy, Y_grid, R_grid, kernel, config):
    data_reg = config['train']['reg']['DATA']
    pinn_reg = config['train']['reg']['PINN']

    N = len(X_sample)
    M = len(quad_grid)
    gma = data_reg
    rho = pinn_reg

    # fitting process:
    #  generate K, H, G, W

    Kxx = kernel.K(X_sample, X_sample)
    Hxz = kernel.H(X_sample, quad_grid)
    Gzz = kernel.G(quad_grid, quad_grid)
    Kxz = kernel.K(X_sample, quad_grid)
    Hzz = kernel.H(quad_grid, quad_grid)
    Kzz = kernel.K(quad_grid, quad_grid)


    cov_obs = jnp.block([[Kxx + (gma/N) * jnp.eye(N), Hxz], [Hxz.T, Gzz + rho * jnp.diag(quad_weights)]])
    cov_prior = jnp.block([[Kxx, Hxz], [Hxz.T, Gzz]])
    cov_cross = jnp.block([[Kxx, Hxz, Kxz], [Hxz.T, Gzz, Hzz.T]]).T
    obs_block = jnp.concatenate([Y_sample_noisy, R_grid], axis=0)
    posterior_mean = cov_cross @ jnp.linalg.lstsq(cov_obs, obs_block)[0]

    Fhat = posterior_mean[:N]
    Ghat = posterior_mean[N:N+M]
    Fhat_grid = posterior_mean[N+M:]

    joint = jnp.concatenate((Fhat, Ghat), axis=0)

    RKHS = jnp.sum(joint.flatten() * jnp.linalg.lstsq(cov_prior, joint)[0].flatten())
    train_data_loss = gma * (1/N) * jnp.linalg.norm(Y_sample_noisy - Fhat)**2
    train_phys_loss = rho * jnp.linalg.norm(jnp.sqrt(quad_weights)*(Ghat - R_grid)) ** 2

    L = train_data_loss + train_phys_loss + RKHS

    noise_cst = N * 0.5 * jnp.log(2 * jnp.pi * (N / gma)) + 0.5 * (
                M * jnp.log(2 * jnp.pi / rho) - jnp.sum(jnp.log(quad_weights)))

    _, logdet = jnp.linalg.slogdet(jnp.eye(N + M) + \
                                   jnp.block([[(gma/N) * Kxx, (gma/N) * Hxz], [rho * quad_weights[:, None] * Hxz.T, rho * quad_weights[:, None] * Gzz]]))
    PILE = L + 0.5 * logdet + noise_cst

    phys_gen = jnp.linalg.norm(jnp.sqrt(quad_weights)*(Ghat - R_grid))**2
    data_gen = jnp.linalg.norm(jnp.sqrt(quad_weights)*(Fhat_grid - Y_grid))**2

    if True:
        import matplotlib.pyplot as plt
        import numpy as np
        plt.suptitle(f'g={gma}, r={rho}')
        plt.subplot(3, 2, 1)
        plt.title('Fhat')
        plt.imshow(np.array(Fhat_grid).reshape((30, 30)))
        plt.colorbar()
        plt.subplot(3, 2, 2)
        plt.title('Ghat')
        plt.imshow(np.array(Ghat).reshape((30, 30)))
        plt.colorbar()
        plt.subplot(3, 2, 3)
        plt.title('Fhat - Y_grid')
        plt.imshow(np.array(Fhat_grid - Y_grid).reshape((30, 30)))
        plt.colorbar()
        plt.subplot(3, 2, 4)
        plt.title('Ghat - R_grid')
        plt.imshow(np.array(Ghat - R_grid).reshape((30, 30)))
        plt.colorbar()
        plt.subplot(3, 2, 5)
        plt.title('Y_grid')
        plt.imshow(np.array(Y_grid).reshape((30, 30)))
        plt.colorbar()
        plt.subplot(3, 2, 6)
        plt.title('R_grid')
        plt.imshow(np.array(R_grid).reshape((30, 30)))
        plt.colorbar()
        plt.show()
    return PILE, L, logdet, RKHS, data_gen, phys_gen



def run(config, key):
    pinn_reg = config['train']['reg']['PINN']
    data_reg = config['train']['reg']['DATA']

    operator = retrieve_op(config)

    X_sample = retrieve_samples(config, key)
    _, _, dim_grid, W = retrieve_grid(config)
    Z = dim_grid.reshape((-1, 2))

    noise_var = config['train']['truth']['noise']


    Y_sample = operator.eval_solution(X_sample)
    Y_sample_noisy = Y_sample + jnp.sqrt(noise_var) * jr.normal(key, shape=(len(X_sample), 1))
    Y_grid = operator.eval_solution(Z)
    R_grid = operator.eval_forcing(Z)

    opt_steps = config['train']['opt']['steps']

    if 'with-boundary' in config['train']['samples'].keys() and config['train']['samples']:
        # TODO: technically, the boundary array doesn't contain the actual boundary, it contains points (1-eps, ...), (-1+eps, ...)
        bdy = jnp.concatenate((dim_grid[0, :], dim_grid[-1, :], dim_grid[:, 0], dim_grid[:, -1]), axis=0)
        X_sample = jnp.concatenate((X_sample, bdy), axis=0)
        Y_sample_noisy = jnp.concatenate((Y_sample_noisy, jnp.zeros((len(bdy), 1))), axis=0)

    if opt_steps > 0:
        gen_diagnostics_interval = config['train']['diagnostics']['gen_every']
        pile_diagnostics_interval = config['train']['diagnostics']['pile_every']

        optimizer = retrieve_optimizer(config)
        model, model_params = retrieve_model(config, key)
        model_diff_grid = lambda params: jax.vmap(operator.apply(lambda _Z: model.apply_fn(params, _Z)[0]), in_axes=(0,))(Z).reshape((-1, 1))

        opt_params = optimizer.init(model_params)


        interp_train_loss = lambda t: \
            lambda p: data_reg * (1/len(X_sample)) * jnp.linalg.norm(model.apply_fn(p, X_sample) - Y_sample_noisy) ** 2 + \
                      t * pinn_reg * jnp.linalg.norm(W**(1/2) * (model_diff_grid(p) - R_grid))** 2

        hybrid_train_loss = lambda p: data_reg * (1/len(X_sample)) * jnp.linalg.norm(model.apply_fn(p, X_sample) - Y_sample_noisy) ** 2 + \
                               pinn_reg * jnp.linalg.norm(W**(1/2) * (model_diff_grid(p) - R_grid))** 2

        data_train_loss = lambda p: data_reg * (1/len(X_sample)) * jnp.linalg.norm(model.apply_fn(p, X_sample) - Y_sample_noisy) ** 2


        gen_errors = []
        pile_scores = []


        t = tqdm.tqdm(opt_steps)
        for i in range(opt_steps):
            do_save = False

            if i % pile_diagnostics_interval == 0:
                do_save = True
                pile_score = pile(quad_grid=Z,
                                  quad_weights=W,
                                  X_sample=X_sample,
                                  Y_sample_noisy=Y_sample_noisy,
                                  R_grid=R_grid,
                                  model=model,
                                  model_diff_grid=model_diff_grid,
                                  params=model_params,
                                  operator=operator,
                                  config=config)

                pile_scores.append(pile_score)
                print(f"PILE = {pile_score[0]}, L={pile_score[1]}, logdet={pile_score[2]}, RKHS={pile_score[3]}")

            if i % gen_diagnostics_interval == 0:
                do_save = True
                gen_error = generalization_error(quad_grid=Z,
                                                 quad_weights=W,
                                                 R_grid=R_grid,
                                                 Y_grid=Y_grid,
                                                 model=model,
                                                 model_diff_grid=model_diff_grid,
                                                 params=model_params)
                gen_errors.append(gen_error)
                print(f"Generalization errors: data = {gen_error[0]}, phys = {gen_error[1]}")

            if do_save:
                diagnostics = {
                    "generalization": gen_errors,
                    "pile": pile_scores
                }
                save_diagnostics(diagnostics, config)

            if i < config['train']['opt']['n_data_pretrain']:
                cur_loss, grad = jax.value_and_grad(data_train_loss)(model_params)
                updates, opt_params = optimizer.update(grad, opt_params, model_params)
                model_params = optax.apply_updates(model_params, updates)
                t.set_description(f'Training loss: {cur_loss:.8f}')
                t.update(1)
            elif i < config['train']['opt']['n_data_pretrain'] + config['train']['opt']['n_interp_train']:
                inc = i - config['train']['opt']['n_data_pretrain']
                tot = (config['train']['opt']['n_data_pretrain'] + config['train']['opt']['n_interp_train'])
                theta = inc / tot
                cur_loss, grad = jax.value_and_grad(interp_train_loss(theta))(model_params)
                updates, opt_params = optimizer.update(grad, opt_params, model_params)
                model_params = optax.apply_updates(model_params, updates)
                t.set_description(f'Training loss: {cur_loss:.8f}')
                t.update(1)
            else:
                # might be worth re-initializing optimizer parameters, but it could have
                # the unintended side effect of resetting momentums and messing things up
                cur_loss, grad = jax.value_and_grad(hybrid_train_loss)(model_params)
                updates, opt_params = optimizer.update(grad, opt_params, model_params)
                model_params = optax.apply_updates(model_params, updates)
                t.set_description(f'Training loss: {cur_loss:.8f}')
                t.update(1)

    if opt_steps == 0:
        kernel = retrieve_model(config, key)
        PILE, L, logdet, RKHS, PINN_loss, DATA_loss = linear_diagnostics(quad_grid=Z,
                                                                         quad_weights=W,
                                                                         X_sample=X_sample,
                                                                         Y_sample_noisy=Y_sample_noisy,
                                                                         Y_grid=Y_grid,
                                                                         R_grid=R_grid,
                                                                         kernel=kernel,
                                                                         config=config)
        pile_score_dict = [{
            "iter": 0,
            "pile": PILE,
            "ker_weighted_loss": L,
            "log_det": logdet,
            "rkhs": RKHS
        }]
        print(f"PILE = {PILE}, L={L}, logdet={logdet}, RKHS={RKHS}")

        gen_error_dict = [{
            "iter": 0,
            "model_gen_error": DATA_loss,
            "model_phys_error": PINN_loss
        }]

        diagnostics = {
            "generalization": gen_error_dict,
            "pile": pile_score_dict
        }
        print(f"DATA LOSS = {DATA_loss}, PINN LOSS = {PINN_loss}")
        save_diagnostics(diagnostics, config)