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

def generalization_error(model, params, operator, W, Z, Y_grid):

    model_gen_error = jnp.linalg.norm(jnp.sqrt(W) @ (model.apply_fn(params, Z) - Y_grid)) ** 2 / len(Z)
    model_phys_error = jnp.linalg.norm(
        jnp.sqrt(W) @ jax.vmap(operator.apply(lambda _Z: model.apply_fn(params, _Z)[0]), in_axes=(0,))(Z)) ** 2 / len(Z)

    return (model_gen_error, model_phys_error)

def pile(W, X, Z, Y_train, Y_grid, model, params, operator, config):

    kernelizer = nt.empirical_kernel_fn(model.apply_fn)
    kfunc = lambda x, y: kernelizer(x, y, 'ntk', params)
    kernel = NNetKernel(model, operator, params)

    data_reg = config['train']['reg']['DATA']
    pinn_reg = config['train']['reg']['PINN']

    N = len(X)
    M = len(Z)
    gamma = data_reg / N
    rho = pinn_reg
    W = jnp.diag(W)

    # fitting process:
    #  generate K, H, G, W
    Kxx = kernel.K(X, X)
    G = kernel.G(Z, Z)
    Hxz = kernel.H(X, Z)


    In = jnp.eye(N)
    O = jnp.zeros((N, M))
    cov = jnp.block([[Kxx, Hxz], [Hxz.T, G]])
    noise = jnp.block([[gamma * In, O], [O.T, rho * W]])

    Fhat = model.apply_fn(params, X)
    Ghat = jax.vmap(operator.apply(lambda _Z: model.apply_fn(params, _Z)[0]), in_axes=(0,))(Z).reshape((-1, 1))
    joint = jnp.concatenate((Fhat, Ghat), axis=0)

    #eta = 0.1
    #  compute PILE
    L = gamma * jnp.linalg.norm(Y_train - Fhat) ** 2 + \
        rho * jnp.linalg.norm(jnp.sqrt(W) @ Ghat) ** 2
    RKHS = jnp.sum(joint.flatten() * jnp.linalg.solve(cov, joint).flatten())

    noise_cst = N * 0.5 * jnp.log(2*jnp.pi*(1/gamma)) + 0.5 * (M * jnp.log(2 * jnp.pi / rho) - jnp.sum(jnp.log(jnp.diag(W))))
    _, logdet = jnp.linalg.slogdet(jnp.eye(N + M) + noise @ cov)
    PILE = L + RKHS + 0.5 * logdet + noise_cst
    RKHS = jnp.sum(joint.flatten() * jnp.linalg.solve(cov, joint).flatten())
    return PILE, L, RKHS, logdet, noise_cst

def linear_diagnostics(W, X, Z, Y_train, Y_grid, kernel, config):
    data_reg = config['train']['reg']['DATA']
    pinn_reg = config['train']['reg']['PINN']

    N = len(X)
    M = len(Z)
    gamma = data_reg / N
    rho = pinn_reg
    W = jnp.diag(W)

    # fitting process:
    #  generate K, H, G, W
    Kxx = kernel.K(X, X)
    Kzz = kernel.K(Z, Z)
    Kxz = kernel.K(X, Z)
    Hxz = kernel.H(X, Z)
    Hzz = kernel.H(Z, Z)
    G = kernel.G(Z, Z)

    cov = jnp.block([[Kxx, Hxz, Kxz], [Hxz.T, G, Hzz.T], [Kxz.T, Hzz, Kzz]])
    In = jnp.eye(N)
    O = jnp.zeros((N, M))
    Om = jnp.zeros((M, M))
    noise = jnp.block([[gamma*In, O, O], [O.T, rho*W, Om], [O.T, Om, Om]])
    Y_ = jnp.concatenate((Y_train, jnp.zeros((2*M, 1))), axis=0)

    Yhat = jnp.linalg.solve(jnp.eye(N+2*M) + cov @ noise, cov @ noise @ Y_)
    Fhat = Yhat[:N]
    Ghat = Yhat[N:N+M]
    Fhat_grid = Yhat[N+M:]

    joint = jnp.concatenate((Fhat, Ghat), axis=0)

    L = gamma * jnp.linalg.norm(Y_train - Fhat) ** 2 + \
        rho * jnp.linalg.norm(jnp.sqrt(W) @ Ghat) ** 2 + \
        jnp.sum(joint.flatten() * jnp.linalg.solve(cov[:N+M, :N+M], joint).flatten())

    noise_cst = N * 0.5 * jnp.log(2 * jnp.pi * (1 / gamma)) + 0.5 * (
                M * jnp.log(2 * jnp.pi / rho) - jnp.sum(jnp.log(jnp.diag(W))))
    _, logdet = jnp.linalg.slogdet(jnp.eye(N + M) + noise[:N+M, :N+M] @ cov[:N+M,:N+M])
    PILE = L + 0.5 * logdet + noise_cst
    RKHS = jnp.sum(joint.flatten() * jnp.linalg.solve(cov[:N+M, :N+M], joint).flatten())

    PINN_loss = jnp.linalg.norm(W @ Ghat)**2/M
    DATA_loss = jnp.linalg.norm(jnp.sqrt(W)@(Fhat_grid - Y_grid))**2/N

    return PILE, L, logdet, RKHS, PINN_loss, DATA_loss



def run(config, key):
    pinn_reg = config['train']['reg']['PINN']
    data_reg = config['train']['reg']['DATA']

    operator = retrieve_op(config)

    X = retrieve_samples(config, key)
    _, _, dim_grid, W = retrieve_grid(config)
    Z = dim_grid.reshape((-1, 2))

    noise_var = config['train']['truth']['noise']
    Y_true = operator.eval_solution(X)
    Y_noisy = Y_true + jnp.sqrt(noise_var) * jr.normal(key, shape=(len(X), 1))
    Y_grid = operator.eval_solution(Z)

    opt_steps = config['train']['opt']['steps']


    if opt_steps > 0:
        gen_diagnostics_interval = config['train']['diagnostics']['gen_every']
        pile_diagnostics_interval = config['train']['diagnostics']['pile_every']

        optimizer = retrieve_optimizer(config)
        model, model_params = retrieve_model(config, key)

        opt_params = optimizer.init(model_params)

        train_loss = lambda p: (data_reg / len(X)) * jnp.linalg.norm(model.apply_fn(p, X) - Y_noisy) ** 2 + \
                               (pinn_reg) * jnp.linalg.norm(W**(1/2) @ jax.vmap(operator.apply(lambda _Z: model.apply_fn(p, _Z)[0]), in_axes=(0,))(Z)) ** 2


        gen_errors = []
        pile_scores = []


        t = tqdm.tqdm(opt_steps)
        for i in range(opt_steps):
            do_save = False
            if i % pile_diagnostics_interval == 0:
                do_save = True
                pile_score = pile(W, X, Z, Y_noisy, Y_grid, model, model_params, operator, config)
                pile_scores.append({
                    "iter": i,
                    "pile": pile_score[0],
                    "train_loss": pile_score[1],
                    "rkhs": pile_score[2],
                    "log_det": pile_score[3],
                    "noise_cst": pile_score[4]
                })
                print(f"PILE = {pile_score[0]}, L={pile_score[1]}, logdet={pile_score[2]}, RKHS={pile_score[3]}")


            if i % gen_diagnostics_interval == 0:
                do_save = True
                gen_error = generalization_error(model, model_params, operator, W, Z, Y_grid)
                gen_errors.append({
                    "iter": i,
                    "model_gen_error": gen_error[0],
                    "model_phys_error": gen_error[1]
                })
                print(gen_error)

            if do_save:
                diagnostics = {
                    "generalization": gen_errors,
                    "pile": pile_scores
                }
                save_diagnostics(diagnostics, config)

            cur_loss, grad = jax.value_and_grad(train_loss)(model_params)
            updates, opt_params = optimizer.update(grad, opt_params, model_params)
            model_params = optax.apply_updates(model_params, updates)
            t.set_description(f'Training loss: {cur_loss:.8f}')
            t.update(1)

    if opt_steps == 0:
        kernel = retrieve_model(config, key)
        PILE, L, logdet, RKHS, PINN_loss, DATA_loss = linear_diagnostics(W, X, Z, Y_noisy, Y_grid, kernel, config)
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