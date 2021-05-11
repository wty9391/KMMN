def init_gaussian_mixture_parameter(n_components, min_z=1, max_z=300):
    moutinoulli = []
    mean = []
    variance = []
    step = (max_z - min_z) / (n_components+1)

    for i in range(n_components):
        moutinoulli.append(1)
        this_mean = (i+1)*step + min_z
        mean.append(this_mean)
        variance.append((step/2)**2)

    return moutinoulli, mean, variance


def cdf_to_pdf(cdf):
    r = [cdf[0]]
    r.extend([cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(0, len(cdf) - 1)])
    return r


