def parse_superquad_file(filename: str) -> dict:
    """Import data from a superquad file.

    Parameters:
    -----------
    filename: str
        The file to read data from.
    """
    data = import_superquad_data(filename)
    outdata = {}
    _ = next(data)    # control numbers (unused)
    labels = list(next(data))
    outdata['components'] = labels
    temperature = next(data)
    outdata['temperature'] = temperature

    logB = next(data)
    outdata['log_beta'] = logB

    stoich = next(data)
    outdata['stoichiometry'] = stoich
    beta_flags = next(data)

    titrations = []
    outdata['titrations'] = titrations
    for emfd in data:
        # cascade unpacking
        amounts, electr, dataset = emfd
        plot_keys, order, t0, buret, tflag = amounts
        V0, errV, n, hindex, emf0, erremf0 = electr
        V, emf = dataset
        titrations.append({
            'initial amount': t0,
            'buret concentration': buret,
            'standard potential': emf0,
            'potential error': erremf0,
            'electroactive': hindex,
            'starting volume': V0,
            'volume error': errV,
            'potential': emf,
            'titre': V
        })
    return outdata
        

def import_superquad_data(filename):
    """Import data from Superquad file.

    This function imports data from a file which complies with SUPERQUAD
    file format.

    Parameters:
        filename (string or file): A readable source from where the data is read.

    Yields:
        * title (str): title of the project
        * control numbers (sequence)
        * labels (list of str): the labels of the principal components
        * temperature (float): the temperature
        * logB (:class:`numpy.ndarray`): the constants in log10 units
        * P (:class:`numpy.ndarray`): the stoichiometric coefficients
        * flags (:class:`numpy.ndarray`): the refinement flags
        * emf (generator of :class:`numpy.ndarray`): the potential read
        * V (generator of :class:`numpy.ndarray`): the volume of titre
        * E0 (generator of floats): the standard potential
        * n (generator of int): the number of electrons involved
        * E0flags (generator of int): the refinement flags for E0
        * error_E0 (generator of float): the error associated to E0
        * V0 (generator of float): the initial volume
        * error_V0 (generator of float): the error associated to volume
        * T0 (generator of :class:`numpy.ndarray`): the initial amounts for the
          principal components
        * T0flags (generator of :class:`numpy.ndarray`): the refinement flags
          for T0.
        * buret (generator of :class:`numpy.ndarray`): the concentration in the
          buret.
        * hindex (generator of int): The index of the electroactive component.
        * fRTnF (generator of float): Nernst's propocionality number.
    """
    def read_amounts(handler):
        plot_keys = []
        order = []
        t0 = []
        buret = []
        tflag = []
        for line in handler:
            if line.strip() == '':
                if len(t0) == 0:
                      return None
                else:
                      return plot_keys, order, t0, buret, tflag
                # break
                # return plot_keys, order, t0, buret, tflag

            aux = line.split()
            plot_keys.append(int(aux[0]))
            order.append(int(aux[1]))
            t0.append(float(aux[2]))
            buret.append(float(aux[3]))
            tflag.append(int(aux[4]))

    def read_electrodes(handler):
        volume, err_volume = map(float, handler.readline().split())
        aux = handler.readline().split()
        n, hindex = map(int, aux[0:2])
        emf0, erremf0 = map(float, aux[2:4])
        assert handler.readline().strip() == ''
        return volume, err_volume, n, hindex, emf0, erremf0

    def read_data(handler):
        aux = []
        for line in handler:
            if line.strip() == '':
                break
            aux.append(tuple(map(float, line.split())))
        return tuple(zip(*aux))

    def read_titration(handler):
        while True:
            amm = read_amounts(handler)
            if amm is None:
                return
            elc = read_electrodes(handler)
            dat = read_data(handler)

            yield (amm, elc, dat)

    with open(filename, "r") as f:
        yield f.readline()      # title
        numbers = tuple(int(i) for i in f.readline().split())
        yield numbers   # control numbers
        num_species = numbers[2]
        yield tuple(f.readline().strip() for i in range(num_species))  # labels
        yield float(f.readline())  # temperature

        B = []
        P = []
        keys = []
        for line in f:
            if line.strip() == '':
                break
            b_, *p_, k_ = line.split()
            B.append(float(b_))
            P.append([int(_) for _ in p_])
            keys.append(int(k_))

        yield B
        yield P
        yield keys
        yield from read_titration(f)
