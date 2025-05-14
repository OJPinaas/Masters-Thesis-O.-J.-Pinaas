import time
import numpy as np
import brainsignals.neural_simulations as ns
import LFPy
from neuron import h


from brainsignals.neural_simulations import return_hay_cell, return_hallermann_cell
from brainsignals import hallermann_axon_model as ha_ax


def simulate(cell, n_cells, jitter, elecs, popheight, brainsurfacelevel=0,
             celldensity=100000):
    """
    Simulate the cell population and return the potentials.

    Parameters
    ----------
    cell : LFPy.Cell
        The cell to simulate.
    n_cells : int
        Number of cells in the population.
    jitter : float
        Jitter in ms for the spike times.
    elecs : list
        List of electrode objects.
    popheight : float
        Height of the population in µm.
    brainsurfacelevel : float, optional
        Height of the brain surface in µm. The default is 0.
    celldensity : int, optional
        Cell density in cells per mm^3. The default is 100000.

    Returns
    -------
    potentials : np.array
        The potentials recorded at the electrodes.
    morph_data : list
        List of dictionaries containing the x and z coordinates of the cell segments.

    Raises
    ------
    ValueError
        If the electrode type is not recognized.
    """
    # define the simulation parameters
    dt = cell.dt
    tstop = cell.tstop

    num_tsteps = int(tstop / dt + 1)
    pop_radius = find_pop_radius(n_cells, popheight, celldensity)

    # create datastorage
    potentials = np.zeros((len(elecs), num_tsteps * 4))
    morph_data = []

    # create cell population
    rs = pop_radius * np.sqrt(np.random.rand(n_cells))
    theta = np.random.uniform(0, 2 * np.pi, n_cells)
    pop_xs = rs * np.cos(theta)
    pop_ys = rs * np.sin(theta)
    pop_zs = np.random.uniform(0, popheight, n_cells)
    cell_rots = np.random.uniform(0, 2 * np.pi, n_cells)

    # create array for the spike times
    t_shift = np.random.normal(0, jitter, n_cells)

    # place the first cell at the origin
    pop_xs[0] = 0
    pop_ys[0] = 0
    pop_zs[0] = 0
    cell_rots[0] = 0

    # find the difference between the tallest point and the 0 segment of the cell
    # then shift the cell down so the tallest point is 10 um below the
    # brainsurfacelevel
    z_max = np.max(cell.z)
    pop_z_max = popheight
    z_diff = z_max + pop_z_max - cell.z[0].mean()
    z_shift = brainsurfacelevel - 10 - z_diff
    pop_zs += z_shift

    # move the first cell to the origin
    cell.set_pos(x=pop_xs[0], y=pop_ys[0], z=(pop_zs[0]))
    cell.set_rotation(z=cell_rots[0])
    # simulate the cell
    cell.simulate(rec_imem=True, rec_vmem=True)

    # shift potentials to start at 0
    cell.imem -= cell.imem[:, 0, None]
    # loop over the cell population
    split = time.time()
    for cell_idx in range(n_cells):
        print(f"Measuring cell {cell_idx+1} of {n_cells} cells")
        if n_cells > 1 and cell_idx != 0:
            split = time.time() - split
            print((f"Last cell took {round(split)} seconds to simulate"))
            split = time.time()
        # move cell to new position and rotate
        cell.set_pos(x=pop_xs[cell_idx], y=pop_ys[cell_idx], z=(pop_zs[cell_idx]))
        cell.set_rotation(z=cell_rots[cell_idx])

        # move cell spike to proper position
        t_shift_idx = int(t_shift[cell_idx] / dt)
        t0 = num_tsteps + t_shift_idx
        t1 = t0 + len(cell.tvec)

        # loop over the electrodes and measure the potentials
        # if the electrode is a four-sphere model, use the get
        # dipole_potential_from_multi_dipoles method to get the potentials
        # if the electrode is a RecExtElectrode, use the get_transformation_matrix method
        # to get the potentials
        # if not either of these, raise an error
        for i, elec in enumerate(elecs):
            if isinstance(elec, LFPy.FourSphereVolumeConductor):
                pot = elec.get_dipole_potential_from_multi_dipoles(cell) * 1e3
                potentials[i, t0:t1] += pot.squeeze()
            elif isinstance(elec, LFPy.RecExtElectrode):
                # set the cell for the electrode on first loop
                if elec.cell is None:
                    elec.cell = cell
                pot = elec.get_transformation_matrix() @ cell.imem * 1e3
                potentials[i, t0:t1] += pot.squeeze()
            else:
                raise ValueError("Unknown electrode type: {}".format(type(elec)))

        # store the morphology of the cell, y coordinates are not used
        morph_data.append({
            "cell_x": cell.x.copy(),
            "cell_z": cell.z.copy()
            })

    # shift the potentials to start at 0
    potentials -= potentials[:, 0, None]

    return potentials, morph_data


# ─── Simulation setup functions ──────────────────────────────────────────────────
def elec_setup(electrode_positions=None, brainsurfacelevel=0, eeglike=False,
               foursphere=False, method='pointsource'):
    """
    Function to set up the electrode positions for LFP recording.
    The function creates a list of electrode objects at specified positions
    above the cell population. The default positions are:
    - 10 µm above
    - 30 µm above
    - 100 µm above
    - 300 µm above
    - 6 mm above
    - 12 mm

    The function also creates an EEG-like electrode at 12 mm above the cell population.
    The function returns a list of electrode objects.
    The electrodes are created with a point source method and a conductivity of 0.3 S/m.
    The EEG-like electrode is created with a conductivity of 1 S/m.
    The four-sphere electrode is created with a conductivity of 0.276 S/m for
    the first sphere, 1.65 S/m for the second sphere, 0.01 S/m for the third sphere,
    and 0.465 S/m for the fourth sphere.
    The four-sphere model is created using the LFPy.FourSphereVolumeConductor class.
    Eeglike electrode and four-sphere electrode are created at the same position as
    the last electrode.
    These are only created if the corresponding parameters are set to True.

    Parameters
    ----------
    electrode_positions : np.array, optional
        Numpy array of electrode positions on the form [[x, y, z]]. Defaults to None.
    eeglike : bool, optional
        If True, create an EEG-like electrode at the last position. Defaults to False.
    foursphere : bool, optional
        If True, create a four-sphere electrode at the last position. Defaults to False.
    method : str, optional
        Method to use for the electrode. Defaults to 'pointsource'.
    brainsurfacelevel : float, optional
        Height of the brain surface in µm. The default is 0.

    Returns
    -------
    elecs : list
        List of LFPy electrode objects.
    """
    # Define electrode positions
    if electrode_positions is None:
        electrode_positions = np.array([
            [0, 0, brainsurfacelevel],          # 10 µm above
            [0, 0, brainsurfacelevel + 20],     # 30 µm above
            [0, 0, brainsurfacelevel + 90],     # 100 µm above
            [0, 0, brainsurfacelevel + 290],    # 300 µm above
            [0, 0, brainsurfacelevel + 5990],   # 6 mm above
            [0, 0, brainsurfacelevel + 11990],  # 12 mm above
        ])

    # Create a list to store the electrodes
    elecs = []
    # create electrodes at each position
    for i in range(len(electrode_positions)):
        elec = LFPy.RecExtElectrode(
            cell=None,
            x=electrode_positions[i, 0],
            y=electrode_positions[i, 1],
            z=electrode_positions[i, 2],
            method=method,
            sigma=0.3  # Conductivity of the medium (S/m)
        )
        elecs.append(elec)

    if eeglike:
        # Create EEG-like electrode at furthest specified location above the cell
        # population with a lower conductivity (0.02 S/m)
        x, y, z = electrode_positions[-1]
        EEGlikeElec = LFPy.RecExtElectrode(
            cell=None,
            x=x,
            y=y,
            z=z,
            method=method,
            sigma=1  # Higher conductivity (lowers signal) to acount for skull and
                     # skin
        )
        elecs.append(EEGlikeElec)

    if foursphere:
        # Create four-sphere electrode at the same position as the last electrode
        # accounting for the four-sphere model using the
        # LFPy.FourSphereVolumeConductor class
        x, y, z = electrode_positions[-1]
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
        r_electrodes = np.array([x, y, z]).T
        radii = [brainsurfacelevel, brainsurfacelevel + 2000, brainsurfacelevel + 7000,
                 brainsurfacelevel + 12000]  # Radii of the four spheres in µm
        sigmas = [0.276, 1.65, 0.01, 0.465]  # Conductivities of the four spheres in S/m
        # Create the four-sphere model electrode at position of the last electrode
        spheremodel = LFPy.FourSphereVolumeConductor(
            r_electrodes=r_electrodes,
            radii=radii,
            sigmas=sigmas
        )
        elecs.append(spheremodel)
    return elecs


def find_pop_radius(n_cells, somapopheight, celldensity=100000):
    """
    Calculate the radius of a spherical population of cells given the number of cells,
    the height of the soma, and the cell density.
    The formula used is:
        V = n_cells / celldensity
        V = pi * r^2 * h
    Rearranging gives:
        r = sqrt(n_cells / (pi * celldensity * h))

    Parameters
    ----------
    n_cells : int
        Number of cells in the population.
    somapopheight : int
        Height of the cylinder somas can occupy in µm.
    celldensity : int
        Cell density in cells per mm^3.

    Returns
    -------
    somapopradius : float
        Radius of the population in µm.
    """
    somapopheight = somapopheight / 1000  # Convert to mm
    somapopradius = np.sqrt(n_cells / (np.pi * celldensity * somapopheight))
    # Convert back to µm
    somapopradius = somapopradius * 1000
    return somapopradius


# ─── Cell creation functions ────────────────────────────────────────────────────────
def hay_soma(tstop, dt, spike_times=[5], make_passive=True):
    """
    Generate a Hay cell with a single synapse at the soma.
    The cell is created with a single synapse at the soma. The synapse is an
    Exp2Syn synapse with a weight of 0.1 and a time constant of 0.1 ms.
    The synapse is activated at 5 ms by default.
    By default the cell is created with a passive membrane model.

    Parameters
    ----------
    tstop : float
        The time to stop the simulation in ms.
    dt : float
        The time step of the simulation in ms.
    make_passive : bool, optional
        If True, the cell is created with a passive membrane model. The default is True.

    Returns
    -------
    cell : LFPy.Cell
        The cell object.
    """
    weight = 0.1
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=make_passive)
    synapse = LFPy.Synapse(cell, idx=0,
                           syntype='Exp2Syn', weight=weight,
                           tau1=0.1, tau2=1.)
    synapse.set_spike_times(np.array(spike_times))
    return cell


def insert_current_stimuli(cell):
    """
    Insert a current stimulus into the cell at the first segment.
    The current stimulus is a point source current with a duration of 0.3 ms
    and an amplitude of -0.05 nA.

    Parameters
    ----------
    cell : LFPy.Cell
        The cell object to insert the current stimulus into.

    Returns
    -------
    cell : LFPy.Cell
        The cell object with the current stimulus inserted.
    """
    stim_params = {'amp': -0.05,
                   'idx': 0,
                   'pptype': "ISyn",
                   'dur': 0.3,
                   'delay': 0}
    # no need for the returned synapse object from LFPy.StimIntElectrode
    LFPy.StimIntElectrode(cell, **stim_params)
    return cell


def hallermann_axon(tstop, dt, axon_type):
    """
    Generate a Hallermann axon with a single synapse at the axon hillock.

    Parameters
    ----------
    tstop : float
        The time to stop the simulation in ms.
    dt : float
        The time step of the simulation in ms.
    axon_type : str
        The type of axon to create. Must be 'unmyelinated' or 'myelinated'.

    Returns
    -------
    cell : LFPy.Cell
        The cell object.

    Raises
    ------
    ValueError
        If the axon_type is not 'unmyelinated' or 'myelinated'.
    """
    # ─── Select axon constructor ────────────────────────────────────────────────
    if axon_type == "unmyelinated":
        cell_func = ha_ax.return_constructed_unmyelinated_axon
    elif axon_type == "myelinated":
        cell_func = ha_ax.return_constructed_myelinated_axon
    else:
        raise ValueError("Invalid axon_type! Must be 'unmyelinated' or 'myelinated'.")

    # needed to suppress the print statements in the constructor
    import io
    import contextlib
    fake_out = io.StringIO()
    with contextlib.redirect_stdout(fake_out):
        _ = return_hallermann_cell(tstop=tstop, dt=dt)

    # Create the axon
    cell = cell_func(dt, tstop, num_splits=4, split_angle_fraction=1)
    # insert stim electrode in hillock (idx=0)
    cell = insert_current_stimuli(cell)
    return cell


def hay_single_apical_exp(tstop, dt, spike_times=[5], make_passive=True):
    """
    Generate a Hay cell with a single synapse at a random segment of an apical dendrite.
    The cell is created with a single synapse at a random segment of an apical dendrite.
    The synapse is an Exp2Syn synapse with a weight of 0.1 and a time constant of 0.1 ms.
    By default the synapse is activated at 5 ms.
    By default the cell is created with a passive membrane model.

    Parameters
    ----------
    tstop : float
        The time to stop the simulation in ms.
    dt : float
        The time step of the simulation in ms.
    spike_times : list, optional
        The times to activate the synapse in ms. The default is [15, 20, 25].
    make_passive : bool, optional
        If True, the cell is created with a passive membrane model. The default is True.

    Returns
    -------
    cell : LFPy.Cell
        The cell object.
    """

    weight = 0.1
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=make_passive)
    # Find idx of the axon head
    apic_idxs = cell.get_rand_idx_area_norm(section='allsec', z_max=1e9,
                                            z_min=700, nidx=1)
    # Create a synapse at a random apical location
    print("apic_idxs:", apic_idxs, 'length:', len(apic_idxs))
    synapse = LFPy.Synapse(cell, idx=apic_idxs,
                           syntype='Exp2Syn', weight=weight,
                           tau1=0.1, tau2=1.)
    synapse.set_spike_times(np.array(spike_times))
    return cell


def hay_apical(tstop, dt, spike_times=[5], make_passive=True):
    """
    Generate a Hay cell with ten synapses at the apical dendrite.
    The synapses are Exp2Syn synapses with a weight of 0.01 and a time constant of 0.1 ms.
    By default the synapses are activated at 5 ms, all at the same time.
    By default the cell is created with a passive membrane model.

    Parameters
    ----------
    tstop : float
        The time to stop the simulation in ms.
    dt : float
        The time step of the simulation in ms.
    spike_times : list, optional
        The times to activate the synapse in ms. The default is [5].
    make_passive : bool, optional
        If True, the cell is created with a passive membrane model. The default is True.

    Returns
    -------
    cell : LFPy.Cell
        The cell object.
    """

    weight = 0.01
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=make_passive)
    # Find idx of the axon head
    apic_idxs = cell.get_rand_idx_area_norm(section='apic', z_max=1e9,
                                            z_min=700, nidx=10)
    # Create a synapse at a random apical location
    print("apic_idxs:", apic_idxs, 'length:', len(apic_idxs))
    for idx in apic_idxs:
        synapse = LFPy.Synapse(cell, idx=idx,
                               syntype='Exp2Syn', weight=weight,
                               tau1=0.1, tau2=1.)
        synapse.set_spike_times(np.array(spike_times))
    return cell


def hay_active_soma(tstop, dt, spike_times=[5]):
    """
    Generate an active Hay cell with a synapse in the soma.
    The cell is created with active processes. The synapse is an Exp2Syn synapse
    with a weight of 0.1 and a time constant of 0.1 ms.
    The synapse is activated at 5 ms by default.

    Returns
    -------
    cell : LFPy.Cell
        The cell object.
    """
    weight = 0.1
    cell = return_hay_cell(tstop=tstop, dt=dt, make_passive=False)
    synapse = LFPy.Synapse(cell, idx=0,
                           syntype='Exp2Syn', weight=weight,
                           tau1=0.1, tau2=1.)
    synapse.set_spike_times(np.array(spike_times))
    return cell


# ─── Saving and loading functions ────────────────────────────────────────────────
def save_simulation(potentials, morphs, start_time, paramdict=None):
    """
    Save the potentials and morphs to a file.
    The file is saved in the current working directory with the name
    'simulation (start_time).npz'.
    The start time is formatted as 'YYYY-MM-DD HH-MM-SS'.
    The potentials and morphs are saved as numpy arrays.
    The parameters are saved as a dictionary.
    The potentials are saved as a 2D array with shape (n_electrodes, n_time_steps).
    The morphs are saved as a list of dictionaries with the x and z coordinates of the
    cell segments.

    Parameters
    ----------
    potentials : np.array
        The potentials recorded at the electrodes.
    morphs : list
        List of dictionaries containing the x and z coordinates of the cell segments.
    start_time : float
        The start time of the simulation in seconds since the epoch.
    paramdict : dict, optional
        Dictionary of parameters used in the simulation. The default is None.
    """
    # Find start time as hour and minute
    start_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(start_time))
    if paramdict is not None:
        try:
            celltype = paramdict['celltype']
        except ValueError:
            raise ValueError("celltype not found in paramdict. \
            Please provide a valid celltype.")
        try:
            n_cells = paramdict['n_cells']
        except ValueError:
            raise ValueError("n_cells not found in paramdict. \
            Please provide a valid n_cells.")
        try:
            filename = f"{n_cells} x {celltype} simulation {start_time}.npz"
        except ValueError:
            raise ValueError("n_cells and celltype not found in paramdict. \
            Please provide valid n_cells and celltype.")
    else:
        filename = f"simulation {start_time}.npz"
    # save the potentials and morphs to a file
    np.savez(filename, potentials=potentials,
             morphs=morphs, paramdict=paramdict, allow_pickle=True)
    print(f"Simulation saved to file: {filename}")


def load_simulation(filename):
    """
    Load the potentials and morphs from a file.
    The file is loaded from the current working directory with the name specified.
    File has to have .npz format.
    The potentials and morphs are loaded as numpy arrays.
    The parameters are loaded as a dictionary.
    The potentials are loaded as a 2D array with shape (n_electrodes, n_time_steps).
    The morphs are loaded as a list of dictionaries with the x and z coordinates of the
    cell segments.

    Parameters
    ----------
    filename : str
        The name of the file to load. The file must be in the current working directory.

    Returns
    -------
    potentials : np.array
        The potentials recorded at the electrodes.
    morphs : list
        List of dictionaries containing the x and z coordinates of the cell segments.
    parameters : dict
        Dictionary of parameters used in the simulation.
    """
    # load the potentials and morphs from a file
    data = np.load(filename, allow_pickle=True)
    potentials = data['potentials']
    morphs = data['morphs']
    parameters = data['paramdict']
    print(f"Simulation loaded from file: {filename}")
    return potentials, morphs, parameters


# ─── Main function ─────────────────────────────────────────────────────────────
def run(paramdict={}):
    """
    Run the simulation with the given parameters.
    The function creates a cell of the specified type, sets up the electrodes,
    simulates the cell, and saves the results to a file.
    The function also prints the execution time of the simulation.
    The function takes a dictionary of parameters as input. The parameters need
    to be passed with a dictionary:
    parameters
    ----------
    celltype:
        The type of cell to simulate.
    tstop: float
        The time to stop the simulation in ms.
    dt: float
        The time step of the simulation in ms.
    n_cells: int
        The number of cells to simulate.
    celldensity: int
        The cell density in cells per mm^3.
    somapopheight: int
        The height of the population in µm.
    jitter: float
        The jitter in ms for the spike times.
    brainsurfacelevel: int
        The height of the brain surface in µm.
    eeglike: bool
        If True, create an EEG-like electrode at the last position.
    foursphere: bool
        If True, create a four-sphere electrode at the last position.
    method: str
        The method to use for the electrode. Defaults to 'pointsource'.

    raises
    ------
    ValueError
        If the celltype is not any of applicable models.
    """
    start = time.time()
    celltype = paramdict['celltype']
    if celltype == "hay_soma":
        cell = hay_soma(paramdict['tstop'], paramdict['dt'])
    elif celltype == "hallermann_axon":
        cell = hallermann_axon(paramdict['tstop'], paramdict['dt'],
                               axon_type=paramdict['axon_type'])
    elif celltype == "hay_single_apical_exp":
        cell = hay_single_apical_exp(paramdict['tstop'], paramdict['dt'])
    elif celltype == "hay_apical":
        cell = hay_apical(tstop=paramdict['tstop'], dt=paramdict['dt'])
    elif celltype == "hay_active_soma":
        cell = hay_active_soma(paramdict['tstop'], paramdict['dt'])
    else:
        raise ValueError("Invalid cell name! Must be 'hay_soma', 'hallermann_axon', \
                          'hay_apical', 'hay_active_soma', or 'hay_single_apical_exp'.")

    elecs = elec_setup(brainsurfacelevel=paramdict['brainsurfacelevel'],
                       eeglike=paramdict['eeglike'],
                       foursphere=paramdict['foursphere'],
                       method=paramdict['method'])

    potentials, morphs = simulate(cell, paramdict['n_cells'], paramdict['jitter'],
                                  elecs, paramdict['somapopheight'],
                                  brainsurfacelevel=paramdict['brainsurfacelevel'],
                                  celldensity=paramdict['celldensity'])

    save_simulation(potentials, morphs, start, paramdict=parameters)

    end = time.time()
    print(f"Simulation of {paramdict['n_cells']} cells of type {celltype} completed.")
    print(f"""
        Execution time: {int((end - start) // 60)} minutes
        and {int((end - start) % 60)} seconds""")


def runcombined(paramdict={}):
    """
    Run the simulation of combined cells with the given parameters.
    The function creates a hay_apical and a hallermann_axon cell, sets up the electrodes,
    simulates the cells, and saves the results to a file.
    The function also prints the execution time of the simulation.
    The function takes a dictionary of parameters as input. The parameters need
    to be passed with a dictionary:
    parameters
    ----------
    tstop: float
        The time to stop the simulation in ms.
    dt: float
        The time step of the simulation in ms.
    n_cells: int
        The number of cells to simulate.
    celldensity: int
        The cell density in cells per mm^3.
    somapopheight: int
        The height of the population in µm.
    jitter: float
        The jitter in ms for the spike times.
    brainsurfacelevel: int
        The height of the brain surface in µm.
    eeglike: bool
        If True, create an EEG-like electrode at the last position.
    foursphere: bool
        If True, create a four-sphere electrode at the last position.
    method: str
        The method to use for the electrode. Defaults to 'pointsource'.
    axon_type: str
        The type of axon to create. Must be 'unmyelinated' or 'myelinated'.

    raises
    ------
    ValueError
        If axon_type is not valid.
    """
    start = time.time()
    paramdict['celltype'] = 'combined'

    celltype = paramdict['celltype']
    tstop = paramdict['tstop']
    dt = paramdict['dt']
    n_cells = paramdict['n_cells']
    celldensity = paramdict['celldensity']
    popheight = paramdict['somapopheight']
    jitter = paramdict['jitter']
    brainsurfacelevel = paramdict['brainsurfacelevel']
    axon_type = paramdict['axon_type']

    elecs = elec_setup(brainsurfacelevel=paramdict['brainsurfacelevel'],
                       eeglike=paramdict['eeglike'],
                       foursphere=paramdict['foursphere'],
                       method=paramdict['method'])

    num_tsteps = int(tstop / dt + 1)
    pop_radius = find_pop_radius(n_cells, popheight, celldensity)

    # create datastorage
    cell_potentials = np.zeros((len(elecs), num_tsteps * 4))
    axon_potentials = np.zeros((len(elecs), num_tsteps * 4))
    morph_data = []

    # create cell population
    rs = pop_radius * np.sqrt(np.random.rand(n_cells))
    theta = np.random.uniform(0, 2 * np.pi, n_cells)
    pop_xs = rs * np.cos(theta)
    pop_ys = rs * np.sin(theta)
    pop_zs = np.random.uniform(0, popheight, n_cells)
    cell_rots = np.random.uniform(0, 2 * np.pi, n_cells)

    # create array for the spike times
    t_shift = np.random.normal(0, jitter, n_cells)

    # place the first cell at the origin
    pop_xs[0] = 0
    pop_ys[0] = 0
    pop_zs[0] = 0
    cell_rots[0] = 0

    if axon_type == 'myelinated':
        spikeoffset = 26 * dt
    elif axon_type == 'unmyelinated':
        spikeoffset = 103 * dt
    else:
        raise ValueError("Invalid axon_type! Must be 'unmyelinated' or 'myelinated'.")
    pyra = 0
    axon = 1
    split = time.time()
    for model in [pyra, axon]:
        if model is pyra:
            cell = hay_apical(tstop, dt, spike_times=[1 + spikeoffset])
            z_max = np.max(cell.z)
            z_diff = z_max - cell.z[0].mean() + popheight
            z_shift = brainsurfacelevel - 10 - z_diff
            cell_zs = pop_zs + z_shift
            cell.simulate(rec_imem=True, rec_vmem=False)
            cell.imem -= cell.imem[:, 0, None]

        elif model is axon:
            cell = hallermann_axon(tstop, dt, axon_type=axon_type)
            z_max = np.max(cell.z)
            z_diff = z_max - cell.z[0].mean() + popheight
            z_shift = brainsurfacelevel - 10 - z_diff
            cell_zs = pop_zs + z_shift
            cell.simulate(rec_imem=True, rec_vmem=False)
            cell.imem -= cell.imem[:, 0, None]

        for cell_idx in range(n_cells):
            # Timing and progress
            print(f"Measuring cell {cell_idx+1} of {n_cells} cells")
            if n_cells > 1 and cell_idx != 0:
                split = time.time() - split
                print((f"Last cell took {round(split)} seconds to simulate"))
                split = time.time()

            # move cell to new position and rotate
            cell.set_pos(x=pop_xs[cell_idx], y=pop_ys[cell_idx], z=(cell_zs[cell_idx]))
            cell.set_rotation(z=cell_rots[cell_idx])

            # move cell spike to proper position
            t_shift_idx = int(t_shift[cell_idx] / dt)
            t0 = num_tsteps + t_shift_idx
            t1 = t0 + len(cell.tvec)

            # loop over the electrodes and measure the potentials
            # if the electrode is a four-sphere model, use the get
            # dipole_potential_from_multi_dipoles method to get the potentials
            # if the electrode is a RecExtElectrode,
            # use the get_transformation_matrix method
            # to get the potentials
            # if not either of these, raise an error
            for i, elec in enumerate(elecs):
                if model is axon:
                    if isinstance(elec, LFPy.FourSphereVolumeConductor):
                        pot = elec.get_dipole_potential_from_multi_dipoles(cell) * 1e3
                        axon_potentials[i, t0:t1] += pot.squeeze()
                    elif isinstance(elec, LFPy.RecExtElectrode):
                        # set the cell for the electrode on first loop
                        elec.cell = cell
                        pot = elec.get_transformation_matrix() @ cell.imem * 1e3
                        axon_potentials[i, t0:t1] += pot.squeeze()
                    else:
                        raise ValueError("Unknown electrode type: {}".format(type(elec)))
                elif model is pyra:
                    if isinstance(elec, LFPy.FourSphereVolumeConductor):
                        pot = elec.get_dipole_potential_from_multi_dipoles(cell) * 1e3
                        cell_potentials[i, t0:t1] += pot.squeeze()
                    elif isinstance(elec, LFPy.RecExtElectrode):
                        # set the cell for the electrode on first loop
                        elec.cell = cell
                        pot = elec.get_transformation_matrix() @ cell.imem * 1e3
                        cell_potentials[i, t0:t1] += pot.squeeze()
                    else:
                        raise ValueError("Unknown electrode type: {}".format(type(elec)))
            # store the morphology of the cell, y coordinates are not used
            if model is pyra:
                morph_data.append({
                    "cell_x": cell.x.copy(),
                    "cell_z": cell.z.copy()
                    })
            if model is axon:
                morph_data.append({
                    "axon_x": cell.x.copy(),
                    "axon_z": cell.z.copy()
                    })
        # delete to make sure the cell is not used again
        del cell

    # shift the potentials to start at 0
    cell_potentials -= cell_potentials[:, 0, None]
    axon_potentials -= axon_potentials[:, 0, None]

    # add the potentials from the axon and cell together
    potentials = cell_potentials + axon_potentials
    potdict = {}
    potdict['combined'] = potentials
    potdict['axon'] = axon_potentials
    potdict['cell'] = cell_potentials
#   plot_potentials(potentials, dt)
    morphs = morph_data

    save_simulation(potdict, morphs, start, paramdict=paramdict)

    end = time.time()
    print(f"Simulation of {paramdict['n_cells']} cells of type {celltype} completed.")
    print(f"""
        Execution time: {int((end - start) // 60)} minutes
        and {int((end - start) % 60)} seconds""")


def sweep():
    axon_types = ['myelinated', 'unmyelinated']
    celltypeslist = ['hallermann_axon', 'hay_soma', 'hay_single_apical_exp',
                     'hay_apical', 'hay_active_soma']
    n_cellslist = [1, 10, 100, 1000, 10000]
    jitterlist = [0, 1, 2, 5, 10, 15, 20, 25]

    # run the simulation for all combinations of parameters
    for n_cells in n_cellslist:
        parameters['n_cells'] = n_cells
        if n_cells >= 100:
            # removing foursphere for large populations
            parameters['foursphere'] = False
        for jitter in jitterlist:
            parameters['jitter'] = jitter
            for celltype in celltypeslist:
                parameters['celltype'] = celltype
                if celltype == 'hallermann_axon':
                    for axon_type in axon_types:
                        parameters['axon_type'] = axon_type
                        run(paramdict=parameters)
                else:
                    run(paramdict=parameters)


def sweepCombined():
    axon_types = ['myelinated', 'unmyelinated']
    n_cellslist = [1, 10, 100, 1000, 10000]
    jitterlist = [0, 1, 2, 5, 10, 15, 20, 25]

    # run the simulation for all combinations of parameters
    for n_cells in n_cellslist:
        for jitter in jitterlist:
            for axon_type in axon_types:
                parameters['axon_type'] = axon_type
                parameters['n_cells'] = n_cells
                parameters['jitter'] = jitter
                runcombined(paramdict=parameters)


# ─── Example usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    np.random.seed(12345)
    ns.load_mechs_from_folder(ns.cell_models_folder)

    brainsurfacelevel = 87000  # µm
    dt = 2**-5  # ms
    tstop = 150  # ms
    n_cells = 10  # N
    celldensity = 100000  # N/mm**3
    somapopheight = 25  # µm
    jitter = 1  # ms
    foursphere = True
    eeglike = True
    method = 'pointsource'
    axon_type = 'myelinated'

    parameters = {
        "brainsurfacelevel": brainsurfacelevel,
        "dt": dt,
        "tstop": tstop,
        "n_cells": n_cells,
        "celldensity": celldensity,
        "somapopheight": somapopheight,
        "jitter": jitter,
        'method': method,
        "foursphere": foursphere,
        "eeglike": eeglike,
        "axon_type": axon_type
    }

    # Run all simulations for simple models
    sweep()
    # Run all simulations for combined models
    sweepCombined()
