import time
import numpy as np
import matplotlib.pyplot as plt
import brainsignals.neural_simulations as ns
import LFPy
from neuron import h


from brainsignals.neural_simulations import return_hay_cell, return_hallermann_cell
from brainsignals import hallermann_axon_model as ha_ax
# from brainsignals import plotting_convention as pc


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


# ─── Plotting functions ────────────────────────────────────────────────────────
def plot_membrane_potential(cell):
    """Plot the membrane potential of the first segment of the cell"""
    plt.plot(cell.tvec, cell.vmem[0], label="Cell 1")
    plt.show()


def plot_morphology(morphs, brainsurfacelevel):
    """Plot the morphology of the cells and the surface level of the brain"""
    fig, ax = plt.subplots()
    n_cells = len(morphs)
    for i, morph in enumerate(morphs):
        ax.plot(morph['cell_x'].T, morph['cell_z'].T,
                color=plt.cm.Greys(0.4 + i / n_cells * 0.6), rasterized=False)
    # plot brain surface level
    ax.axhline(y=brainsurfacelevel, color='k', linestyle='--',
               label='Brain surface level')
    ax.set_xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    ax.set_title("Morphology of simulated cells")
    plt.show()


def plot_morphs_elecs(morphs, elecs, brainsurfacelevel):
    """Plot the morphology of the cells with the electrodes and the surface level
    of the brain"""
    fig, ax = plt.subplots(figsize=(4, 12))
    # plot the morphology of the cells
    n_cells = len(morphs)
    for i, morph in enumerate(morphs):
        ax.plot(morph['cell_x'].T, morph['cell_z'].T,
                color=plt.cm.Greys(0.4 + i / n_cells * 0.6), rasterized=False)
    # plot the electrodes
    for i, elec in enumerate(elecs):
        if isinstance(elec, LFPy.FourSphereVolumeConductor):
            ax.scatter(elec.rxyz[:, 0], elec.rxyz[:, 2], color='r',
                       label='Electrode {}'.format(i+1))
        elif isinstance(elec, LFPy.RecExtElectrode):
            ax.scatter(elec.x, elec.z, color='b', label='Electrode {}'.format(i+1))
        else:
            raise ValueError("Unknown electrode type: {}".format(type(elec)))
    # mark the electrodes
    for i, elec in enumerate(elecs):
        if isinstance(elec, LFPy.FourSphereVolumeConductor):
            ax.annotate(f"Electrode {i+1}", (elec.rxyz[0, 0], elec.rxyz[0, 2]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        elif isinstance(elec, LFPy.RecExtElectrode):
            ax.annotate(f"Electrode {i+1}", (elec.x, elec.z),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            raise ValueError("Unknown electrode type: {}".format(type(elec)))
    
    # plot brain surface level
    ax.axhline(y=brainsurfacelevel, color='k', linestyle='--',
               label='Brain surface level')

    ax.set_yscale('log')
    ax.set_xlabel("X (µm)")
    plt.ylabel("Z (µm)")
    ax.set_title("Morphology of simulated cells")
    plt.show()


def plot_potentials(potentials, dt):
    """Plot the potentials at the electrodes"""
    fig, ax = plt.subplots()
    # change the colormap to viridis
    cmap = plt.get_cmap('viridis')
    # plot the potentials at the electrodes with different colors
    t = np.arange(potentials.shape[1]) * dt
    for i in range(potentials.shape[0]):
        ax.plot(t, potentials[i], label=f"Electrode {i+1}", color=cmap(i / potentials.shape[0]))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (µV)")
    ax.set_yscale('symlog')
    ax.set_title("Potentials at electrodes")
    plt.legend()
    plt.show()


def plot_last_three_electrodes(potentials, dt):
    """plot the potentials of the last three electrodes"""
    fig, ax = plt.subplots()
    t = np.arange(potentials.shape[1]) * dt
    ax.plot(t, potentials[-3], label=f"Electrode {len(potentials)-2}")
    ax.plot(t, potentials[-2], label=f"Electrode {len(potentials)-1}")
    ax.plot(t, potentials[-1], label=f"Electrode {len(potentials)}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (µV)")
    ax.set_yscale('symlog')
    ax.set_title("Potentials at last three electrodes")
    plt.legend()
    plt.show()


def plot_first_last_vmem(cell, potentials):
    """plot membrane potential at idx 0 and at last idx"""
    fig, ax = plt.subplots()
    ax.plot(cell.tvec, cell.vmem[0], label="First idx")
    ax.plot(cell.tvec, cell.vmem[-1], label="Last idx")
    ax.plot(cell.tvec, potentials[0][:len(cell.tvec)], label="Electrode 1")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title("Membrane potential at idx 0 and last idx")
    ax.legend()
    plt.show()


def EEGlike_plot():
    """ Run a simulation and plot the 4 EEGlike electrode to check validity of the
        Foursphere approximations"""
    cell = hay_soma(tstop, dt, spike_times=[5], make_passive=True)
    elecs = elec_setup(brainsurfacelevel=brainsurfacelevel, eeglike=eeglike,
                       foursphere=foursphere, method=method)
    elecs.append(LFPy.RecExtElectrode(cell=cell, x=0, y=0, z=brainsurfacelevel + 11990,
                                      method=method, sigma=0.02))

    potentials, morphs = simulate(cell, n_cells, jitter, elecs, somapopheight,
                                  brainsurfacelevel=brainsurfacelevel,
                                  celldensity=celldensity)

    # Find start of spike
    start = np.where(potentials[-2] > 0)[0][0]
    # Find end of spike
    end = np.where(potentials[-2] > 0)[0][-1]

    # trim the potentials to the spike
    potentials = potentials[:, start-10:end+10]
    # convert the potentials to nV
    potentials = potentials * 1e3

    # set viridis color map
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(potentials.shape[1]) * dt
    # plot the potentials at the electrodes with different colors
    ax.plot(t, potentials[-4], label="RecExt 0.3 S/m", color=cmap(0.2))
    ax.plot(t, potentials[-3], label="RecExt 1 S/m", color=cmap(0.4))
    ax.plot(t, potentials[-1], label="RecExt 0.02 S/m", color=cmap(0.6))
    ax.plot(t, potentials[-2], label="Four-sphere", color=cmap(0.8), linestyle='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Potential (nV)")
    ax.set_yscale('symlog')
    ax.set_title("Potentials at EEGlike Electrodes")
    max_pot = round(np.max(potentials[-1]))
    min_pot = round(np.min(potentials[-1]))
    # create ticks for the y axis
    ticks = np.linspace(min_pot, max_pot, 5)
    ax.set_yticks(ticks)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    potentials, morps, paramdict = load_simulation('10000 x hallermann_axon simulation 2025-05-12 15-50-34.npz')
    paramdict = paramdict.item()
    dt = paramdict['dt']
    print(paramdict['axon_type'])
    plot_potentials(potentials=potentials, dt=dt)   
 