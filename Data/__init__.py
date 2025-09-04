"""Data handling utilities for microlensing simulations."""

import numpy as np
import sys
import os
import pandas as pd
import warnings
import json


class Data:
    """Interface for loading light curves and truth parameters.

    The class provides methods to iterate through Data Challenge
    events and to parse the corresponding light curve files and master
    CSV tables.
    """

    def __init__(self):
        """Create an empty container for event data.

        Notes
        -----
        The constructor does not set any attributes or perform I/O.  Side
        effects are produced when calling :meth:`new_event` which writes the
        files ``emcee_run_list.txt`` and ``emcee_complete.txt`` in the
        specified directory.
        """
        self.sim_time0 = None
        self.model_derivatives = None
        self._data_path = None
        self._config_file = None
        self._config = None
        # Initialize _config before calling _load_prm_time_correction without data_dir
        # Use a temporary path if _data_path is not yet set
        temp_data_dir = os.getcwd() if self._data_path is None else self._data_path
        self._load_config(temp_data_dir)
        self._load_prm_time_correction()

    def _load_config(self, data_dir):
        """Load or create config file for this data directory.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the data files.
        """
        self._config_file = os.path.join(data_dir, '.gulls_config.json')
        
        if os.path.exists(self._config_file):
            with open(self._config_file, 'r') as f:
                self._config = json.load(f)
        else:
            self._config = {
                'master_file': None,
                'prm_file': None,
                'prefix': None # Initialize prefix here too
            }

    def _save_config(self):
        """Save current config to file."""
        if self._config_file is not None:
            with open(self._config_file, 'w') as f:
                json.dump(self._config, f, indent=4)

    def _load_prm_time_correction(self, data_dir=None):
        """Load time correction from prm file if it exists.
        
        Looks for a `.prm` file in the current directory and parent directories
        up to 3 levels up. If found, uses `SIMULATION_ZERO_TIME` as the time correction.
        """
        # Determine the directory to start searching from
        if data_dir is None:
            data_dir = self._data_path # Use _data_path if data_dir is not provided
            if data_dir is None: # Fallback if _data_path is also None
                data_dir = os.getcwd()

        # First check if we have a saved prm file path in config and it exists
        if self._config and self._config.get('prm_file') and os.path.exists(self._config['prm_file']):
            prm_path = self._config['prm_file']
            with open(prm_path, 'r') as f:
                for line in f:
                    if line.startswith('SIMULATION_ZERO_TIME='):
                        try:
                            self.sim_time0 = float(line.split('=')[1].strip())
                            print(f"Loaded time correction from {prm_path}: {self.sim_time0}")
                            return
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse SIMULATION_ZERO_TIME from {prm_path}")
        
        # If not found in config or path invalid, look in data_dir and current_dir
        # Loop through potential directories to find .prm file
        search_dirs = [data_dir, os.getcwd()]
        for current_search_dir in search_dirs:
            # Ensure the directory exists before listing its contents
            if not os.path.isdir(current_search_dir):
                continue

            for file_name in os.listdir(current_search_dir):
                if file_name.endswith('.prm'):
                    prm_path = os.path.join(current_search_dir, file_name)
                    print(f"\nFound .prm file in {current_search_dir}: {prm_path}")
                    response = input("Use this file for time correction? (y/n): ")
                    if response.lower() != 'y':
                        print("Skipping this .prm file")
                        continue
                    
                    # Save the path if user confirms
                    self._config['prm_file'] = prm_path
                    self._save_config()
                    
                    with open(prm_path, 'r') as f:
                        for line in f:
                            if line.startswith('SIMULATION_ZERO_TIME='):
                                try:
                                    self.sim_time0 = float(line.split('=')[1].strip())
                                    print(f"Loaded time correction from {prm_path}: {self.sim_time0}")
                                    return
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not parse SIMULATION_ZERO_TIME from {prm_path}")
        
        print("No SIMULATION_ZERO_TIME loaded from .prm file.")


    def new_event(self, path, sort="alphanumeric"):
        r"""Return the next lightcurve and its true parameters.

        Parameters
        ----------
        path : str
            Directory containing the Data Challenge files.  The directory must
            include the master ``*.csv`` file as well as the ``*.det.lc``
            lightcurve files.
        sort : str, optional
            Selection method for the next event.  Only ``'alphanumeric'`` is
            currently supported and will process files in lexicographic order.

        Returns
        -------
        tuple
            ``(event_name, truths, data)`` where ``event_name`` is the event
            identifier, ``truths`` is a :class:`pandas.Series` with the event
            parameters and additional derived values, and ``data`` is the
            dictionary returned by :meth:`load_data`. Returns ``(None, None, None)``
            if no new event is found (graceful exit).

        Raises
        ------
        FileNotFoundError
            If no master file is found or selected.
            If no light curve files (.det.lc) are found in the directory.
        KeyError
            If 'lcname' or 'LCOutput' columns are missing in the master file.
        ValueError
            If master file naming convention is invalid.
        """
        # Initialize return values to None
        event_name, truths, data = None, None, None

        files = os.listdir(path)
        self.model_derivatives = None
        self._data_path = path
        self._load_config(path) # Load config specific to the data path
        files = sorted(files)

        if path[-1] != "/":
            path = path + "/"

        run_list_file_path = path + "emcee_run_list.txt"
        if not os.path.exists(run_list_file_path):
            # Create an empty run list file if it doesn't exist
            np.savetxt(run_list_file_path, np.array([]), fmt="%s")

        if not os.path.exists(
            path + "emcee_complete.txt"
        ):  # if the complete list doesn't exist, create it
            complete_list = np.array([])
            np.savetxt(path + "emcee_complete.txt", complete_list, fmt="%s")

        # Check if we have a saved master file path
        master_file = None # Initialize master_file
        if self._config and self._config.get('master_file') and os.path.exists(self._config['master_file']):
            master_file = self._config['master_file']
        else:
            # Look for master file
            for f_name in files: # Use f_name to avoid conflict with 'file_lc' later
                if f_name.endswith(('.csv', '.out')):
                    master_file = path + f_name
                    print(f"\nFound master file: {master_file}")
                    response = input("Use this file as master file? (y/n): ")
                    if response.lower() != 'y':
                        print("Skipping this master file")
                        master_file = None # Reset if skipped
                        continue
                    
                    # Save the path if user confirms
                    self._config['master_file'] = master_file
                    self._save_config()
                    break
            if master_file is None:
                raise FileNotFoundError("No master file found or selected in the specified path. Cannot proceed.")

        # Filter for light curve files and check if any exist
        lc_files_candidates = [f for f in files if "det.lc" in f]
        if not lc_files_candidates:
            raise FileNotFoundError(f"No light curve files (.det.lc) found in the directory: '{path}'. Cannot proceed.")

        found_event_to_process = False # Flag to indicate if a new event was successfully processed

        if sort == "alphanumeric":
            for f_lc_candidate in sorted(lc_files_candidates): # Iterate only over .det.lc files
                # --- Robustly load run_list ---
                current_run_list = []
                if os.path.exists(run_list_file_path) and os.path.getsize(run_list_file_path) > 0:
                    with open(run_list_file_path, 'r') as f:
                        for line in f:
                            stripped_line = line.strip()
                            if stripped_line: # Only add non-empty lines
                                current_run_list.append(stripped_line)
                current_run_list = np.array(current_run_list, dtype=str) # Ensure it's a NumPy array of strings
                # --- End robust load ---

                # --- Debug prints ---
                print(f"DEBUG: Current run_list: {current_run_list}")
                print(f"DEBUG: Candidate file: {f_lc_candidate}")
                print(f"DEBUG: Is candidate in run_list? {f_lc_candidate in current_run_list}")
                # --- End debug prints ---

                if (f_lc_candidate not in current_run_list): # Check only if it's not in run_list
                    print(f"Processing new event: {f_lc_candidate}")
                    # Add to run_list immediately before processing
                    new_run_list = np.hstack([current_run_list, f_lc_candidate])
                    np.savetxt(run_list_file_path, new_run_list, fmt="%s")

                    lc_file_name = f_lc_candidate.split(".")[0]
                    event_identifiers = lc_file_name.split("_")
                    event_id = event_identifiers[-1]
                    sub_run = event_identifiers[-3]
                    field = event_identifiers[-2]

                    data_file = path + f_lc_candidate

                    data = self.load_data(
                        data_file
                    )  # bjd, flux, flux_err, tshift, ushift

                    event_name = f"{field}_{sub_run}_{event_id}"

                    obs0_data = data[0].copy()
                    simt = obs0_data[7]
                    bjd = obs0_data[0]

                    truths = self.get_params(
                        master_file, event_id, sub_run, field, simt, bjd
                    )
                    
                    # --- Handle lcname mismatch: LOG and PROCEED ---
                    if (f_lc_candidate != truths["lcname"]):
                        print(f"WARNING: Light curve file name mismatch for event {event_name}:")
                        print(f"  File: {f_lc_candidate}")
                        print(f"  Truths 'lcname': {truths['lcname']}")
                        if len(f_lc_candidate) != len(truths["lcname"]):
                            print(f"  Length mismatch: {len(f_lc_candidate)} != {len(truths['lcname'])}")
                        print("  Proceeding with processing despite mismatch.")
                    else:
                        print("Data file and true params 'lcname' match.")
                    
                    # If we reached here, it means we found a suitable f_lc_candidate
                    # and successfully loaded its data and truths (even with mismatch).
                    found_event_to_process = True
                    break # Exit the for loop, we found our event.
                # If f_lc_candidate is already in run_list, continue to next file
                else:
                    print(f"Skipping already processed event: {f_lc_candidate}")
                    continue # Explicitly continue to next iteration if already run

        # After the loop, if no new event was found to process, return None, None, None
        if not found_event_to_process:
            print(f"All light curve files in '{path}' have already been processed or no new ones found.")
            return None, None, None # Graceful exit

        # If a new event was found, return its details
        return event_name, truths, data

    def load_data(self, data_file):
        r"""Load a Data Challenge lightcurve file.
        The data array contains both measured and true values:
        - The measured values (indices 1 and 2) are used for actual fitting
        - The true values (indices 5 and 6) are used only for diagnostics and plotting
        - The naming convention might be confusing, but the code correctly uses
          the measured values and their uncertainties for the chi-square calculations

        Parameters
        ----------
        data_file : str
            Path to the lightcurve file to load.

        Returns
        -------
        dict
            Keys are observatory codes and values are ``(N, 8)`` arrays for
            that observatory.

        Dictionary mapping observatory codes to numpy arrays of shape (8, n_points)
        containing the following data for each point:
            [0] = "BJD" (time)
            [1] = "measured_relative_flux" (the actual observed flux values used for fitting)
            [2] = "measured_relative_flux_error" (the uncertainties used for fitting)
            [3:5] = "parallax_shift_t", "parallax_shift_u" (parallax shift components)
            [5] = "true_relative_flux" (the true underlying flux, used only for diagnostics)
            [6] = "true_relative_flux_error" (the true uncertainties, used only for diagnostics)
            [7] = "Simulation_time"

        Notes
        -----
        The lightcurve columns are:
            [0] "Simulation_time"
            [1] "measured_relative_flux"
            [2] "measured_relative_flux_error"
            [3] "true_relative_flux"
            [4] "true_relative_flux_error"
            [5] "observatory_code"
            [6] "saturation_flag"
            [7] "best_single_lens_fit"
            [8] "parallax_shift_t"
            [9] "parallax_shift_u"
            [10] "BJD"
            [11] "source_x"
            [12] "source_y"
            [13] "lens1_x"
            [14] "lens1_y"
            [15] "lens2_x"
            [16] "lens2_y"
            [17] "X"                # observatory position (not in all datasets)
            [18] "Y"
            [19] "Z"
            [20] "dTheta1"          # Fisher stuff (not in all datasets)
            [21] "dTheta2"
            [22] "dTheta3"
            [23] ...

        Magnitudes can be computed using:

        .. math::
            m = m_{source} + 2.5 \log f_s - 2.5 \log{F}

        where :math:`F=fs*\mu + (1-fs)` is the relative flux (in the file),
        :math:`\mu` is the magnification, and

        .. math::
            \sigma_m = 2.5/\ln{10} \sigma_F/F.

        These are listed in the header information in lines ``#fs`` and
        ``#Obssrcmag`` with order matching the observatory code order.
        The observatory codes correspond to 0=W146, 1=Z087, 2=K213

        Bugs/issues/caveats:
        The output file columns list a limb darkening parameter of Gamma=0, it
        is actually Gamma=0.36 (in all filters).
        The orbit for the Z087 observatory appears to be different to the W146
        and K213 observatories.
        Dev is working on producing the ephemerides, but for single observatory
        parallax, using interpolated versions of the ones available for the
        data challenge will probably be accurate enough, or an Earth ephemeris
        with
        the semimajor axis (but not period) increased by 0.01 AU.
        Lenses with masses smaller than the isochrone grid limits (I believe
        0.1 MSun) will have filler values for magnitudes and lens stellar
        properties.
        There may be some spurious detections in the list where the single lens
        fit failed. Please let dev know if you find any of these events so that
        we can improve the single lens fitter."""

        # Define the expected column names in order
        expected_columns = [
            "Simulation_time",
            "measured_relative_flux",
            "measured_relative_flux_error",
            "true_relative_flux",
            "true_relative_flux_error",
            "observatory_code",
            "saturation_flag",
            "best_single_lens_fit",
            "parallax_shift_t",
            "parallax_shift_u",
            "BJD",
            "source_x",
            "source_y",
            "lens1_x",
            "lens1_y",
            "lens2_x",
            "lens2_y",
            "X",
            "Y",
            "Z",
            "dTheta1",
            "dTheta2",
            "dTheta3",
            "dTheta4",
            "dTheta5",
            "dTheta6",
            "dTheta7",
            "dTheta8",
            "dTheta9",
            "dTheta10",
            "dTheta11",
            "dTheta12",
            "dTheta13",
            "dTheta14",
            "dTheta15",
            "dTheta16",
            "dTheta17"
        ]

        # First, read the file to see how many columns it actually has
        # Read just the first few lines to determine column count
        with open(data_file, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 15:  # Read a few lines after the header
                    break
                lines.append(line.strip())
        
        # Find the first data line (not starting with #)
        data_line = None
        skip_rows = 1
        for line in lines:
            if not line.startswith('#'):
                data_line = line
                break
            skip_rows += 1

        if data_line is None:
            raise ValueError(f"Could not find data line in {data_file}")
        
        # Count the actual number of columns in the data
        actual_columns = len(data_line.split())
        print(f"Detected {actual_columns} columns in {data_file}")
        
        # Use only the columns that exist in the file
        header = expected_columns[:actual_columns]
        print(f"Header: {header}")
        
        # If we have more columns than expected, add generic names
        if actual_columns > len(expected_columns):
            for i in range(len(expected_columns), actual_columns):
                header.append(f"extra_col_{i}")
            print(f"Warning: File has {actual_columns} columns, expected up to {len(expected_columns)}")

        data = pd.read_csv(
            data_file, sep=r"\s+", skiprows=skip_rows, names=header
        )  # delim_whitespace=True is the same as sep=r'\s+', but older.
        # The 'r' in sep=r'\s+' means raw string, which is not necessary.
        # Otherwise you get annoying warnings.

        print(f"Data columns: {data.columns}")

        # Try to load prm file again if we don't have sim_time0
        if self.sim_time0 is None:
            self._load_prm_time_correction(data_dir=os.path.dirname(data_file)) # Pass the directory of the data file
            
        # Only calculate from data if we still don't have sim_time0
        if self.sim_time0 is None:
            print("No prm file found, calculating time correction from data...")
            self.sim_time0 = np.sum(data["BJD"] - data["Simulation_time"]) / len(
                data["BJD"]
            )
            print(f"Calculated time correction: {self.sim_time0}")

        # Select only the columns we need for processing
        required_columns = [
            "BJD",
            "measured_relative_flux",
            "measured_relative_flux_error",
            "parallax_shift_t",
            "parallax_shift_u",
            "observatory_code",
            "true_relative_flux",
            "true_relative_flux_error",
            "Simulation_time",
        ]
        
        # Check which required columns are available
        available_columns = [col for col in required_columns if col in data.columns]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # If crucial columns are missing, raise an error or handle gracefully
            if "BJD" in missing_columns or "measured_relative_flux" in missing_columns or "measured_relative_flux_error" in missing_columns:
                raise ValueError(f"Essential columns missing in {data_file}: {missing_columns}")
            print(f"Warning: Non-essential columns missing in {data_file}: {missing_columns}")
            data = data[available_columns] # Proceed with available columns

        # cov is any column name that starts with "dTheta"
        col_names = [col for col in data.columns if col.startswith("dTheta")]
        print(f"Fisher columns: {col_names}")
        if len(col_names) > 0:  
            self.model_derivatives = data[col_names].to_numpy()

            # REORDER DERIVATIVES TO MATCH OUR PARAMETER ORDER
            # Simulation team's order: [t0, tE, u0, alpha, s, q, rs, piEN, piEE, flux_params...]
            # Our parameter order:     [s, q, rho, u0, alpha, t0, tE, piEE, piEN]
            # 
            # Mapping from their indices to our indices:
            # Their: [0=t0, 1=tE, 2=u0, 3=alpha, 4=s, 5=q, 6=rs, 7=piEN, 8=piEE]
            # Ours:  [0=s,  1=q,  2=rho, 3=u0,   4=alpha, 5=t0, 6=tE, 7=piEE, 8=piEN]
            # 
            # So reorder mapping: [4, 5, 6, 2, 3, 0, 1, 8, 7] (their indices → our order)
            
            # Determine how many observatory groups we have and extract only the first 9 params per group
            n_cols = self.model_derivatives.shape[1]
            
            # Assume we have multiple observatory groups, each with the same parameter structure
            # Count parameters per group by finding flux parameters (assume they start after the 9 main params)
            # For now, let's assume we want the first 9 parameters from the first observatory group
            
            if n_cols >= 9:  # Make sure we have at least 9 parameters
                # Extract first 9 derivatives (from first observatory group)
                derivatives_first_group = self.model_derivatives[:, :9]
                
                # Reorder from simulation order to our order
                # Their order: [t0, tE, u0, alpha, s, q, rs, piEN, piEE]
                # Our order:   [s, q, rho, u0, alpha, t0, tE, piEE, piEN]
                reorder_indices = [4, 5, 6, 2, 3, 0, 1, 8, 7]
                
                # Apply reordering
                self.model_derivatives = derivatives_first_group[:, reorder_indices]
                
                print(f"Reordered derivatives from simulation order to our parameter order")
                print(f"Original shape: {derivatives_first_group.shape}")
                print(f"Reordered shape: {self.model_derivatives.shape}")
                print(f"Parameter order is now: [s, q, rho, u0, alpha, t0, tE, piEE, piEN]")
            else:
                print(f"Warning: Only {n_cols} derivative columns found, expected at least 9")

            # Form the data covariance matrix (diagonal matrix of flux uncertainties)
            # We need to get the flux errors for all data points
            flux_errors = data["measured_relative_flux_error"].values
            self.data_covariance = np.diag(flux_errors**2)  # C = diag(σ²)
            
            # Calculate C^-1 (inverse of diagonal matrix is just 1/diagonal elements)
            self.data_covariance_inv = np.diag(1.0 / flux_errors**2)  # C^-1 = diag(1/σ²)

            # Calculate the Fisher matrix: F = ∇ᵀC⁻¹∇
            # where ∇ is the matrix of model derivatives
            n_params = self.model_derivatives.shape[1]  # Use reordered derivatives
            n_data = len(flux_errors)
            self.fisher_matrix = np.zeros((n_params, n_params))
            
            # More efficient calculation using matrix operations
            # F_ij = Σ_k (∂f_k/∂θ_i) * (1/σ_k²) * (∂f_k/∂θ_j)
            for i in range(n_params):
                for j in range(i, n_params):
                    # Sum over all data points
                    fisher_element = np.sum(
                        self.model_derivatives[:, i] * (1.0 / flux_errors**2) * self.model_derivatives[:, j]
                    )
                    self.fisher_matrix[i, j] = fisher_element
                    self.fisher_matrix[j, i] = fisher_element  # Symmetric matrix

            # Calculate the inverse of the Fisher matrix
            self.model_covariance = np.linalg.inv(self.fisher_matrix)
            # Calculate 1-sigma Fisher uncertainties for each parameter
            self.model_parameter_uncertainties = np.sqrt(np.diag(self.model_covariance))

            # --- DEBUG PRINTS FOR FISHER CALCULATIONS ---
            print("\n--- Fisher Calculation Debug ---")
            print(f"Fisher Matrix shape: {self.fisher_matrix.shape}")
            print(f"Fisher Matrix (first 3x3): \n{self.fisher_matrix[:min(3, n_params),:min(3, n_params)]}") # Adjusted for smaller n_params
            print(f"Model Covariance shape: {self.model_covariance.shape}")
            print(f"Model Covariance (first 3x3): \n{self.model_covariance[:min(3, n_params),:min(3, n_params)]}") # Adjusted for smaller n_params
            print(f"Model Parameter Uncertainties (1-sigma): \n{self.model_parameter_uncertainties}")
            print("--- End Fisher Calculation Debug ---\n")

        data = data[required_columns]

        data_dict = {}
        for code in data["observatory_code"].unique():
            # Select columns that are available for this observatory
            available_obs_columns = [col for col in available_columns if col != "observatory_code"]
            data_obs = data[data["observatory_code"] == code][available_obs_columns].reset_index(drop=True)
            data_dict[code] = data_obs.to_numpy().T

        return data_dict

    def _read_master_file(self, master_file):
        """Read master file, supporting both CSV and HDF5 formats."""
        if master_file.endswith(('.hdf5', '.h5')):
            print(f"Reading HDF5 master file: {master_file}")
            
            # Try pandas first (works well for properly formatted HDF5)
            try:
                master = pd.read_hdf(master_file)
                print(f"Successfully read HDF5 file with pandas, shape: {master.shape}")
                print(f"Columns include: {master.columns[:10].tolist()}...")
                return master
            except Exception as pandas_error:
                print(f"pandas read_hdf failed: {pandas_error}")
                
            # Fall back to manual h5py reading if pandas fails
            try:
                import h5py
                with h5py.File(master_file, 'r') as hdf:
                    print(f"HDF5 file keys: {list(hdf.keys())}")
                    
                    # Use the first available dataset
                    data_key = list(hdf.keys())[0]
                    print(f"Using dataset key: {data_key}")
                    
                    dataset = hdf[data_key]
                    
                    # Convert to pandas DataFrame
                    if hasattr(dataset, 'dtype') and dataset.dtype.names:
                        # Structured array
                        data_dict = {name: dataset[name][:] for name in dataset.dtype.names}
                        master = pd.DataFrame(data_dict)
                    else:
                        raise ValueError(f"HDF5 dataset format not supported: {type(dataset)}")
                    
                    print(f"Successfully read HDF5 file manually, shape: {master.shape}")
                    return master
                    
            except ImportError:
                raise ImportError("h5py package is required to read HDF5 files. Install with: pip install h5py")
            except Exception as e:
                raise ValueError(f"Failed to read HDF5 file {master_file}: {e}")
        else:
            # Read CSV/text file
            return pd.read_csv(master_file, header=0, sep=r'[,    \s]+', engine='python')

    def get_params(
        self, master_file, event_id, sub_run, field, epoch=None, bjd=None
    ):
        r"""Return the true parameters for an event.

        Parameters
        ----------
        master_file : str
            Path to the master ``*.csv`` file containing event information.
        event_id : int or str
            Identifier of the event in ``master_file``.
        sub_run : int or str
            Data Challenge sub-run number.
        field : int or str
            Field identifier within the Data Challenge.
        epoch : array-like, optional
            Simulation epochs used to convert times to BJD.
        bjd : array-like, optional
            Barycentric Julian Date corresponding to ``epoch``.

        Returns
        -------
        pandas.Series
            Series of event parameters with additional keys ``params`` and
            ``tcroin`` that have been converted to BJD.

        Notes
        -----
        Several quantities are converted for convenience:

        * ``alpha``, ``Planet_inclination`` and ``Planet_orbphase`` are
          converted from degrees to radians.
        * ``t0lens1`` and ``tcroin`` are converted from simulation time to BJD
          either by interpolation of ``epoch``/``bjd`` or by using the
          offset determined from the lightcurve.
        * ``Planet_period`` is converted from years to days.
        """
        event_id = int(event_id)
        sub_run = int(sub_run)
        field = int(field)

        # Read master file using the new helper method
        master = self._read_master_file(master_file)
        # print(master.head())

        truths = master[
            (master["EventID"] == int(event_id))
            & (master["SubRun"] == int(sub_run))
            & (master["Field"] == int(field))
        ].iloc[0]

        # print(self.sim_time0)

        s = truths["Planet_s"]
        q = truths["Planet_q"]
        rho = truths["rho"]
        u0 = truths["u0lens1"]  # croin
        alpha = truths["alpha"] * np.pi / 180  # convert to radians
        if epoch is not None and bjd is not None:
            t0_sim = truths["t0lens1"]
            t0_bjd = np.interp(t0_sim, epoch, bjd)
            t0 = t0_bjd
            tc_sim = truths["tcroin"]
            tc_bjd = np.interp(tc_sim, epoch, bjd)
            tcroin = tc_bjd
        else:
            t0 = truths["t0lens1"] + self.sim_time0  # convert to BJD
            tcroin = truths["tcroin"] + self.sim_time0  # convert to BJDs
        truths["t0lens1"] = t0
        tE = truths["tE_ref"]
        piEE = truths["piEE"]
        piEN = truths["piEN"]
        i = truths["Planet_inclination"] * np.pi / 180  # convert to radians
        phase = (
            truths["Planet_orbphase"] * np.pi / 180
        )  # convert to radians # centre on tcroin
        period = truths["Planet_period"] * 365.25  # convert to days
        # phase_change = truths['tcroin'] / period
        # phase = phase + phase_change  # centre on t0
        # phase = phase % (2.0*np.pi)  # make sure it's between 0 and 2pi
        truths["params"] = [
            s,
            q,
            rho,
            u0,
            alpha,
            t0,
            tE,
            piEE,
            piEN,
            i,
            phase,
            period,
        ]

        truths["tcroin"] = tcroin

        # After reading the master file and extracting truths
        # Use 'lcname' or 'LCOutput' as available
        lcname_col = None
        if 'lcname' in truths:
            lcname_col = 'lcname'
        elif 'LCOutput' in truths:
            lcname_col = 'LCOutput'
        else:
            # If neither is found, it's a critical error for matching
            raise KeyError("Neither 'lcname' nor 'LCOutput' found in master file for event.")

        # Always set 'lcname' for downstream code, using the found column
        # CRITICAL FIX: Revert to original logic for lcname construction based on unique values
        def is_number(val):
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False

        if is_number(truths[lcname_col]) and float(truths[lcname_col]) in [0, 1]:
            # Check if prefix is already in config file
            if self._config.get('prefix') is not None: 
                prefix = self._config['prefix']
            else:
                # Determine the naming convention
                # Check if SubRun column has 1 unique value
                if len(master['SubRun'].unique()) == 1:
                    suffixA = "_" + str(int(master['SubRun'].unique()[0])) # Cast to int
                else:
                    suffixA = ""
                # Check if Field column has 1 unique value
                if len(master['Field'].unique()) == 1:
                    suffixB = "_" + str(int(master['Field'].unique()[0])) # Cast to int
                else:
                    suffixB = ""

                # Strip file extension from master_file
                master_file_name = os.path.splitext(os.path.basename(master_file))[0]

                # Determine the correct suffix order
                if master_file_name.endswith(suffixA + suffixB):
                    suffix = suffixA + suffixB
                elif master_file_name.endswith(suffixB + suffixA):
                    suffix = suffixB + suffixA
                elif master_file_name.endswith(suffixA):
                    suffix = suffixA
                elif master_file_name.endswith(suffixB):
                    suffix = suffixB
                else:
                    # If no clear suffix pattern matches, default to empty suffix
                    # and let the prefix be the full master file name
                    suffix = ""
                    warnings.warn(f"Could not determine clear suffix pattern for master file: {master_file_name}. Using full name as prefix.")
                
                # get master_file prefix 
                prefix = master_file_name[: -len(suffix)] if suffix else master_file_name

                # save naming convention to config file
                self._config['prefix'] =  prefix
                self._save_config()

            # Construct new lcname: <prefix>_<SubRun>_<Field>_<EventID>.det.lc
            truths['lcname'] = f"{prefix}_{int(truths['SubRun'])}_{int(truths['Field'])}_{int(truths['EventID'])}.det.lc"
        else:
            # If lcname_col is not a number, assume it's already the correct filename string
            truths['lcname'] = truths[lcname_col]
        
        return truths
