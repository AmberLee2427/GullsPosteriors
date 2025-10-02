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
        self.gamma = 0.36  # Default limb darkening, will be overridden from .prm file
        self.model_derivatives = None
        self._data_path = None
        self._config_file = None
        self._config = None
        self.vbm_metadata = {'version': 'unknown', 'rel_tol': 1e-4, 'timeout': 300, 'failures': {}}
        self.vbm_rel_tol = 1e-4
        self.vbm_timeout = 300
        self.config_file = None
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
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not parse SIMULATION_ZERO_TIME from {prm_path}")
                            elif line.startswith('LD_GAMMA='):
                                try:
                                    self.gamma = float(line.split('=')[1].strip())
                                    print(f"Loaded limb darkening gamma from {prm_path}: {self.gamma}")
                                except (ValueError, IndexError):
                                    print(f"Warning: Could not parse LD_GAMMA from {prm_path}")
                    return
        
        print("No SIMULATION_ZERO_TIME loaded from .prm file.")



    def _ensure_vbm_metadata(self):
        """Ensure VBMicrolensing metadata is tracked in the local config."""
        updated = False
        if self._config is None:
            self._config = {}
        vbm_meta = self._config.get('vbm_metadata')
        if not isinstance(vbm_meta, dict):
            vbm_meta = {}
            self._config['vbm_metadata'] = vbm_meta
            updated = True

        if 'version' not in vbm_meta:
            try:
                import VBMicrolensing  # pylint: disable=import-error
                version = getattr(VBMicrolensing, '__version__', getattr(VBMicrolensing, 'VERSION', 'unknown'))
            except ImportError:
                version = 'unavailable'
            vbm_meta['version'] = version
            updated = True

        if 'rel_tol' not in vbm_meta:
            vbm_meta['rel_tol'] = 1e-4
            updated = True

        if 'timeout' not in vbm_meta:
            vbm_meta['timeout'] = 300
            updated = True

        if 'failures' not in vbm_meta:
            vbm_meta['failures'] = {}
            updated = True

        if updated:
            self._save_config()

        self.vbm_metadata = vbm_meta
        self.vbm_rel_tol = vbm_meta.get('rel_tol', 1e-4)
        self.vbm_timeout = vbm_meta.get('timeout', 300)
        self.config_file = self._config_file

    def _initialize_directory(self, path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Data directory '{path}' does not exist or is not a directory.")

        files = sorted(os.listdir(path))
        self.model_derivatives = None
        self._data_path = path
        self._load_config(path)
        self._ensure_vbm_metadata()

        normalized_path = path if path.endswith(os.sep) else path + os.sep
        run_list_file_path = self._ensure_run_tracking_files(normalized_path)
        master_file = self._resolve_master_file(normalized_path, files)

        return normalized_path, files, master_file, run_list_file_path


    def _ensure_run_tracking_files(self, path):
        run_list_file_path = os.path.join(path, "emcee_run_list.txt")
        if not os.path.exists(run_list_file_path):
            open(run_list_file_path, 'w').close()

        complete_file_path = os.path.join(path, "emcee_complete.txt")
        if not os.path.exists(complete_file_path):
            open(complete_file_path, 'w').close()

        return run_list_file_path


    def _resolve_master_file(self, path, files):
        master_file = None
        if self._config and self._config.get('master_file') and os.path.exists(self._config['master_file']):
            master_file = self._config['master_file']
        else:
            for f_name in files:
                if f_name.endswith(('.csv', '.out', '.out.csv', '.out.hdf5', 'outh5')):
                    candidate = os.path.join(path, f_name)
                    print(f"\nFound master file: {candidate}")
                    response = input("Use this file as master file? (y/n): ")
                    if response.lower() != 'y':
                        print("Skipping this master file")
                        continue

                    self._config['master_file'] = candidate
                    self._save_config()
                    master_file = candidate
                    break
            if master_file is None:
                raise FileNotFoundError("No master file found or selected in the specified path. Cannot proceed.")

        return master_file


    def _read_run_list(self, run_list_file_path):
        entries = []
        if os.path.exists(run_list_file_path) and os.path.getsize(run_list_file_path) > 0:
            with open(run_list_file_path, 'r') as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        entries.append(stripped)
        if entries:
            return np.array(entries, dtype=str)
        return np.array([], dtype=str)


    def _write_run_list(self, run_list_file_path, entries):
        with open(run_list_file_path, 'w') as handle:
            for entry in entries:
                handle.write(f"{entry}\n")


    def _append_to_run_list(self, run_list_file_path, current_entries, entry):
        if isinstance(current_entries, np.ndarray):
            entries_list = current_entries.tolist()
        else:
            entries_list = list(current_entries)

        if entry in entries_list:
            return np.array(entries_list, dtype=str)

        entries_list.append(entry)
        self._write_run_list(run_list_file_path, entries_list)
        return np.array(entries_list, dtype=str)


    def _load_event_from_lcfile(self, path, master_file, lc_filename):
        data_file = os.path.join(path, lc_filename)
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Light curve file '{lc_filename}' not found in '{path}'.")

        data = self.load_data(data_file)

        lc_file_name = lc_filename.split(".")[0]
        event_identifiers = lc_file_name.split("_")
        if len(event_identifiers) < 3:
            raise ValueError(f"Unrecognized light curve file naming convention: '{lc_filename}'.")

        event_id = event_identifiers[-1]
        sub_run = event_identifiers[-3]
        field = event_identifiers[-2]

        event_name = f"{field}_{sub_run}_{event_id}"

        obs0_data = data[0].copy()
        simt = obs0_data[7]
        bjd = obs0_data[0]

        truths = self.get_params(
            master_file, event_id, sub_run, field, simt, bjd
        )

        if (lc_filename != truths["lcname"]):
            print(f"WARNING: Light curve file name mismatch for event {event_name}:")
            print(f"  File: {lc_filename}")
            print(f"  Truths 'lcname': {truths['lcname']}")
            if len(lc_filename) != len(truths["lcname"]):
                print(f"  Length mismatch: {len(lc_filename)} != {len(truths['lcname'])}")
            print("  Proceeding with processing despite mismatch.")
        else:
            print("Data file and true params 'lcname' match.")

        return event_name, truths, data


    def _infer_prefix_from_lc_files(self, lc_files):
        if not lc_files:
            raise ValueError("Cannot infer prefix without light curve files.")

        sample_name = sorted(lc_files)[0]
        base_name = sample_name[:-len('.det.lc')] if sample_name.endswith('.det.lc') else sample_name
        parts = base_name.split('_')
        if len(parts) < 4:
            raise ValueError(f"Unrecognized light curve naming convention: {sample_name}")
        prefix = '_'.join(parts[:-3])
        if not prefix:
            raise ValueError(f"Could not determine prefix from light curve name: {sample_name}")

        if self._config is not None and not self._config.get('prefix'):
            self._config['prefix'] = prefix
            self._save_config()

        return prefix


    def _resolve_lc_candidate(self, identifier, lc_files):
        lc_files_set = set(lc_files)

        def _parse_parts(name):
            base = name[:-len('.det.lc')] if name.endswith('.det.lc') else name.split('.')[0]
            parts = base.split('_')
            if len(parts) < 4:
                raise ValueError(f"Unrecognized light curve naming convention: {name}")
            prefix = '_'.join(parts[:-3])
            return prefix, int(parts[-3]), int(parts[-2]), int(parts[-1])

        if isinstance(identifier, (tuple, list)) and len(identifier) >= 3:
            event_id, sub_run, field = [int(x) for x in identifier[:3]]
            prefix = self._config.get('prefix') if self._config else None
            if not prefix:
                prefix = self._infer_prefix_from_lc_files(lc_files)
            candidate = f"{prefix}_{sub_run}_{field}_{event_id}.det.lc"
            if candidate in lc_files_set:
                return candidate

            for fname in lc_files:
                try:
                    _, sub, fld, eid = _parse_parts(fname)
                except ValueError:
                    continue
                if (sub, fld, eid) == (sub_run, field, event_id):
                    return fname

            raise FileNotFoundError(
                f"No light curve file found for identifiers {identifier}."
            )

        if isinstance(identifier, (int, float)):
            identifier = str(int(identifier))

        if isinstance(identifier, str):
            if identifier.endswith('.det.lc'):
                if identifier in lc_files_set:
                    return identifier
                raise FileNotFoundError(
                    f"Requested light curve '{identifier}' not found in directory."
                )

            matches = [f for f in lc_files if identifier in f]
            if not matches:
                raise FileNotFoundError(
                    f"No light curve found matching identifier '{identifier}'."
                )

            if len(matches) == 1:
                return matches[0]

            suffix_matches = [f for f in matches if f.split('.')[0].endswith(identifier)]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

            raise ValueError(
                f"Identifier '{identifier}' matched multiple light curves: {matches}. "
                "Provide a more specific name (e.g., full .det.lc filename)."
            )

        raise ValueError(f"Unsupported identifier type for light curve selection: {type(identifier)}")



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
        event_name, truths, data = None, None, None

        normalized_path, files, master_file, run_list_file_path = self._initialize_directory(path)

        lc_files_candidates = [f for f in files if "det.lc" in f]
        if not lc_files_candidates:
            raise FileNotFoundError(
                f"No light curve files (.det.lc) found in the directory: '{normalized_path}'. Cannot proceed."
            )

        if sort != "alphanumeric":
            raise ValueError("Only 'alphanumeric' sorting is currently supported.")

        current_run_list = self._read_run_list(run_list_file_path)
        found_event_to_process = False

        for f_lc_candidate in sorted(lc_files_candidates):
            print(f"DEBUG: Current run_list: {current_run_list}")
            print(f"DEBUG: Candidate file: {f_lc_candidate}")
            print(f"DEBUG: Is candidate in run_list? {f_lc_candidate in current_run_list}")

            if f_lc_candidate in current_run_list:
                print(f"Skipping already processed event: {f_lc_candidate}")
                continue

            print(f"Processing new event: {f_lc_candidate}")
            event_name, truths, data = self._load_event_from_lcfile(normalized_path, master_file, f_lc_candidate)
            current_run_list = self._append_to_run_list(run_list_file_path, current_run_list, f_lc_candidate)
            found_event_to_process = True
            break

        if not found_event_to_process:
            print(
                f"All light curve files in '{normalized_path}' have already been processed or no new ones found."
            )
            return None, None, None

        return event_name, truths, data


    def load_event_by_identifier(self, path, identifier):
        normalized_path, files, master_file, run_list_file_path = self._initialize_directory(path)

        lc_files_candidates = [f for f in files if "det.lc" in f]
        if not lc_files_candidates:
            raise FileNotFoundError(
                f"No light curve files (.det.lc) found in the directory: '{normalized_path}'. Cannot proceed."
            )

        target_file = self._resolve_lc_candidate(identifier, lc_files_candidates)
        print(f"Processing specified event from list: {target_file}")

        # Check if this event has already been processed
        current_run_list = self._read_run_list(run_list_file_path)
        if target_file in current_run_list:
            print(f"Skipping already processed event: {target_file}")
            raise ValueError(f"Event {target_file} has already been processed. Check emcee_run_list.txt")

        event_name, truths, data = self._load_event_from_lcfile(normalized_path, master_file, target_file)

        self._append_to_run_list(run_list_file_path, current_run_list, target_file)

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
            "dTheta17",
            "dTheta18",
            "dTheta19",
        ]

        # First, read the file to see how many columns it actually has
        # Read just the first few lines to determine column count
        with open(data_file, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 20:  # Read a few lines after the header
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

        # Fisher derivatives are in columns 21-29 (0-indexed: 20-28) for 40-column format
        # Try to find dTheta columns first, then fall back to positional extraction
        col_names = [col for col in data.columns if col.startswith("dTheta")]
        
        # If no dTheta columns, try positional extraction for columns 21-29
        if len(col_names) == 0 and len(data.columns) >= 29:
            print("No dTheta columns found, extracting Fisher derivatives from columns 21-29...")
            # Columns 21-29 in 1-indexed = columns 20-28 in 0-indexed
            fisher_col_indices = list(range(20, 29))  # [20, 21, 22, 23, 24, 25, 26, 27, 28]
            
            # Get column names for these positions
            all_col_names = list(data.columns)
            col_names = [all_col_names[i] for i in fisher_col_indices if i < len(all_col_names)]
            print(f"Extracted Fisher columns from positions 21-29: {col_names}")
        
        print(f"Fisher columns: {col_names}")
        if len(col_names) >= 9:
            # Build full derivative matrix including ALL derivative columns (model + flux)
            all_derivatives = data[col_names].to_numpy()
            n_total_deriv_cols = all_derivatives.shape[1]
            
            # Split: first 9 are model params, rest are flux/nuisance
            model_derivs_raw = all_derivatives[:, :9]
            flux_derivs = all_derivatives[:, 9:] if n_total_deriv_cols > 9 else None
            
            # Reorder model derivatives from simulation to internal order
            # Their order: [t0, log10tE, u0, alpha, log10s, log10q, log10rho, piEN, piEE]
            # Our order:   [log10s, log10q, log10rho, u0, alpha, t0, log10tE, piEE, piEN]
            reorder_indices = [4, 5, 6, 2, 3, 0, 1, 8, 7]
            model_derivs_reordered = model_derivs_raw[:, reorder_indices]
            self.model_derivatives = model_derivs_reordered  # Keep for backward compatibility
            
            # Build full derivative matrix: [model_params, flux_params]
            if flux_derivs is not None and flux_derivs.size > 0:
                full_derivatives = np.hstack([model_derivs_reordered, flux_derivs])
                n_flux_params = flux_derivs.shape[1]
                print(f"Including {n_flux_params} flux derivative columns in Fisher calculation")
            else:
                full_derivatives = model_derivs_reordered
                n_flux_params = 0
                print("No flux derivatives found; Fisher will be model-only")
            
            n_model_params = 9
            n_total_params = full_derivatives.shape[1]
            print(f"Fisher matrix: {n_model_params} model + {n_flux_params} flux = {n_total_params} total params")

            # Data weights
            flux_errors = data["measured_relative_flux_error"].values
            weights = 1.0 / (flux_errors**2)
            
            # --- Model-only Fisher before adding flux params ---
            model_weighted = model_derivs_reordered * weights[:, None]
            model_only_fisher = model_derivs_reordered.T @ model_weighted
            try:
                model_only_covariance = np.linalg.inv(model_only_fisher)
                model_only_success = True
            except np.linalg.LinAlgError:
                model_only_covariance = np.linalg.pinv(model_only_fisher, rcond=1e-10)
                model_only_success = False
            model_only_uncertainties = np.sqrt(np.abs(np.diag(model_only_covariance)))

            # --- Full Fisher including flux params ---
            weighted_derivs = full_derivatives * weights[:, None]
            full_fisher = full_derivatives.T @ weighted_derivs
            
            # Ninja-level error handling for Fisher inversion
            fisher_success = False
            try:
                # Check condition number (how close to singular)
                cond_num = np.linalg.cond(full_fisher)
                if cond_num > 1e12:  # Very ill-conditioned
                    print(f"Fisher matrix poorly conditioned (cond={cond_num:.1e}), using pseudo-inverse")
                    # Use SVD-based pseudo-inverse with tuned tolerance
                    full_covariance = np.linalg.pinv(full_fisher, rcond=1e-10)
                else:
                    # Standard inversion
                    full_covariance = np.linalg.inv(full_fisher)
                fisher_success = True
                print(f"Fisher inversion successful (condition number: {cond_num:.1e})")
                
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Fisher inversion failed: {e}. Using fallback zeros.")
                # Graceful fallback: zero covariance matrix (obviously wrong but won't crash)
                full_covariance = np.zeros((n_total_params, n_total_params))
                fisher_success = False
            
            if fisher_success:
                # Partition blocks using local counts
                F_mm = full_fisher[:n_model_params, :n_model_params]
                F_mf = full_fisher[:n_model_params, n_model_params:]
                F_fm = full_fisher[n_model_params:, :n_model_params]
                F_ff = full_fisher[n_model_params:, n_model_params:]

                # Extract marginalized covariance block (block of inverse is already marginalized)
                self.model_covariance = full_covariance[:n_model_params, :n_model_params]
                self.model_parameter_uncertainties = np.sqrt(np.abs(np.diag(self.model_covariance)))

                # Schur complement fisher (F_mm - F_mf F_ff^{-1} F_fm)
                if n_flux_params > 0:
                    try:
                        F_ff_inv = np.linalg.inv(F_ff)
                    except np.linalg.LinAlgError:
                        F_ff_inv = np.linalg.pinv(F_ff, rcond=1e-10)
                    schur_fisher = F_mm - F_mf @ F_ff_inv @ F_fm
                else:
                    schur_fisher = F_mm.copy()

                try:
                    schur_cov = np.linalg.inv(schur_fisher)
                except np.linalg.LinAlgError:
                    schur_cov = np.linalg.pinv(schur_fisher, rcond=1e-10)

                try:
                    denom = np.maximum(1e-30, np.abs(self.model_covariance))
                    schur_diff = np.max(np.abs(schur_cov - self.model_covariance) / denom)
                except Exception:
                    schur_diff = None

                # Store Schur artifacts only as diagnostics
                self.schur_fisher_matrix = schur_fisher
                self.schur_covariance_alt = schur_cov
                self.schur_max_rel_diff = schur_diff
                # Primary fisher_matrix now from full inverse (inverse of marginalized covariance)
                try:
                    self.fisher_matrix = np.linalg.inv(self.model_covariance)
                except np.linalg.LinAlgError:
                    self.fisher_matrix = np.linalg.pinv(self.model_covariance, rcond=1e-10)
                    print("Note: Used pseudo-inverse for model covariance when forming fisher_matrix.")
            else:
                self.model_covariance = model_only_covariance
                self.model_parameter_uncertainties = model_only_uncertainties
                self.fisher_matrix = model_only_fisher
                if not model_only_success:
                    print("Warning: model-only Fisher also required pseudo-inverse; results may be unstable.")

            # Store model-only artifacts regardless
            self.model_only_fisher = model_only_fisher
            self.model_only_covariance = model_only_covariance
            self.model_only_uncertainties = model_only_uncertainties
            self.model_only_inversion_success = model_only_success
            
            # Store additional attributes for debugging
            self.full_fisher_matrix = full_fisher
            self.full_covariance = full_covariance
            self.n_flux_params = n_flux_params
            self.fisher_inversion_success = fisher_success

            # --- DEBUG PRINTS FOR FISHER CALCULATIONS ---
            print("\n--- Fisher Calculation Debug ---")
            print(f"Full Fisher shape: {full_fisher.shape}")
            print(f"Model-only Fisher shape: {self.model_only_fisher.shape}")
            print(f"Model-only inversion success: {self.model_only_inversion_success}")
            if fisher_success and n_flux_params > 0:
                print(f"Schur fisher (diagnostic) shape: {self.schur_fisher_matrix.shape}")
                print(f"Primary fisher (from inverse full covariance) shape: {self.fisher_matrix.shape}")
                print(f"Max rel diff (Schur covariance vs block inverse): {self.schur_max_rel_diff}")
            print(f"Marginalized covariance shape: {self.model_covariance.shape}")
            print(f"Raw model-only uncertainties: {self.model_only_uncertainties}")
            print(f"Marginalized uncertainties: {self.model_parameter_uncertainties}")
            print("--- End Fisher Calculation Debug ---\n")

            # Store additional attributes (after debug so they exist regardless)
            self.full_fisher_matrix = full_fisher
            self.full_covariance = full_covariance
            self.n_flux_params = n_flux_params
            self.fisher_inversion_success = fisher_success

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

        # Try to read gamma from master file if we have the default value
        if 'LDgamma' in truths:
            try:
                gamma_val = float(truths['LDgamma'])
                self.gamma = gamma_val
                try:
                    truths.at['gamma'] = gamma_val
                except Exception:
                    pass
                print(f"Loaded limb darkening gamma from master file: {self.gamma}")
            except (ValueError, TypeError):
                print(f"Warning: Could not parse LDgamma from master file: {truths['LDgamma']}")

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
    
    @staticmethod
    def make_master_csvs_from_hdf5(data_file, output_dir, chunk_size=10000):
        """Create master CSV files from chunks of an HDF5 data file.

        Parameters
        ----------
        data_file : str
            Path to the HDF5 data file.
        output_dir : str
            Directory to save the master CSV files.
        chunk_size : int, optional
            Number of events per CSV file, by default 10000.
        """
        chunks = pd.read_hdf(data_file, chunksize=chunk_size)
        for i, chunk in enumerate(chunks):
            chunk.to_csv(os.path.join(output_dir, f"master_{i}.csv"), index=False)
