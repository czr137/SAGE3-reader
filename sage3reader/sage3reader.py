import numpy as np
from struct import unpack
import xarray as xr


def l2binary_to_dataset(file) -> xr.Dataset:
    """
    Read the Level 2 Solar Event Species Profiles for a version 5 SAGE III or SAGE III ISS binary file.
    https://eosweb.larc.nasa.gov/sites/default/files/project/sage3/guide/Data_Product_User_Guide.pdf
    """

    # Read all the data into memory
    with open(file, 'rb') as f:
        # Read the File Header
        (profile_id, yyyyddd, instrument_time, fill_value_int, fill_value_float, mission_id) = \
            unpack('>iififi', f.read(6 * 4))

        # Read the Version Tracking data
        (L0DO_ver, L0_ver, software_ver, dataproduct_ver, spectroscopy_ver, gram95_ver, met_ver) = \
            unpack('>fffffff', f.read(7 * 4))

        # Read the File Description
        (altitude_spacing, num_bins, num_aer_wavelengths, num_ground_tracks, num_aer_bins) = \
            unpack('>fiiii', f.read(5 * 4))

        # Read the Event Type data
        (event_type_spacecraft, event_type_earth, beta_angle, event_status_flags) = unpack('>iifi', f.read(4 * 4))

        # Read Data Capture Start Information
        (start_date, start_time, start_latitude, start_longitude, start_altitude) = unpack('>iifff', f.read(5 * 4))

        # Read Data Capture End Information
        (end_date, end_time, end_latitude, end_longitude, end_altitude) = unpack('>iifff', f.read(5 * 4))

        # Read Ground Track Information
        gt_date = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)))
        gt_time = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)))
        gt_latitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)))
        gt_longitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)))
        gt_ray_dir = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)))

        # Read Profile Altitude Levels data
        homogeneity = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))
        altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        potential_altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))

        # Read the Inupt T/P for Retrievals
        input_temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        input_temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        input_pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        input_pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        input_tp_source_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the Derived Tropopause data
        (temperature_tropopause, altitude_tropopause) = unpack('>ff', f.read(2 * 4))

        # Read the Composite Ozone data
        o3_composite = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_composite_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_composite_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_composite_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_composite_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the Mesospheric Ozone data
        o3_mesospheric = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mesospheric_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mesospheric_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mesospheric_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mesospheric_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the MLR Ozone data
        o3_mlr = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mlr_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mlr_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mlr_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_mlr_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the Ozone Least Squares data
        o3 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        o3_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read Water Vapor data
        water_vapor = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        water_vapor_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        water_vapor_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the NO2 data
        no2 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        no2_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        no2_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        no2_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        no2_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the Retrieved T/P data
        temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)))
        tp_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)))

        # Read the Aerosol Information
        aerosol_wavelengths = np.array(unpack('>' + 'f' * num_aer_wavelengths, 
                                              f.read(num_aer_wavelengths * 4)))
        aerosol_half_bandwidths = np.array(unpack('>' + 'f' * num_aer_wavelengths, 
                                                  f.read(num_aer_wavelengths * 4)))
        stratospheric_optical_depth = np.array(unpack('>' + 'f' * num_aer_wavelengths, 
                                                      f.read(num_aer_wavelengths * 4)))
        stratospheric_optical_depth_error = np.array(unpack('>' + 'f' * num_aer_wavelengths, 
                                                            f.read(num_aer_wavelengths * 4)))
        stratospheric_optical_depth_qa_flags = np.array(unpack('>' + 'i' * num_aer_wavelengths, 
                                                               f.read(num_aer_wavelengths * 4)))

        # Read the Aerosol Extinction data
        aerosol_extinction = np.ndarray((num_aer_wavelengths, num_aer_bins))
        aerosol_extinction_error = np.ndarray((num_aer_wavelengths, num_aer_bins))
        aerosol_extinction_qa_flags = np.ndarray((num_aer_wavelengths, num_aer_bins))
        for i in range(num_aer_wavelengths):
            aerosol_extinction[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                    f.read(num_aer_bins * 4)))
            aerosol_extinction_error[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                          f.read(num_aer_bins * 4)))
            aerosol_extinction_qa_flags[i] = np.array(unpack('>' + 'i' * num_aer_bins,
                                                             f.read(num_aer_bins * 4)))

        # Read the Aerosol Extinction Ratio data
        aerosol_spectral_dependence_flag = np.array(unpack('>' + 'f' * num_aer_bins,
                                                           f.read(num_aer_bins * 4)))
        extinction_ratio = np.array(unpack('>' + 'f' * num_aer_bins,
                                           f.read(num_aer_bins * 4)))
        extinction_ratio_error = np.array(unpack('>' + 'f' * num_aer_bins,
                                                 f.read(num_aer_bins * 4)))
        extinction_ratio_qa_flags = np.array(unpack('>' + 'f' * num_aer_bins,
                                                    f.read(num_aer_bins * 4)))

        # Return the data as an xarray dataset
        ds = xr.Dataset(
            {
                'yyyyddd': yyyyddd,
                'mission_time': instrument_time,
                'event_type_spacecraft': event_type_spacecraft,
                'event_type_earth': event_type_earth,
                'beta_angle': beta_angle,
                'event_status_flags': event_status_flags,
                'start_data': start_date,
                'start_time': start_time,
                'start_latitude': start_latitude,
                'start_longitude': start_longitude,
                'start_altitude': start_altitude,
                'end_data': end_date,
                'end_time': end_time,
                'end_latitude': end_latitude,
                'end_longitude': end_longitude,
                'end_altitude': end_altitude,
                'gt_date': (['num_ground_tracks'], gt_date),
                'gt_time': (['num_ground_tracks'], gt_time),
                'gt_latitude': (['num_ground_tracks'], gt_latitude),
                'gt_longitude': (['num_ground_tracks'], gt_longitude),
                'gt_ray_dir': (['num_ground_tracks'], gt_ray_dir),
                'homogeneity': (['altitude'], homogeneity),
                'potential_alt': (['altitude'], potential_altitude),
                'input_temperature': (['altitude'], input_temperature),
                'input_temperature_error': (['altitude'], input_temperature_error),
                'input_pressure': (['altitude'], input_pressure),
                'input_pressure_error': (['altitude'], input_pressure_error),
                'input_tp_source_flags': (['altitude'], input_tp_source_flags),
                'temperature_tropopause': temperature_tropopause,
                'altitude_tropopause': altitude_tropopause,
                'o3_composite': (['altitude'], o3_composite),
                'o3_composite_error': (['altitude'], o3_composite_error),
                'o3_composite_slant_path': (['altitude'], o3_composite_slant_path),
                'o3_composite_slant_path_error': (['altitude'], o3_composite_slant_path_error),
                'o3_composite_qa_flags': (['altitude'], o3_composite_qa_flags),
                'o3_mesospheric': (['altitude'], o3_mesospheric),
                'o3_mesospheric_error': (['altitude'], o3_mesospheric_error),
                'o3_mesospheric_slant_path': (['altitude'], o3_mesospheric_slant_path),
                'o3_mesospheric_slant_path_error': (['altitude'], o3_mesospheric_slant_path_error),
                'o3_mesospheric_qa_flags': (['altitude'], o3_mesospheric_qa_flags),
                'o3_mlr': (['altitude'], o3_mlr),
                'o3_mlr_error': (['altitude'], o3_mlr_error),
                'o3_mlr_slant_path': (['altitude'], o3_mlr_slant_path),
                'o3_mlr_slant_path_error': (['altitude'], o3_mlr_slant_path_error),
                'o3_mlr_qa_flags': (['altitude'], o3_mlr_qa_flags),
                'o3': (['altitude'], o3),
                'o3_error': (['altitude'], o3_error),
                'o3_slant_path': (['altitude'], o3_slant_path),
                'o3_slant_path_error': (['altitude'], o3_slant_path_error),
                'o3_qa_flags': (['altitude'], o3_qa_flags),
                'water_vapor': (['altitude'], water_vapor),
                'water_vapor_error': (['altitude'], water_vapor_error),
                'water_vapor_qa_flags': (['altitude'], water_vapor_qa_flags),
                'no2': (['altitude'], no2),
                'no2_error': (['altitude'], no2_error),
                'no2_slant_path': (['altitude'], no2_slant_path),
                'no2_slant_path_error': (['altitude'], no2_slant_path_error),
                'no2_qa_flags': (['altitude'], no2_qa_flags),
                'temperature': (['altitude'], temperature),
                'temperature_error': (['altitude'], temperature_error),
                'pressure': (['altitude'], pressure),
                'pressure_error': (['altitude'], pressure_error),
                'tp_qa_flags': (['altitude'], tp_qa_flags),
                'Half–Bandwidths of Aerosol Channels': (['Aerosol_wavelengths'], aerosol_half_bandwidths),
                'stratospheric_optical_depth': (['Aerosol_wavelengths'], stratospheric_optical_depth),
                'stratospheric_optical_depth_error': (['Aerosol_wavelengths'], stratospheric_optical_depth_error),
                'stratospheric_optical_depth_qa_flags': (['Aerosol_wavelengths'], stratospheric_optical_depth_qa_flags),
                'aerosol_extinction': (['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction),
                'aerosol_extinction_error': (['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction_error),
                'aerosol_extinction_qa_flags': (['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction_qa_flags),
                'aerosol_spectral_dependence_flag': (['Aerosol_altitude'], aerosol_spectral_dependence_flag),
                'extinction_ratio': (['Aerosol_altitude'], extinction_ratio),
                'extinction_ratio_error': (['Aerosol_altitude'], extinction_ratio_error),
                'extinction_ratio_qa_flags': (['Aerosol_altitude'], extinction_ratio_qa_flags)
            },
            coords={
                'profile_id': profile_id,
                'altitude': altitude,
                'Aerosol_wavelengths': aerosol_wavelengths,
                'Aerosol_altitude': altitude[:num_aer_bins]
            },
            attrs={
                'Mission Identification': mission_id,
                'Version: Definitive Orbit Processing': round(L0DO_ver, 5),
                'Version: Level 0 Processing': round(L0_ver, 5),
                'Version: Software Processing': round(software_ver, 5),
                'Version: Data Product': round(dataproduct_ver, 5),
                'Version: Spectroscopy': round(spectroscopy_ver, 5),
                'Version: GRAM 95': round(gram95_ver, 5),
                'Version: Meteorological': round(met_ver, 5),
                'Altitude–Based Grid Spacing': altitude_spacing
            })

        # Assert dimension lengths are correct
        assert (len(ds.num_ground_tracks) == num_ground_tracks)
        assert (len(ds.altitude) == num_bins)
        assert (len(ds.Aerosol_wavelengths) == num_aer_wavelengths)

        return ds
