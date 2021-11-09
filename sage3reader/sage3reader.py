from pathlib import Path
import numpy as np
from struct import unpack
import xarray as xr
from datetime import datetime
from glob import glob


def yyyymmddhhmmss_to_datetime(yyyymmdd: int, hhmmss: int) -> datetime:
    """ Convert an integer pair of yyyymmdd and hhmmss to a useful time format.
    """
    return datetime(
        yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100,
        hhmmss // 10000, (hhmmss // 100) % 100, hhmmss % 100)


def l2_v5_0_binary_to_dataset(file) -> xr.Dataset:
    """
    Read the Level 2 Solar Event Species Profiles for a version 5 SAGE III or SAGE III ISS binary file.
    https://eosweb.larc.nasa.gov/sites/default/files/project/sage3/guide/Data_Product_User_Guide.pdf
    """

    # Read all the data into memory
    with open(file, 'rb') as f:
        # Read the File Header
        (event_id, yyyyddd, instrument_time, fill_value_int, fill_value_float, mission_id) = \
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
        gt_date = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.int32)
        gt_time = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.int32)
        gt_latitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        gt_longitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        gt_ray_dir = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)

        # Read Profile Altitude Levels data
        homogeneity = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)
        altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        potential_altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)

        # Read the Input Temp/Pres for Retrievals
        input_temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_tp_source_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Derived Tropopause data
        (temperature_tropopause, altitude_tropopause) = unpack('>ff', f.read(2 * 4))

        # Read the Composite Ozone data
        o3_composite = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Mesospheric Ozone data
        o3_mesospheric = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the MLR Ozone data
        o3_mlr = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Ozone Least Squares data
        o3 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read Water Vapor data
        water_vapor = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        water_vapor_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        water_vapor_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the NO2 data
        no2 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_slant_path = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_slant_path_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Retrieved T/P data
        temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        tp_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Aerosol Information
        aerosol_wavelengths = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                              f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        aerosol_half_bandwidths = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                  f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                      f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth_error = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                            f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth_qa_flags = np.array(unpack('>' + 'i' * num_aer_wavelengths,
                                                               f.read(num_aer_wavelengths * 4)), dtype=np.int32)

        # Read the Aerosol Extinction data
        aerosol_extinction = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.float32)
        aerosol_extinction_error = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.float32)
        aerosol_extinction_qa_flags = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.int32)
        for i in range(num_aer_wavelengths):
            aerosol_extinction[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                    f.read(num_aer_bins * 4)))
            aerosol_extinction_error[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                          f.read(num_aer_bins * 4)))
            aerosol_extinction_qa_flags[i] = np.array(unpack('>' + 'i' * num_aer_bins,
                                                             f.read(num_aer_bins * 4)))

        # Read the Aerosol Extinction Ratio data
        aerosol_spectral_dependence_flag = np.array(unpack('>' + 'i' * num_aer_bins,
                                                           f.read(num_aer_bins * 4)), dtype=np.int32)
        extinction_ratio = np.array(unpack('>' + 'f' * num_aer_bins,
                                           f.read(num_aer_bins * 4)), dtype=np.float32)
        extinction_ratio_error = np.array(unpack('>' + 'f' * num_aer_bins,
                                                 f.read(num_aer_bins * 4)), dtype=np.float32)
        extinction_ratio_qa_flags = np.array(unpack('>' + 'i' * num_aer_bins,
                                                    f.read(num_aer_bins * 4)), dtype=np.int32)

        # Convert date and time pairs to a single datetime.
        start_datetime = yyyymmddhhmmss_to_datetime(start_date, start_time)
        end_datetime = yyyymmddhhmmss_to_datetime(end_date, end_time)
        gt_datetime = [yyyymmddhhmmss_to_datetime(date, time) if date != fill_value_int else np.datetime64('NaT')
                       for (date, time) in zip(gt_date, gt_time)]

        # Return the data as an xarray dataset
        ds = xr.Dataset(
            {
                'yyyyddd': np.int32(yyyyddd),
                'mission_time': np.float32(instrument_time),
                'event_type_spacecraft': np.int32(event_type_spacecraft),
                'event_type_earth': np.int32(event_type_earth),
                'beta_angle': np.float32(beta_angle),
                'event_status_flags': np.int32(event_status_flags),
                'start_time': start_datetime,
                'start_latitude': np.float32(start_latitude),
                'start_longitude': np.float32(start_longitude),
                'start_altitude': np.float32(start_altitude),
                'end_time': end_datetime,
                'end_latitude': np.float32(end_latitude),
                'end_longitude': np.float32(end_longitude),
                'end_altitude': np.float32(end_altitude),
                'gt_time': (['num_ground_tracks'], gt_datetime),
                'gt_latitude': (['num_ground_tracks'], gt_latitude),
                'gt_longitude': (['num_ground_tracks'], gt_longitude),
                'gt_ray_dir': (['num_ground_tracks'], gt_ray_dir),
                'homogeneity': (['altitude'], np.int32(homogeneity)),
                'potential_alt': (['altitude'], potential_altitude),
                'input_temperature': (['altitude'], input_temperature),
                'input_temperature_error': (['altitude'], input_temperature_error),
                'input_pressure': (['altitude'], input_pressure),
                'input_pressure_error': (['altitude'], input_pressure_error),
                'input_tp_source_flags': (['altitude'], input_tp_source_flags),
                'temperature_tropopause': np.float32(temperature_tropopause),
                'altitude_tropopause': np.float32(altitude_tropopause),
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
                'Half-Bandwidths of Aerosol Channels': (['Aerosol_wavelengths'], aerosol_half_bandwidths),
                'stratospheric_optical_depth': (['Aerosol_wavelengths'], stratospheric_optical_depth),
                'stratospheric_optical_depth_error': (['Aerosol_wavelengths'], stratospheric_optical_depth_error),
                'stratospheric_optical_depth_qa_flags': (['Aerosol_wavelengths'], stratospheric_optical_depth_qa_flags),
                'aerosol_extinction': (['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction),
                'aerosol_extinction_error': (['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction_error),
                'aerosol_extinction_qa_flags': (
                ['Aerosol_wavelengths', 'Aerosol_altitude'], aerosol_extinction_qa_flags),
                'aerosol_spectral_dependence_flag': (['Aerosol_altitude'], aerosol_spectral_dependence_flag),
                'extinction_ratio': (['Aerosol_altitude'], extinction_ratio),
                'extinction_ratio_error': (['Aerosol_altitude'], extinction_ratio_error),
                'extinction_ratio_qa_flags': (['Aerosol_altitude'], extinction_ratio_qa_flags)
            },
            coords={
                'event_id': np.int32(event_id),
                'altitude': altitude,
                'Aerosol_wavelengths': aerosol_wavelengths,
                'Aerosol_altitude': altitude[:num_aer_bins]
            },
            attrs={
                'Mission Identification': mission_id,
                'Version: Definitive Orbit Processing': np.float32(L0DO_ver),
                'Version: Level 0 Processing': np.float32(L0_ver),
                'Version: Software Processing': np.float32(software_ver),
                'Version: Data Product': np.float32(dataproduct_ver),
                'Version: Spectroscopy': np.float32(spectroscopy_ver),
                'Version: GRAM 95': np.float32(gram95_ver),
                'Version: Meteorological': np.float32(met_ver),
                'Altitude-Based Grid Spacing': altitude_spacing,
                '_FillValue': fill_value_int
            })

        # Assert dimension lengths are correct
        assert (len(ds.num_ground_tracks) == num_ground_tracks)
        assert (len(ds.altitude) == num_bins)
        assert (len(ds.Aerosol_wavelengths) == num_aer_wavelengths)

        for var in ds.variables:
            if np.issubdtype(ds[var].dtype, np.floating) and ds[var].size > 1:
                ds[var] = ds[var].where(ds[var] != fill_value_float)

        ds = ds.set_coords(
            ['yyyyddd', 'mission_time', 'start_time', 'start_latitude', 'start_longitude', 'start_altitude', 'end_time',
             'end_latitude', 'end_longitude', 'end_altitude', ])

        return ds


def l2_v5_1_5_2_binary_to_dataset(file) -> xr.Dataset:
    """
    Read the Level 2 Solar Event Species Profiles for a version 5.1/5.2 SAGE III or SAGE III ISS binary file.
    https://eosweb.larc.nasa.gov/sites/default/files/project/sage3/guide/Data_Product_User_Guide.pdf
    """
    print(file)
    # Read all the data into memory
    with open(file, 'rb') as f:
        # Read the File Header

        event_str = f.read(12)

        (_, date, fraction_time, latitude, longitude, time, fill_value_int, fill_value_float, mission_id) = \
            unpack('>iifffiifi', f.read(9 * 4))

        # Read the Version Tracking data
        (L0DO_ver, CCD_ver, L0_ver, software_ver, dataproduct_ver, spectroscopy_ver, gram95_ver, met_ver) = \
            unpack('>fiffffff', f.read(8 * 4))

        # Read the File Description
        (altitude_spacing, num_bins, num_met_grid, num_aer_wavelengths, num_ground_tracks, num_aer_bins) = \
            unpack('>fiiiii', f.read(6 * 4))

        # Read the Event Type data
        (event_type_spacecraft, event_type_earth, beta_angle, aurora_flag, ephemeris_source) = \
            unpack('>iifii', f.read(5 * 4))

        # Read Ground Track Information
        gt_date = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.int32)
        gt_time = np.array(unpack('>' + 'i' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.int32)
        gt_latitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        gt_longitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        gt_ray_dir = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)

        # Read Space Craft Information
        sc_latitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        sc_longitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)
        sc_altitude = np.array(unpack('>' + 'f' * num_ground_tracks, f.read(num_ground_tracks * 4)), dtype=np.float32)

        # Read Profile Altitude Levels data
        homogeneity = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)
        altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        geopotential_altitude = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)

        # Read the Input Temp/Pres/ND for Retrievals
        input_temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_neutral_density = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_neutral_density_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        input_tp_source_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Derived Tropopause data
        (temperature_tropopause, altitude_tropopause, pressure_tropopause) = unpack('>fff', f.read(3 * 4))

        # Read the Met Temp/Press profiles
        met_pressure = np.array(unpack('>' + 'f' * num_met_grid, f.read(num_met_grid * 4)), dtype=np.float32)
        met_temperature = np.array(unpack('>' + 'f' * num_met_grid, f.read(num_met_grid * 4)), dtype=np.float32)
        met_temperature_error = np.array(unpack('>' + 'f' * num_met_grid, f.read(num_met_grid * 4)), dtype=np.float32)
        met_altitude = np.array(unpack('>' + 'f' * num_met_grid, f.read(num_met_grid * 4)), dtype=np.float32)
        (met_source, ) = unpack('>i', f.read(4))

        # Read the CCD and Bit flags
        (ccd_temperature, spectrometer_zenith_temperature, ccd_temperature_minus_tec, ephemeris_quality, speccalshift,
         speccalstretch) = unpack('>ffffff', f.read(6 * 4))

        azimuth_angle_start, azimuth_angle_end = unpack('>ff', f.read(2 * 4))

        qaflag, = unpack('>i', f.read(4))

        qaflag_altitude = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Composite Ozone data
        o3_composite = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_composite_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Mesospheric Ozone data
        o3_mesospheric = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mesospheric_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the MLR Ozone data
        o3_mlr = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_mlr_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Ozone Least Squares data
        o3 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        o3_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read Water Vapor data
        water_vapor = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        water_vapor_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        water_vapor_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the NO2 data
        no2 = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        no2_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Retrieved T/P data
        temperature = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        temperature_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        pressure = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        pressure_error = np.array(unpack('>' + 'f' * num_bins, f.read(num_bins * 4)), dtype=np.float32)
        tp_qa_flags = np.array(unpack('>' + 'i' * num_bins, f.read(num_bins * 4)), dtype=np.int32)

        # Read the Aerosol Information
        aerosol_wavelengths = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                              f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        aerosol_half_bandwidths = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                  f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        molecular_sct = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                        f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        molecular_sct_uncert = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                               f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                      f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth_error = np.array(unpack('>' + 'f' * num_aer_wavelengths,
                                                            f.read(num_aer_wavelengths * 4)), dtype=np.float32)
        stratospheric_optical_depth_qa_flags = np.array(unpack('>' + 'i' * num_aer_wavelengths,
                                                               f.read(num_aer_wavelengths * 4)), dtype=np.int32)

        # Read the Aerosol Extinction data
        aerosol_extinction = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.float32)
        aerosol_extinction_error = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.float32)
        aerosol_extinction_qa_flags = np.ndarray((num_aer_wavelengths, num_aer_bins), dtype=np.int32)
        for i in range(num_aer_wavelengths):
            aerosol_extinction[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                    f.read(num_aer_bins * 4)))
            aerosol_extinction_error[i] = np.array(unpack('>' + 'f' * num_aer_bins,
                                                          f.read(num_aer_bins * 4)))
            aerosol_extinction_qa_flags[i] = np.array(unpack('>' + 'i' * num_aer_bins,
                                                             f.read(num_aer_bins * 4)))

        # Assert file is fully read.
        assert (f.tell() == f.seek(0, 2))

        # Decode QABits
        (HexErrFlag, ContWindowClosedFlag, TimeQualFlag, DMPExoFlag, BlockExoFlag, SpectCalFlag, SolarEclipseFlag) = \
            (lambda x: [x >> bit & 1 for bit in range(7)])(qaflag)

        event_datetime = yyyymmddhhmmss_to_datetime(date, time) if date != fill_value_int else np.datetime64('NaT')
        gt_datetime = [yyyymmddhhmmss_to_datetime(date, time) if date != fill_value_int else np.datetime64('NaT')
                       for (date, time) in zip(gt_date, gt_time)]

        # Return the data as an xarray dataset
        ds = xr.Dataset(
            {
                'time': event_datetime,
                'year_fraction': np.float32(fraction_time),
                'latitude': np.float32(latitude),
                'longitude': np.float32(longitude),
                'event_type_spacecraft': np.int32(event_type_spacecraft),
                'event_type_earth': np.int32(event_type_earth),
                'beta_angle': np.float32(beta_angle),
                'azimuth_angle_start': np.float32(azimuth_angle_start),
                'azimuth_angle_end': np.float32(azimuth_angle_end),
                'aurora_flag': np.int32(aurora_flag),
                'ephemeris source': np.int32(ephemeris_source),
                'gt_time': (['num_ground_tracks'], gt_datetime),
                'gt_latitude': (['num_ground_tracks'], gt_latitude),
                'gt_longitude': (['num_ground_tracks'], gt_longitude),
                'gt_ray_dir': (['num_ground_tracks'], gt_ray_dir),
                'sc_latitude':  (['num_ground_tracks'], sc_latitude),
                'sc_longitude':  (['num_ground_tracks'], sc_longitude),
                'sc_altitude':  (['num_ground_tracks'], sc_altitude),
                'homogeneity': (['altitude'], np.int32(homogeneity)),
                'geopotential_alt': (['altitude'], geopotential_altitude),
                'input_temperature': (['altitude'], input_temperature),
                'input_temperature_error': (['altitude'], input_temperature_error),
                'input_pressure': (['altitude'], input_pressure),
                'input_pressure_error': (['altitude'], input_pressure_error),
                'input_neutral_density': (['altitude'], input_neutral_density),
                'input_neutral_density_error': (['altitude'], input_neutral_density_error),
                'input_tp_source_flags': (['altitude'], input_tp_source_flags),
                'temperature_tropopause': np.float32(temperature_tropopause),
                'altitude_tropopause': np.float32(altitude_tropopause),
                'pressure_tropopause': np.float32(pressure_tropopause),
                'met_temperature': (['met_pressure'], met_temperature),
                'met_temperature_error': (['met_pressure'], met_temperature_error),
                'met_altitude': (['met_pressure'], met_altitude),
                'met_source': np.float32(met_source),
                'CCD_Temperature': np.float32(ccd_temperature),
                'Spectrometer_Zenith_Temperature': np.float32(spectrometer_zenith_temperature),
                'CCD_Temperature_minus_TEC': np.float32(ccd_temperature_minus_tec),
                'Ephemeris_Quality': np.float32(ephemeris_quality),
                'SpecCalShift': np.float32(speccalshift),
                'SpecCalStretch': np.float32(speccalstretch),
                'o3_composite': (['altitude'], o3_composite),
                'o3_composite_error': (['altitude'], o3_composite_error),
                'o3_composite_qa_flags': (['altitude'], o3_composite_qa_flags),
                'o3_mesospheric': (['altitude'], o3_mesospheric),
                'o3_mesospheric_error': (['altitude'], o3_mesospheric_error),
                'o3_mesospheric_qa_flags': (['altitude'], o3_mesospheric_qa_flags),
                'o3_mlr': (['altitude'], o3_mlr),
                'o3_mlr_error': (['altitude'], o3_mlr_error),
                'o3_mlr_qa_flags': (['altitude'], o3_mlr_qa_flags),
                'o3': (['altitude'], o3),
                'o3_error': (['altitude'], o3_error),
                'o3_qa_flags': (['altitude'], o3_qa_flags),
                'water_vapor': (['altitude'], water_vapor),
                'water_vapor_error': (['altitude'], water_vapor_error),
                'water_vapor_qa_flags': (['altitude'], water_vapor_qa_flags),
                'no2': (['altitude'], no2),
                'no2_error': (['altitude'], no2_error),
                'no2_qa_flags': (['altitude'], no2_qa_flags),
                'temperature': (['altitude'], temperature),
                'temperature_error': (['altitude'], temperature_error),
                'pressure': (['altitude'], pressure),
                'pressure_error': (['altitude'], pressure_error),
                'tp_qa_flags': (['altitude'], tp_qa_flags),
                'aerosol_wavelengths': (['Aerosol_channel'], aerosol_wavelengths),
                'Half Bandwidths of Aerosol Channels': (['Aerosol_channel'], aerosol_half_bandwidths),
                'Molecular_SCT': (['Aerosol_channel'], molecular_sct),
                'Molecular_SCT_uncert': (['Aerosol_channel'], molecular_sct_uncert),
                'stratospheric_optical_depth': (['Aerosol_channel'], stratospheric_optical_depth),
                'stratospheric_optical_depth_error': (['Aerosol_channel'], stratospheric_optical_depth_error),
                'stratospheric_optical_depth_qa_flags': (['Aerosol_channel'], stratospheric_optical_depth_qa_flags),
                'aerosol_extinction': (['Aerosol_channel', 'Aerosol_altitude'], aerosol_extinction),
                'aerosol_extinction_error': (['Aerosol_channel', 'Aerosol_altitude'], aerosol_extinction_error),
                'aerosol_extinction_qa_flags': (['Aerosol_channel', 'Aerosol_altitude'],
                                                aerosol_extinction_qa_flags),
                'HexErrFlag': np.int8(HexErrFlag),
                'ContWindowClosedFlag': np.int8(ContWindowClosedFlag),
                'TimeQualFlag': np.int8(TimeQualFlag),
                'DMPExoFlag': np.int8(DMPExoFlag),
                'BlockExoFlag': np.int8(BlockExoFlag),
                'SpectralCalFlag': np.int8(SpectCalFlag),
                'SolarEclipseFlag': np.int8(SolarEclipseFlag),
                'DMPAltFlag': (['altitude'], np.int8(qaflag_altitude))
            },
            coords={
                'event_id': event_str,
                'altitude': altitude,
                'met_pressure': [1.00e+03, 9.75e+02, 9.50e+02, 9.25e+02, 9.00e+02, 8.75e+02, 8.50e+02,
                                 8.25e+02, 8.00e+02, 7.75e+02, 7.50e+02, 7.25e+02, 7.00e+02, 6.50e+02,
                                 6.00e+02, 5.50e+02, 5.00e+02, 4.50e+02, 4.00e+02, 3.50e+02, 3.00e+02,
                                 2.50e+02, 2.00e+02, 1.50e+02, 1.00e+02, 7.00e+01, 5.00e+01, 4.00e+01,
                                 3.00e+01, 2.00e+01, 1.00e+01, 7.00e+00, 5.00e+00, 4.00e+00, 3.00e+00,
                                 2.00e+00, 1.00e+00, 7.00e-01, 5.00e-01, 4.00e-01, 3.00e-01, 1.00e-01],
                'Aerosol_channel': range(num_aer_wavelengths),
                'Aerosol_altitude': altitude[:num_aer_bins]
            },
            attrs={
                'Mission Identification': mission_id,
                'Version: Definitive Orbit Processing': np.float32(L0DO_ver),
                'Version: CCD Table': np.float32(CCD_ver),
                'Version: Level 0 Processing': np.float32(L0_ver),
                'Version: Software Processing': np.float32(software_ver),
                'Version: Data Product': np.float32(dataproduct_ver),
                'Version: Spectroscopy': np.float32(spectroscopy_ver),
                'Version: GRAM 95': np.float32(gram95_ver),
                'Version: Meteorological': np.float32(met_ver),
                'Altitude Based Grid Spacing': altitude_spacing,
                '_FillValue': fill_value_int
            })

        # Assert dimension lengths are correct
        assert(len(ds.num_ground_tracks) == num_ground_tracks)
        assert(len(ds.altitude) == num_bins)
        assert(len(ds.met_pressure) == num_met_grid)
        assert(len(ds.Aerosol_channel) == num_aer_wavelengths)

        for var in ds.variables:
            if np.issubdtype(ds[var].dtype, np.floating) and ds[var].size > 1:
                ds[var] = ds[var].where(ds[var] != fill_value_float)

        ds = ds.set_coords(['time', 'latitude', 'longitude'])
        return ds


def multi_path_l2binary_to_dataset(path, version='5.10'):
    files = sorted(Path(path).rglob('*.*'))
    if version in ['5.1', '5.10', '5.2', '5.20']:
        ds = xr.concat([l2_v5_1_5_2_binary_to_dataset(file) for file in files], dim='event_id')
    elif version in ['5.0', '5.00']:
        ds = xr.concat([l2_v5_0_binary_to_dataset(file) for file in files], dim='event_id')
    return ds.sortby('event_id')
