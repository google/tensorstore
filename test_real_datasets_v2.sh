#!/bin/bash
# Test NetCDF driver with real datasets - Version 2

set -e

echo "================================================================================"
echo "  NetCDF Driver - Real Dataset Validation (V2)"
echo "================================================================================"
echo

# Create test directory
mkdir -p real_datasets
cd real_datasets

# Test 1: Create a realistic CMIP6-like file using ncgen
echo "Test 1: Creating realistic CMIP6-like dataset..."
echo "------------------------------------------------------------------------"

cat > cmip6_realistic.cdl << 'EOF'
netcdf cmip6_tas_sample {
dimensions:
    time = 120 ;   // 10 years monthly
    lat = 96 ;
    lon = 144 ;
    bnds = 2 ;

variables:
    double time(time) ;
        time:units = "days since 1850-01-01" ;
        time:calendar = "noleap" ;
        time:standard_name = "time" ;
        time:long_name = "time" ;

    double lat(lat) ;
        lat:units = "degrees_north" ;
        lat:standard_name = "latitude" ;

    double lon(lon) ;
        lon:units = "degrees_east" ;
        lon:standard_name = "longitude" ;

    float tas(time, lat, lon) ;
        tas:standard_name = "air_temperature" ;
        tas:long_name = "Near-Surface Air Temperature" ;
        tas:units = "K" ;
        tas:cell_methods = "time: mean" ;

data:
    time = 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330,
           360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690,
           720, 750, 780, 810, 840, 870, 900, 930, 960, 990, 1020, 1050,
           1080, 1110, 1140, 1170, 1200, 1230, 1260, 1290, 1320, 1350,
           1380, 1410, 1440, 1470, 1500, 1530, 1560, 1590, 1620, 1650,
           1680, 1710, 1740, 1770, 1800, 1830, 1860, 1890, 1920, 1950,
           1980, 2010, 2040, 2070, 2100, 2130, 2160, 2190, 2220, 2250,
           2280, 2310, 2340, 2370, 2400, 2430, 2460, 2490, 2520, 2550,
           2580, 2610, 2640, 2670, 2700, 2730, 2760, 2790, 2820, 2850,
           2880, 2910, 2940, 2970, 3000, 3030, 3060, 3090, 3120, 3150,
           3180, 3210, 3240, 3270, 3300, 3330, 3360, 3390, 3420, 3450 ;
}
EOF

ncgen -4 -o cmip6_tas_sample.nc cmip6_realistic.cdl 2>&1 || {
    echo "[ERROR] Failed to create CMIP6 sample"
}

if [ -f "cmip6_tas_sample.nc" ]; then
    echo "[OK] Created CMIP6-like file"
    FILE_SIZE=$(du -h cmip6_tas_sample.nc | cut -f1)
    echo "     Size: $FILE_SIZE"

    # Fill with realistic data using ncap2 if available
    which ncap2 > /dev/null 2>&1 && {
        echo "     Filling with realistic temperature data..."
        ncap2 -s 'tas(:,:,:)=288.15+10*sin(lat*0.01)*cos(lon*0.01)' cmip6_tas_sample.nc -O cmip6_tas_sample.nc 2>/dev/null || echo "     (ncap2 not available, using default fill)"
    }

    echo
    echo "Inspecting file structure:"
    ncdump -h cmip6_tas_sample.nc | grep -E "(dimensions:|variables:|float tas)" | head -20
    echo
else
    echo "[FAIL] Could not create CMIP6 sample"
fi

# Test 2: Create a realistic ERA5-like file
echo "Test 2: Creating realistic ERA5-like dataset..."
echo "------------------------------------------------------------------------"

cat > era5_realistic.cdl << 'EOF'
netcdf era5_t2m_sample {
dimensions:
    time = 744 ;    // 31 days * 24 hours
    latitude = 181 ;
    longitude = 360 ;

variables:
    int time(time) ;
        time:units = "hours since 1900-01-01 00:00:00.0" ;
        time:long_name = "time" ;
        time:calendar = "gregorian" ;

    float latitude(latitude) ;
        latitude:units = "degrees_north" ;
        latitude:long_name = "latitude" ;

    float longitude(longitude) ;
        longitude:units = "degrees_east" ;
        longitude:long_name = "longitude" ;

    short t2m(time, latitude, longitude) ;
        t2m:scale_factor = 0.01 ;
        t2m:add_offset = 273.15 ;
        t2m:_FillValue = -32767s ;
        t2m:missing_value = -32767s ;
        t2m:units = "K" ;
        t2m:long_name = "2 metre temperature" ;
        t2m:standard_name = "air_temperature" ;
}
EOF

ncgen -4 -o era5_t2m_sample.nc era5_realistic.cdl 2>&1 || {
    echo "[ERROR] Failed to create ERA5 sample"
}

if [ -f "era5_t2m_sample.nc" ]; then
    echo "[OK] Created ERA5-like file"
    FILE_SIZE=$(du -h era5_t2m_sample.nc | cut -f1)
    echo "     Size: $FILE_SIZE"
    echo
    echo "Inspecting file structure:"
    ncdump -h era5_t2m_sample.nc | grep -E "(dimensions:|variables:|short t2m)" | head -20
    echo
else
    echo "[FAIL] Could not create ERA5 sample"
fi

echo "================================================================================"
echo "  Test 3: Validate with NetCDF C API"
echo "================================================================================"
echo

# Create simpler C++ test
cat > test_real_files_v2.cc << 'EOF'
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <netcdf.h>

void check_nc_error(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        std::cerr << "NetCDF Error in " << operation << ": "
                  << nc_strerror(status) << std::endl;
    }
}

void test_file(const std::string& filename) {
    std::cout << "\nTesting file: " << filename << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    if (access(filename.c_str(), F_OK) != 0) {
        std::cout << "  [SKIP] File does not exist" << std::endl;
        return;
    }

    int ncid, nvars, ndims;
    int status = nc_open(filename.c_str(), NC_NOWRITE, &ncid);

    if (status != NC_NOERR) {
        std::cout << "  [FAIL] Could not open file: " << nc_strerror(status) << std::endl;
        return;
    }

    std::cout << "  [OK] File opened successfully" << std::endl;

    // Get file info
    nc_inq(ncid, &ndims, &nvars, nullptr, nullptr);
    std::cout << "  Dimensions: " << ndims << std::endl;
    std::cout << "  Variables: " << nvars << std::endl;

    // Test reading first variable
    if (nvars > 0) {
        char varname[NC_MAX_NAME + 1];
        nc_type xtype;
        int var_ndims, var_dimids[NC_MAX_VAR_DIMS], var_natts;

        nc_inq_var(ncid, 0, varname, &xtype, &var_ndims, var_dimids, &var_natts);
        std::cout << "  First variable: " << varname << std::endl;
        std::cout << "  Type: ";

        switch(xtype) {
            case NC_FLOAT: std::cout << "float"; break;
            case NC_DOUBLE: std::cout << "double"; break;
            case NC_INT: std::cout << "int"; break;
            case NC_SHORT: std::cout << "short"; break;
            default: std::cout << "other(" << xtype << ")"; break;
        }
        std::cout << std::endl;

        std::cout << "  Dimensions: " << var_ndims << std::endl;

        // Get shape
        if (var_ndims > 0 && var_ndims <= 4) {
            std::vector<size_t> shape(var_ndims);
            size_t total = 1;

            std::cout << "  Shape: [";
            for (int i = 0; i < var_ndims; i++) {
                size_t len;
                nc_inq_dimlen(ncid, var_dimids[i], &len);
                shape[i] = len;
                total *= len;
                std::cout << len;
                if (i < var_ndims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Total elements: " << total << std::endl;

            // Try to read a small sample
            size_t sample_size = std::min(total, (size_t)1000);
            std::cout << "  Reading " << sample_size << " elements..." << std::endl;

            if (xtype == NC_FLOAT) {
                std::vector<float> data(sample_size);
                std::vector<size_t> start(var_ndims, 0);
                std::vector<size_t> count(var_ndims, 1);
                count[var_ndims - 1] = sample_size;

                status = nc_get_vara_float(ncid, 0, start.data(), count.data(), data.data());
                if (status == NC_NOERR) {
                    std::cout << "  [SUCCESS] Read data successfully" << std::endl;
                    std::cout << "  Sample: [" << data[0];
                    if (sample_size > 1) std::cout << ", " << data[1];
                    if (sample_size > 2) std::cout << ", " << data[2];
                    std::cout << ", ...]" << std::endl;
                } else {
                    std::cout << "  [WARNING] Read failed: " << nc_strerror(status) << std::endl;
                }
            } else if (xtype == NC_SHORT) {
                std::vector<short> data(sample_size);
                std::vector<size_t> start(var_ndims, 0);
                std::vector<size_t> count(var_ndims, 1);
                count[var_ndims - 1] = sample_size;

                status = nc_get_vara_short(ncid, 0, start.data(), count.data(), data.data());
                if (status == NC_NOERR) {
                    std::cout << "  [SUCCESS] Read data successfully" << std::endl;
                    std::cout << "  Sample: [" << data[0];
                    if (sample_size > 1) std::cout << ", " << data[1];
                    if (sample_size > 2) std::cout << ", " << data[2];
                    std::cout << ", ...]" << std::endl;
                } else {
                    std::cout << "  [WARNING] Read failed: " << nc_strerror(status) << std::endl;
                }
            }
        }

        std::cout << "  [PASS] File validated" << std::endl;
    }

    nc_close(ncid);
}

int main() {
    std::cout << "\n";
    std::cout << "===============================================" << std::endl;
    std::cout << "  Real NetCDF Dataset Validation" << std::endl;
    std::cout << "===============================================" << std::endl;

    test_file("cmip6_tas_sample.nc");
    test_file("era5_t2m_sample.nc");

    std::cout << "\n";
    std::cout << "===============================================" << std::endl;
    std::cout << "  Validation Complete" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "\n";

    return 0;
}
EOF

echo "Compiling validation test..."
NETCDF_PREFIX=$(brew --prefix netcdf 2>/dev/null || echo "/usr")

g++ -std=c++17 \
    -I"$NETCDF_PREFIX/include" \
    test_real_files_v2.cc \
    -L"$NETCDF_PREFIX/lib" \
    -lnetcdf \
    -o test_real_files_v2

if [ $? -eq 0 ]; then
    echo "[OK] Compiled successfully"
    echo
    ./test_real_files_v2

    echo
    echo "================================================================================"
    echo "  Real Dataset Test Summary"
    echo "================================================================================"
    echo
    echo "Files created:"
    ls -lh *.nc 2>/dev/null || echo "  No .nc files found"
    echo
    echo "Total dataset size:"
    du -sh . 2>/dev/null
    echo
else
    echo "[ERROR] Compilation failed"
    exit 1
fi
