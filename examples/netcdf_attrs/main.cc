#include <netcdf.h>
#include <iostream>
#include <vector>
#include <string>

static void fail(int rc, const char* where){
  std::cerr << where << ": " << nc_strerror(rc) << "\n";
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " FILE VAR\n";
    return 1;
  }
  std::string path = argv[1];
  std::string var  = argv[2];

  int ncid; int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc) fail(rc, "nc_open");

  int varid; rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if (rc) fail(rc, "nc_inq_varid");

  int natts=0; rc = nc_inq_varnatts(ncid, varid, &natts);
  if (rc) fail(rc, "nc_inq_varnatts");

  std::cout << "Attributes for " << var << " (" << natts << "):\n";
  for (int i=0;i<natts;i++){
    char name[NC_MAX_NAME+1]={0};
    rc = nc_inq_attname(ncid, varid, i, name);
    if (rc) fail(rc,"nc_inq_attname");

    nc_type t=NC_NAT; size_t len=0;
    rc = nc_inq_att(ncid, varid, name, &t, &len);
    if (rc) fail(rc,"nc_inq_att");

    std::cout << " - " << name << " = ";

    if (t==NC_CHAR || t==NC_STRING){
      std::vector<char> buf(len+1,0);
      rc = nc_get_att_text(ncid, varid, name, buf.data());
      if (rc) fail(rc,"nc_get_att_text");
      std::cout << buf.data();
    } else if (t==NC_DOUBLE){
      std::vector<double> v(len);
      rc = nc_get_att_double(ncid, varid, name, v.data());
      if (rc) fail(rc,"nc_get_att_double");
      for(size_t j=0;j<len;j++){ std::cout << v[j] << (j+1<len?", ":""); }
    } else if (t==NC_FLOAT){
      std::vector<float> v(len);
      rc = nc_get_att_float(ncid, varid, name, v.data());
      if (rc) fail(rc,"nc_get_att_float");
      for(size_t j=0;j<len;j++){ std::cout << v[j] << (j+1<len?", ":""); }
    } else if (t==NC_INT){
      std::vector<int> v(len);
      rc = nc_get_att_int(ncid, varid, name, v.data());
      if (rc) fail(rc,"nc_get_att_int");
      for(size_t j=0;j<len;j++){ std::cout << v[j] << (j+1<len?", ":""); }
    } else {
      std::cout << "(unhandled type)";
    }
    std::cout << "\n";
  }

  rc = nc_close(ncid); if (rc) fail(rc,"nc_close");
  return 0;
}
