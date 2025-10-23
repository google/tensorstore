#include <netcdf.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

static void die(const std::string& msg, int code = -1) {
  std::cerr << "ERROR: " << msg;
  if (code != -1) std::cerr << " (nc_err=" << code << ": " << nc_strerror(code) << ")";
  std::cerr << "\n";
  std::exit(1);
}
static inline std::string trim(std::string s) {
  auto notspace = [](int ch){ return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
  return s;
}
static std::tuple<long,long,long,bool> parse_slice_item(const std::string& spec) {
  std::string t = trim(spec);
  if (t == ":" || t.empty()) return {0, -1, 1, true};
  std::vector<std::string> parts; std::stringstream ss(t); std::string tok;
  while (std::getline(ss, tok, ':')) parts.push_back(tok);
  auto to_long = [](const std::string& s)->long{
    char* end=nullptr; long v = std::strtol(s.c_str(), &end, 10);
    if (end==s.c_str()) die("Bad integer in slice: '" + s + "'");
    return v;
  };
  if (parts.size()==1) { long a = to_long(parts[0]); return {a, a+1, 1, false}; }
  if (parts.size()==2) {
    long a = parts[0].empty()? 0 : to_long(parts[0]);
    long b = parts[1].empty()? -1: to_long(parts[1]);
    return {a, b, 1, false};
  }
  if (parts.size()==3) {
    long a = parts[0].empty()? 0 : to_long(parts[0]);
    long b = parts[1].empty()? -1: to_long(parts[1]);
    long s = parts[2].empty()? 1 : to_long(parts[2]);
    if (s==0) die("Slice step cannot be 0");
    return {a, b, s, false};
  }
  die("Invalid slice item: '" + spec + "'");
  return {0,0,1,false};
}
static void parse_slices(const std::string& slice_str, const std::vector<size_t>& shape,
                         std::vector<size_t>& start, std::vector<size_t>& count,
                         std::vector<ptrdiff_t>& stride) {
  const int rank = static_cast<int>(shape.size());
  start.assign(rank, 0); count.assign(rank, 0); stride.assign(rank, 1);
  if (slice_str.empty()) { for (int i=0;i<rank;++i) count[i]=shape[i]; return; }
  std::vector<std::string> items; std::stringstream ss(slice_str); std::string tok;
  while (std::getline(ss, tok, ',')) items.push_back(trim(tok));
  if ((int)items.size() != rank) die("Slice spec rank mismatch");
  for (int i=0;i<rank;++i) {
    auto [a,b,s,infer_full] = parse_slice_item(items[i]);
    if (infer_full) { start[i]=0; count[i]=shape[i]; stride[i]=1; continue; }
    if (a < 0) a += (long)shape[i];
    if (b < 0) b  = (long)shape[i];
    if (a < 0 || (size_t)a >= shape[i]) die("Slice start OOB at dim " + std::to_string(i));
    if (b < (long)a) die("Slice end < start at dim " + std::to_string(i));
    if ((size_t)b > shape[i]) b = (long)shape[i];
    if (s <= 0) die("Slice step must be positive");
    start[i]  = (size_t)a; stride[i] = (ptrdiff_t)s;
    size_t n = 0; for (long x=a; x<b; x+=s) ++n; count[i]=n;
  }
}
template <class T> static std::string vec_to_str(const std::vector<T>& v) {
  std::ostringstream os; os << "["; for (size_t i=0;i<v.size();++i){ if(i) os<<", "; os<<v[i]; } os<<"]"; return os.str();
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage:\n  " << argv[0] << " <file.nc> <variable> [slice_spec]\n";
    return 2;
  }
  std::string path = argv[1], var = argv[2], slice = (argc==4)? argv[3] : "";

  int ncid=-1; int st = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (st!=NC_NOERR) die("open failed: " + path, st);
  int varid=-1; st = nc_inq_varid(ncid, var.c_str(), &varid);
  if (st!=NC_NOERR) die("var not found: " + var, st);

  nc_type xtype; int ndims=0, natts=0;
  st = nc_inq_var(ncid, varid, nullptr, &xtype, &ndims, nullptr, &natts);
  if (st!=NC_NOERR) die("nc_inq_var failed", st);
  std::vector<int> dimids(ndims); st = nc_inq_vardimid(ncid, varid, dimids.data());
  if (st!=NC_NOERR) die("nc_inq_vardimid failed", st);

  std::vector<size_t> shape(ndims);
  for (int i=0;i<ndims;++i) { size_t len=0; st = nc_inq_dimlen(ncid, dimids[i], &len); if (st!=NC_NOERR) die("nc_inq_dimlen failed", st); shape[i]=len; }

  std::string dtype; size_t elem_size=0;
  switch (xtype) {
    case NC_INT:    dtype="int32";   elem_size=4; break;
    case NC_FLOAT:  dtype="float32"; elem_size=4; break;
    case NC_DOUBLE: dtype="float64"; elem_size=8; break;
    default: die("Demo supports int32/float32/float64 only — extend if needed.");
  }

  std::cout << "File: "  << path << "\n";
  std::cout << "Var : "  << var  << "\n";
  std::cout << "Rank: "  << ndims << "\n";
  std::cout << "DType: " << dtype << " (" << elem_size << " bytes)\n";
  std::cout << "Shape: " << vec_to_str(shape) << "  (C-order)\n";

  std::vector<size_t> start, count; std::vector<ptrdiff_t> stride;
  parse_slices(slice, shape, start, count, stride);
  std::cout << "Slice start : " << vec_to_str(start)  << "\n";
  std::cout << "Slice count : " << vec_to_str(count)  << "\n";
  std::cout << "Slice stride: " << vec_to_str(stride) << "\n";

  size_t nelem = 1; for (int i=0;i<ndims;++i) nelem *= count[i];
  std::cout << "Elements out: " << nelem << "\n";
  if (nelem==0) { std::cout << "(empty selection)\n"; nc_close(ncid); return 0; }
  bool has_stride=false; for (auto s: stride) if (s!=1) { has_stride=true; break; }

  int err=0;
  auto print_f32 = [&](const float* a){ for (size_t i=0;i<nelem;++i){ if(i) std::cout<<" "; std::cout<<a[i]; } std::cout<<"\n"; };
  auto print_f64 = [&](const double* a){ for (size_t i=0;i<nelem;++i){ if(i) std::cout<<" "; std::cout<<a[i]; } std::cout<<"\n"; };
  auto print_i32 = [&](const int* a){ for (size_t i=0;i<nelem;++i){ if(i) std::cout<<" "; std::cout<<a[i]; } std::cout<<"\n"; };

  if (xtype==NC_FLOAT) {
    std::vector<float> buf(nelem);
    if (has_stride) err = nc_get_varm_float(ncid, varid, start.data(), count.data(), stride.data(), nullptr, buf.data());
    else            err = nc_get_vara_float(ncid, varid, start.data(), count.data(), buf.data());
    if (err!=NC_NOERR) die("read failed", err); print_f32(buf.data());
  } else if (xtype==NC_DOUBLE) {
    std::vector<double> buf(nelem);
    if (has_stride) err = nc_get_varm_double(ncid, varid, start.data(), count.data(), stride.data(), nullptr, buf.data());
    else            err = nc_get_vara_double(ncid, varid, start.data(), count.data(), buf.data());
    if (err!=NC_NOERR) die("read failed", err); print_f64(buf.data());
  } else if (xtype==NC_INT) {
    std::vector<int> buf(nelem);
    if (has_stride) err = nc_get_varm_int(ncid, varid, start.data(), count.data(), stride.data(), nullptr, buf.data());
    else            err = nc_get_vara_int(ncid, varid, start.data(), count.data(), buf.data());
    if (err!=NC_NOERR) die("read failed", err); print_i32(buf.data());
  }

  nc_close(ncid);
  return 0;
}
