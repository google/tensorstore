#include <map>
#include <string>
#include <cstring>
#include <tuple>
#include <vector>


class OmeXml
{
public:
    size_t nc, nz, nt;
    short dim_order;
    std::map<std::string, std::string> xml_metadata_map;
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> tiff_data_list;

    OmeXml();
    void ParseOmeXml(char* buf);
    std::string ToJsonStr();
    
};
